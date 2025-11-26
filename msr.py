import json
import os
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv
import duckdb
import backoff
import requests
import requests.exceptions
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
import traceback
import subprocess
import re

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Parent directory

AGENTS_REPO = "SWE-Arena/bot_data"
AGENTS_REPO_LOCAL_PATH = os.path.join(BASE_DIR, "bot_data")  # Local git clone path
DUCKDB_CACHE_FILE = os.path.join(SCRIPT_DIR, "cache.duckdb")
GHARCHIVE_DATA_LOCAL_PATH = os.path.join(BASE_DIR, "gharchive/data")
LEADERBOARD_FILENAME = f"{os.getenv('COMPOSE_PROJECT_NAME')}.json"
LEADERBOARD_REPO = "SWE-Arena/leaderboard_data"
LEADERBOARD_TIME_FRAME_DAYS = 180

# Git sync configuration (mandatory to get latest bot data)
GIT_SYNC_TIMEOUT = 300  # 5 minutes timeout for git pull

# OPTIMIZED DUCKDB CONFIGURATION
DUCKDB_THREADS = 16
DUCKDB_MEMORY_LIMIT = "128GB"

# Streaming batch configuration
BATCH_SIZE_DAYS = 7  # Process 1 week at a time (~168 hourly files)
# At this size: ~7 days × 24 files × ~100MB per file = ~16GB uncompressed per batch

# Download configuration
DOWNLOAD_WORKERS = 4
DOWNLOAD_RETRY_DELAY = 2
MAX_RETRIES = 5

# Upload configuration
UPLOAD_DELAY_SECONDS = 5
UPLOAD_MAX_BACKOFF = 3600

# Scheduler configuration
SCHEDULE_ENABLED = False
SCHEDULE_DAY_OF_WEEK = 'tue'  # Tuesday
SCHEDULE_HOUR = 0
SCHEDULE_MINUTE = 0
SCHEDULE_TIMEZONE = 'UTC'

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_jsonl(filename):
    """Load JSONL file and return list of dictionaries."""
    if not os.path.exists(filename):
        return []

    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data


def save_jsonl(filename, data):
    """Save list of dictionaries to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def normalize_date_format(date_string):
    """Convert date strings or datetime objects to standardized ISO 8601 format with Z suffix."""
    if not date_string or date_string == 'N/A':
        return 'N/A'

    try:
        if isinstance(date_string, datetime):
            return date_string.strftime('%Y-%m-%dT%H:%M:%SZ')

        date_string = re.sub(r'\s+', ' ', date_string.strip())
        date_string = date_string.replace(' ', 'T')

        if len(date_string) >= 3:
            if date_string[-3:-2] in ('+', '-') and ':' not in date_string[-3:]:
                date_string = date_string + ':00'

        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(f"Warning: Could not parse date '{date_string}': {e}")
        return date_string


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


# =============================================================================
# GHARCHIVE DOWNLOAD FUNCTIONS
# =============================================================================

def download_file(url):
    """Download a GHArchive file with retry logic."""
    filename = url.split("/")[-1]
    filepath = os.path.join(GHARCHIVE_DATA_LOCAL_PATH, filename)

    if os.path.exists(filepath):
        return True

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(response.content)
            return True

        except requests.exceptions.HTTPError as e:
            # 404 means the file doesn't exist in GHArchive - skip without retry
            if e.response.status_code == 404:
                if attempt == 0:  # Only log once, not for each retry
                    print(f"   ○ {filename}: Not available (404) - skipping")
                return False

            # Other HTTP errors (5xx, etc.) should be retried
            wait_time = DOWNLOAD_RETRY_DELAY * (2 ** attempt)
            print(f"   ○ {filename}: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)

        except Exception as e:
            # Network errors, timeouts, etc. should be retried
            wait_time = DOWNLOAD_RETRY_DELAY * (2 ** attempt)
            print(f"   ○ {filename}: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)

    return False


def download_all_gharchive_data():
    """Download all GHArchive data files for the last LEADERBOARD_TIME_FRAME_DAYS."""
    os.makedirs(GHARCHIVE_DATA_LOCAL_PATH, exist_ok=True)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    urls = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        for hour in range(24):
            url = f"https://data.gharchive.org/{date_str}-{hour}.json.gz"
            urls.append(url)
        current_date += timedelta(days=1)

    downloads_processed = 0

    try:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            futures = [executor.submit(download_file, url) for url in urls]
            for future in as_completed(futures):
                downloads_processed += 1

        print(f"   Download complete: {downloads_processed} files")
        return True

    except Exception as e:
        print(f"Error during download: {str(e)}")
        traceback.print_exc()
        return False


# =============================================================================
# HUGGINGFACE API WRAPPERS
# =============================================================================

def is_retryable_error(e):
    """Check if exception is retryable (rate limit or timeout error)."""
    if isinstance(e, HfHubHTTPError):
        if e.response.status_code == 429:
            return True

    if isinstance(e, (requests.exceptions.Timeout,
                     requests.exceptions.ReadTimeout,
                     requests.exceptions.ConnectTimeout)):
        return True

    if isinstance(e, Exception):
        error_str = str(e).lower()
        if 'timeout' in error_str or 'timed out' in error_str:
            return True

    return False


@backoff.on_exception(
    backoff.expo,
    (HfHubHTTPError, requests.exceptions.Timeout, requests.exceptions.RequestException, Exception),
    max_tries=MAX_RETRIES,
    base=300,
    max_value=3600,
    giveup=lambda e: not is_retryable_error(e),
    on_backoff=lambda details: print(
        f"   {details['exception']} error. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/5..."
    )
)
def list_repo_files_with_backoff(api, **kwargs):
    """Wrapper for api.list_repo_files() with exponential backoff."""
    return api.list_repo_files(**kwargs)


@backoff.on_exception(
    backoff.expo,
    (HfHubHTTPError, requests.exceptions.Timeout, requests.exceptions.RequestException, Exception),
    max_tries=MAX_RETRIES,
    base=300,
    max_value=3600,
    giveup=lambda e: not is_retryable_error(e),
    on_backoff=lambda details: print(
        f"   {details['exception']} error. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/5..."
    )
)
def hf_hub_download_with_backoff(**kwargs):
    """Wrapper for hf_hub_download() with exponential backoff."""
    return hf_hub_download(**kwargs)


@backoff.on_exception(
    backoff.expo,
    (HfHubHTTPError, requests.exceptions.Timeout, requests.exceptions.RequestException, Exception),
    max_tries=MAX_RETRIES,
    base=300,
    max_value=3600,
    giveup=lambda e: not is_retryable_error(e),
    on_backoff=lambda details: print(
        f"   {details['exception']} error. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/5..."
    )
)
def upload_file_with_backoff(api, **kwargs):
    """Wrapper for api.upload_file() with exponential backoff."""
    return api.upload_file(**kwargs)


@backoff.on_exception(
    backoff.expo,
    (HfHubHTTPError, requests.exceptions.Timeout, requests.exceptions.RequestException, Exception),
    max_tries=MAX_RETRIES,
    base=300,
    max_value=3600,
    giveup=lambda e: not is_retryable_error(e),
    on_backoff=lambda details: print(
        f"   {details['exception']} error. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/5..."
    )
)
def upload_folder_with_backoff(api, **kwargs):
    """Wrapper for api.upload_folder() with exponential backoff."""
    return api.upload_folder(**kwargs)


def get_duckdb_connection():
    """
    Initialize DuckDB connection with OPTIMIZED memory settings.
    Uses persistent database and reduced memory footprint.
    Automatically removes cache file if lock conflict is detected.
    """
    try:
        conn = duckdb.connect(DUCKDB_CACHE_FILE)
    except Exception as e:
        # Check if it's a locking error
        error_msg = str(e)
        if "lock" in error_msg.lower() or "conflicting" in error_msg.lower():
            print(f"   ⚠ Lock conflict detected, removing {DUCKDB_CACHE_FILE}...")
            if os.path.exists(DUCKDB_CACHE_FILE):
                os.remove(DUCKDB_CACHE_FILE)
                print(f"   ✓ Cache file removed, retrying connection...")
            # Retry connection after removing cache
            conn = duckdb.connect(DUCKDB_CACHE_FILE)
        else:
            # Re-raise if it's not a locking error
            raise

    # OPTIMIZED SETTINGS
    conn.execute(f"SET threads TO {DUCKDB_THREADS};")
    conn.execute("SET preserve_insertion_order = false;")
    conn.execute("SET enable_object_cache = true;")
    conn.execute("SET temp_directory = '/tmp/duckdb_temp';")
    conn.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}';")  # Per-query limit
    conn.execute(f"SET max_memory = '{DUCKDB_MEMORY_LIMIT}';")  # Hard cap

    return conn


def generate_file_path_patterns(start_date, end_date, data_dir=GHARCHIVE_DATA_LOCAL_PATH):
    """Generate file path patterns for GHArchive data in date range (only existing files)."""
    file_patterns = []
    missing_dates = set()

    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

    while current_date <= end_day:
        date_has_files = False
        for hour in range(24):
            pattern = os.path.join(data_dir, f"{current_date.strftime('%Y-%m-%d')}-{hour}.json.gz")
            if os.path.exists(pattern):
                file_patterns.append(pattern)
                date_has_files = True

        if not date_has_files:
            missing_dates.add(current_date.strftime('%Y-%m-%d'))

        current_date += timedelta(days=1)

    if missing_dates:
        print(f"   ○ Skipping {len(missing_dates)} date(s) with no data")

    return file_patterns


# =============================================================================
# STREAMING BATCH PROCESSING
# =============================================================================

def fetch_all_wiki_metadata_streaming(conn, identifiers, start_date, end_date):
    """
    OPTIMIZED: Fetch wiki metadata using streaming batch processing.

    Processes GHArchive files in BATCH_SIZE_DAYS chunks to limit memory usage.
    Instead of loading 180 days (4,344 files) at once, processes 7 days at a time.

    This prevents OOM errors by:
    1. Only keeping ~168 hourly files in memory per batch (vs 4,344)
    2. Incrementally building the results dictionary
    3. Allowing DuckDB to garbage collect after each batch

    Args:
        conn: DuckDB connection instance
        identifiers: List of GitHub usernames/bot identifiers (~1500)
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)

    Returns:
        Dictionary mapping assistant identifier to list of wiki metadata
    """
    identifier_list = ', '.join([f"'{id}'" for id in identifiers])
    metadata_by_agent = defaultdict(list)

    # Calculate total batches
    total_days = (end_date - start_date).days
    total_batches = (total_days // BATCH_SIZE_DAYS) + 1

    # Process in configurable batches
    current_date = start_date
    batch_num = 0
    total_wiki_edits = 0

    print(f"   Streaming {total_batches} batches of {BATCH_SIZE_DAYS}-day intervals...")

    while current_date <= end_date:
        batch_num += 1
        batch_end = min(current_date + timedelta(days=BATCH_SIZE_DAYS - 1), end_date)

        # Get file patterns for THIS BATCH ONLY (not all 180 days)
        file_patterns = generate_file_path_patterns(current_date, batch_end)

        if not file_patterns:
            print(f"   Batch {batch_num}/{total_batches}: {current_date.date()} to {batch_end.date()} - NO DATA")
            current_date = batch_end + timedelta(days=1)
            continue

        # Progress indicator
        print(f"   Batch {batch_num}/{total_batches}: {current_date.date()} to {batch_end.date()} ({len(file_patterns)} files)... ", end="", flush=True)

        # Build file patterns SQL for THIS BATCH
        file_patterns_sql = '[' + ', '.join([f"'{fp}'" for fp in file_patterns]) + ']'

        # Query for this batch
        # Extract wiki edit information from GollumEvent payloads
        query = f"""
        SELECT
            TRY_CAST(json_extract_string(to_json(actor), '$.login') AS VARCHAR) as assistant,
            TRY_CAST(json_array_length(json_extract(to_json(payload), '$.pages')) AS INTEGER) as page_count,
            created_at
        FROM read_json(
            {file_patterns_sql},
            union_by_name=true,
            filename=true,
            compression='gzip',
            format='newline_delimited',
            ignore_errors=true,
            maximum_object_size=2147483648
        )
        WHERE type = 'GollumEvent'
            AND json_extract(to_json(payload), '$.pages') IS NOT NULL
            AND TRY_CAST(json_extract_string(to_json(actor), '$.login') AS VARCHAR) IN ({identifier_list})
        """

        try:
            results = conn.execute(query).fetchall()

            batch_wiki_edits = 0
            for row in results:
                assistant = row[0]
                page_count = row[1] if row[1] is not None else 0
                created_at = normalize_date_format(row[2]) if row[2] else None

                if not assistant or page_count == 0:
                    continue

                # Build wiki metadata
                wiki_metadata = {
                    'page_count': page_count,
                    'created_at': created_at,
                }

                metadata_by_agent[assistant].append(wiki_metadata)
                batch_wiki_edits += page_count
                total_wiki_edits += page_count

            print(f"✓ {batch_wiki_edits} wiki edits found")

        except Exception as e:
            print(f"\n   ✗ Batch {batch_num} error: {str(e)}")
            traceback.print_exc()

        # Move to next batch
        current_date = batch_end + timedelta(days=1)

    # Final summary
    agents_with_data = sum(1 for wiki_events in metadata_by_agent.values() if wiki_events)
    print(f"\n   ✓ Complete: {total_wiki_edits} wiki edits found for {agents_with_data}/{len(identifiers)} assistants")

    return dict(metadata_by_agent)


def sync_agents_repo():
    """
    Sync local bot_data repository with remote using git pull.
    This is MANDATORY to ensure we have the latest bot data.
    Raises exception if sync fails.
    """
    if not os.path.exists(AGENTS_REPO_LOCAL_PATH):
        error_msg = f"Local repository not found at {AGENTS_REPO_LOCAL_PATH}"
        print(f"   ✗ {error_msg}")
        print(f"   Please clone it first: git clone https://huggingface.co/datasets/{AGENTS_REPO}")
        raise FileNotFoundError(error_msg)

    if not os.path.exists(os.path.join(AGENTS_REPO_LOCAL_PATH, '.git')):
        error_msg = f"{AGENTS_REPO_LOCAL_PATH} exists but is not a git repository"
        print(f"   ✗ {error_msg}")
        raise ValueError(error_msg)

    try:
        # Run git pull with extended timeout due to large repository
        result = subprocess.run(
            ['git', 'pull'],
            cwd=AGENTS_REPO_LOCAL_PATH,
            capture_output=True,
            text=True,
            timeout=GIT_SYNC_TIMEOUT
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            if "Already up to date" in output or "Already up-to-date" in output:
                print(f"   ✓ Repository is up to date")
            else:
                print(f"   ✓ Repository synced successfully")
                if output:
                    # Print first few lines of output
                    lines = output.split('\n')[:5]
                    for line in lines:
                        print(f"     {line}")
            return True
        else:
            error_msg = f"Git pull failed: {result.stderr.strip()}"
            print(f"   ✗ {error_msg}")
            raise RuntimeError(error_msg)

    except subprocess.TimeoutExpired:
        error_msg = f"Git pull timed out after {GIT_SYNC_TIMEOUT} seconds"
        print(f"   ✗ {error_msg}")
        raise TimeoutError(error_msg)
    except (FileNotFoundError, ValueError, RuntimeError, TimeoutError):
        raise  # Re-raise expected exceptions
    except Exception as e:
        error_msg = f"Error syncing repository: {str(e)}"
        print(f"   ✗ {error_msg}")
        raise RuntimeError(error_msg) from e


def load_agents_from_hf():
    """
    Load all assistant metadata JSON files from local git repository.
    ALWAYS syncs with remote first to ensure we have the latest bot data.
    """
    # MANDATORY: Sync with remote first to get latest bot data
    print(f"   Syncing bot_data repository to get latest assistants...")
    sync_agents_repo()  # Will raise exception if sync fails

    assistants = []

    # Scan local directory for JSON files
    if not os.path.exists(AGENTS_REPO_LOCAL_PATH):
        raise FileNotFoundError(f"Local repository not found at {AGENTS_REPO_LOCAL_PATH}")

    # Walk through the directory to find all JSON files
    files_processed = 0
    print(f"   Loading assistant metadata from {AGENTS_REPO_LOCAL_PATH}...")

    for root, dirs, files in os.walk(AGENTS_REPO_LOCAL_PATH):
        # Skip .git directory
        if '.git' in root:
            continue

        for filename in files:
            if not filename.endswith('.json'):
                continue

            files_processed += 1
            file_path = os.path.join(root, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    agent_data = json.load(f)

                # Only include active assistants
                if agent_data.get('status') != 'active':
                    continue

                # Extract github_identifier from filename
                github_identifier = filename.replace('.json', '')
                agent_data['github_identifier'] = github_identifier

                assistants.append(agent_data)

            except Exception as e:
                print(f"   ○ Error loading {filename}: {str(e)}")
                continue

    print(f"   ✓ Loaded {len(assistants)} active assistants (from {files_processed} total files)")
    return assistants


def calculate_wiki_stats_from_metadata(metadata_list):
    """Calculate statistics from a list of wiki metadata."""
    total_wiki_edits = sum(item.get('page_count', 0) for item in metadata_list)

    return {
        'total_wiki_edits': total_wiki_edits,
    }


def calculate_monthly_metrics_by_agent(all_metadata_dict, assistants):
    """Calculate monthly metrics for all assistants for visualization."""
    identifier_to_name = {assistant.get('github_identifier'): assistant.get('name') for assistant in assistants if assistant.get('github_identifier')}

    if not all_metadata_dict:
        return {'assistants': [], 'months': [], 'data': {}}

    agent_month_data = defaultdict(lambda: defaultdict(list))

    for agent_identifier, metadata_list in all_metadata_dict.items():
        for wiki_meta in metadata_list:
            created_at = wiki_meta.get('created_at')

            if not created_at:
                continue

            agent_name = identifier_to_name.get(agent_identifier, agent_identifier)

            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                month_key = f"{dt.year}-{dt.month:02d}"
                agent_month_data[agent_name][month_key].append(wiki_meta)
            except Exception as e:
                print(f"Warning: Could not parse date '{created_at}': {e}")
                continue

    all_months = set()
    for agent_data in agent_month_data.values():
        all_months.update(agent_data.keys())
    months = sorted(list(all_months))

    result_data = {}
    for agent_name, month_dict in agent_month_data.items():
        total_wiki_edits_list = []

        for month in months:
            wiki_events_in_month = month_dict.get(month, [])
            total_count = sum(item.get('page_count', 0) for item in wiki_events_in_month)

            total_wiki_edits_list.append(total_count)

        result_data[agent_name] = {
            'total_wiki_edits': total_wiki_edits_list,
        }

    agents_list = sorted(list(agent_month_data.keys()))

    return {
        'assistants': agents_list,
        'months': months,
        'data': result_data
    }


def construct_leaderboard_from_metadata(all_metadata_dict, assistants):
    """Construct leaderboard from in-memory wiki metadata."""
    if not assistants:
        print("Error: No assistants found")
        return {}

    cache_dict = {}

    for assistant in assistants:
        identifier = assistant.get('github_identifier')
        agent_name = assistant.get('name', 'Unknown')

        bot_metadata = all_metadata_dict.get(identifier, [])
        stats = calculate_wiki_stats_from_metadata(bot_metadata)

        cache_dict[identifier] = {
            'name': agent_name,
            'website': assistant.get('website', 'N/A'),
            'github_identifier': identifier,
            **stats
        }

    return cache_dict


def save_leaderboard_data_to_hf(leaderboard_dict, monthly_metrics):
    """Save leaderboard data and monthly metrics to HuggingFace dataset."""
    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi(token=token)

        combined_data = {
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'leaderboard': leaderboard_dict,
            'monthly_metrics': monthly_metrics,
            'metadata': {
                'leaderboard_time_frame_days': LEADERBOARD_TIME_FRAME_DAYS
            }
        }

        with open(LEADERBOARD_FILENAME, 'w') as f:
            json.dump(combined_data, f, indent=2)

        try:
            upload_file_with_backoff(
                api=api,
                path_or_fileobj=LEADERBOARD_FILENAME,
                path_in_repo=LEADERBOARD_FILENAME,
                repo_id=LEADERBOARD_REPO,
                repo_type="dataset"
            )
            return True
        finally:
            if os.path.exists(LEADERBOARD_FILENAME):
                os.remove(LEADERBOARD_FILENAME)

    except Exception as e:
        print(f"Error saving leaderboard data: {str(e)}")
        traceback.print_exc()
        return False


# =============================================================================
# MINING FUNCTION
# =============================================================================

def mine_all_agents():
    """
    Mine wiki metadata for all assistants using STREAMING batch processing.
    Downloads GHArchive data, then uses BATCH-based DuckDB queries.
    """
    print(f"\n[1/4] Downloading GHArchive data...")

    if not download_all_gharchive_data():
        print("Warning: Download had errors, continuing with available data...")

    print(f"\n[2/4] Loading assistant metadata...")

    assistants = load_agents_from_hf()
    if not assistants:
        print("Error: No assistants found")
        return

    identifiers = [assistant['github_identifier'] for assistant in assistants if assistant.get('github_identifier')]
    if not identifiers:
        print("Error: No valid assistant identifiers found")
        return

    print(f"\n[3/4] Mining wiki metadata ({len(identifiers)} assistants, {LEADERBOARD_TIME_FRAME_DAYS} days)...")

    try:
        conn = get_duckdb_connection()
    except Exception as e:
        print(f"Failed to initialize DuckDB connection: {str(e)}")
        return

    current_time = datetime.now(timezone.utc)
    end_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    try:
        # USE STREAMING FUNCTION
        all_metadata = fetch_all_wiki_metadata_streaming(
            conn, identifiers, start_date, end_date
        )
    except Exception as e:
        print(f"Error during DuckDB fetch: {str(e)}")
        traceback.print_exc()
        return
    finally:
        conn.close()

    print(f"\n[4/4] Saving leaderboard...")

    try:
        leaderboard_dict = construct_leaderboard_from_metadata(all_metadata, assistants)
        monthly_metrics = calculate_monthly_metrics_by_agent(all_metadata, assistants)
        save_leaderboard_data_to_hf(leaderboard_dict, monthly_metrics)
    except Exception as e:
        print(f"Error saving leaderboard: {str(e)}")
        traceback.print_exc()
    finally:
        # Clean up DuckDB cache file to save storage
        if os.path.exists(DUCKDB_CACHE_FILE):
            try:
                os.remove(DUCKDB_CACHE_FILE)
                print(f"   ✓ Cache file removed: {DUCKDB_CACHE_FILE}")
            except Exception as e:
                print(f"   ⚠ Failed to remove cache file: {str(e)}")


# =============================================================================
# SCHEDULER SETUP
# =============================================================================

def setup_scheduler():
    """Set up APScheduler to run mining jobs periodically."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logging.getLogger('httpx').setLevel(logging.WARNING)

    scheduler = BlockingScheduler(timezone=SCHEDULE_TIMEZONE)

    trigger = CronTrigger(
        day_of_week=SCHEDULE_DAY_OF_WEEK,
        hour=SCHEDULE_HOUR,
        minute=SCHEDULE_MINUTE,
        timezone=SCHEDULE_TIMEZONE
    )

    scheduler.add_job(
        mine_all_agents,
        trigger=trigger,
        id='mine_all_agents',
        name='Mine GHArchive data for all assistants',
        replace_existing=True
    )

    next_run = trigger.get_next_fire_time(None, datetime.now(trigger.timezone))
    print(f"Scheduler: Weekly on {SCHEDULE_DAY_OF_WEEK} at {SCHEDULE_HOUR:02d}:{SCHEDULE_MINUTE:02d} {SCHEDULE_TIMEZONE}")
    print(f"Next run: {next_run}\n")

    print(f"\nScheduler started")
    scheduler.start()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if SCHEDULE_ENABLED:
        setup_scheduler()
    else:
        mine_all_agents()
