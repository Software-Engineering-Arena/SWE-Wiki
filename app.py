import gradio as gr
from gradio_leaderboard import Leaderboard
import json
import os
import time
import requests
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
import backoff
from dotenv import load_dotenv
import pandas as pd
import random
import plotly.graph_objects as go
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/bot_metadata"  # HuggingFace dataset for agent metadata
LEADERBOARD_FILENAME = f"{os.getenv('COMPOSE_PROJECT_NAME')}.json"
LEADERBOARD_REPO = "SWE-Arena/leaderboard_data"  # HuggingFace dataset for leaderboard data
MAX_RETRIES = 5

LEADERBOARD_COLUMNS = [
    ("Agent Name", "string"),
    ("Website", "string"),
    ("Total Wiki Edits", "number"),
]

# =============================================================================
# HUGGINGFACE API WRAPPERS WITH BACKOFF
# =============================================================================

def is_rate_limit_error(e):
    """Check if exception is a HuggingFace rate limit error (429)."""
    if isinstance(e, HfHubHTTPError):
        return e.response.status_code == 429
    return False


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    max_tries=MAX_RETRIES,
    base=300,
    max_value=3600,
    giveup=lambda e: not is_rate_limit_error(e),
    on_backoff=lambda details: print(
        f"Rate limited. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/5..."
    )
)
def list_repo_files_with_backoff(api, **kwargs):
    """Wrapper for api.list_repo_files() with exponential backoff for rate limits."""
    return api.list_repo_files(**kwargs)


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    max_tries=MAX_RETRIES,
    base=300,
    max_value=3600,
    giveup=lambda e: not is_rate_limit_error(e),
    on_backoff=lambda details: print(
        f"Rate limited. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/5..."
    )
)
def hf_hub_download_with_backoff(**kwargs):
    """Wrapper for hf_hub_download() with exponential backoff for rate limits."""
    return hf_hub_download(**kwargs)


# =============================================================================
# GITHUB USERNAME VALIDATION
# =============================================================================

def validate_github_username(identifier):
    """Verify that a GitHub identifier exists."""
    try:
        response = requests.get(f'https://api.github.com/users/{identifier}', timeout=10)
        return (True, "Username is valid") if response.status_code == 200 else (False, "GitHub identifier not found" if response.status_code == 404 else f"Validation error: HTTP {response.status_code}")
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# =============================================================================
# HUGGINGFACE DATASET OPERATIONS
# =============================================================================

def load_agents_from_hf():
    """Load all agent metadata JSON files from HuggingFace dataset."""
    try:
        api = HfApi()
        agents = []

        # List all files in the repository
        files = list_repo_files_with_backoff(api=api, repo_id=AGENTS_REPO, repo_type="dataset")

        # Filter for JSON files only
        json_files = [f for f in files if f.endswith('.json')]

        # Download and parse each JSON file
        for json_file in json_files:
            try:
                file_path = hf_hub_download_with_backoff(
                    repo_id=AGENTS_REPO,
                    filename=json_file,
                    repo_type="dataset"
                )

                with open(file_path, 'r') as f:
                    agent_data = json.load(f)

                    # Only process agents with status == "active"
                    if agent_data.get('status') != 'active':
                        continue

                    # Extract github_identifier from filename (e.g., "agent[bot].json" -> "agent[bot]")
                    filename_identifier = json_file.replace('.json', '')

                    # Add or override github_identifier to match filename
                    agent_data['github_identifier'] = filename_identifier

                    agents.append(agent_data)

            except Exception as e:
                print(f"Warning: Could not load {json_file}: {str(e)}")
                continue

        print(f"Loaded {len(agents)} agents from HuggingFace")
        return agents

    except Exception as e:
        print(f"Could not load agents from HuggingFace: {str(e)}")
        return None


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


def upload_with_retry(api, path_or_fileobj, path_in_repo, repo_id, repo_type, token, max_retries=5):
    """
    Upload file to HuggingFace with exponential backoff retry logic.

    Args:
        api: HfApi instance
        path_or_fileobj: Local file path to upload
        path_in_repo: Target path in the repository
        repo_id: Repository ID
        repo_type: Type of repository (e.g., "dataset")
        token: HuggingFace token
        max_retries: Maximum number of retry attempts

    Returns:
        True if upload succeeded, raises exception if all retries failed
    """
    delay = 2.0  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token
            )
            if attempt > 0:
                print(f"   Upload succeeded on attempt {attempt + 1}/{max_retries}")
            return True

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay + random.uniform(0, 1.0)
                print(f"   Upload failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                print(f"   Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                delay = min(delay * 2, 60.0)  # Exponential backoff, max 60s
            else:
                print(f"   Upload failed after {max_retries} attempts: {str(e)}")
                raise


def save_agent_to_hf(data):
    """Save a new agent to HuggingFace dataset as {identifier}.json in root."""
    try:
        api = HfApi()
        token = get_hf_token()

        if not token:
            raise Exception("No HuggingFace token found. Please set HF_TOKEN in your Space settings.")

        identifier = data['github_identifier']
        filename = f"{identifier}.json"

        # Save locally first
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        try:
            # Upload to HuggingFace (root directory)
            upload_with_retry(
                api=api,
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=AGENTS_REPO,
                repo_type="dataset",
                token=token
            )
            print(f"Saved agent to HuggingFace: {filename}")
            return True
        finally:
            # Always clean up local file, even if upload fails
            if os.path.exists(filename):
                os.remove(filename)

    except Exception as e:
        print(f"Error saving agent: {str(e)}")
        return False


def load_leaderboard_data_from_hf():
    """
    Load leaderboard data and monthly metrics from HuggingFace dataset.

    Returns:
        dict: Dictionary with 'leaderboard', 'monthly_metrics', and 'metadata' keys
              Returns None if file doesn't exist or error occurs
    """
    try:
        token = get_hf_token()

        # Download file
        file_path = hf_hub_download_with_backoff(
            repo_id=LEADERBOARD_REPO,
            filename=LEADERBOARD_FILENAME,
            repo_type="dataset",
            token=token
        )

        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        last_updated = data.get('metadata', {}).get('last_updated', 'Unknown')
        print(f"Loaded leaderboard data from HuggingFace (last updated: {last_updated})")

        return data

    except Exception as e:
        print(f"Could not load leaderboard data from HuggingFace: {str(e)}")
        return None


# =============================================================================
# UI FUNCTIONS
# =============================================================================

def create_monthly_metrics_plot(top_n=5):
    """
    Create a Plotly figure showing monthly wiki edits as bar charts.

    Args:
        top_n: Number of top agents to show (default: 5)
    """
    # Load from saved dataset
    saved_data = load_leaderboard_data_from_hf()

    if not saved_data or 'monthly_metrics' not in saved_data:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=None,
            xaxis_title=None,
            height=500
        )
        return fig

    metrics = saved_data['monthly_metrics']
    print(f"Loaded monthly metrics from saved dataset")

    # Apply top_n filter if specified
    if top_n is not None and top_n > 0 and metrics.get('agents'):
        # Calculate wiki edits for each agent
        agent_totals = []
        for agent_name in metrics['agents']:
            agent_data = metrics['data'].get(agent_name, {})
            wiki_edits = sum(agent_data.get('total_wiki_edits', []))
            agent_totals.append((agent_name, wiki_edits))

        # Sort by wiki edits and take top N
        agent_totals.sort(key=lambda x: x[1], reverse=True)
        top_agents = [agent_name for agent_name, _ in agent_totals[:top_n]]

        # Filter metrics to only include top agents
        metrics = {
            'agents': top_agents,
            'months': metrics['months'],
            'data': {agent: metrics['data'][agent] for agent in top_agents if agent in metrics['data']}
        }

    if not metrics['agents'] or not metrics['months']:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=None,
            xaxis_title=None,
            height=500
        )
        return fig

    # Create figure
    fig = go.Figure()

    # Generate unique colors for many agents using HSL color space
    def generate_color(index, total):
        """Generate distinct colors using HSL color space for better distribution"""
        hue = (index * 360 / total) % 360
        saturation = 70 + (index % 3) * 10  # Vary saturation slightly
        lightness = 45 + (index % 2) * 10   # Vary lightness slightly
        return f'hsl({hue}, {saturation}%, {lightness}%)'

    agents = metrics['agents']
    months = metrics['months']
    data = metrics['data']

    # Generate colors for all agents
    agent_colors = {agent: generate_color(idx, len(agents)) for idx, agent in enumerate(agents)}

    # Add bar traces for each agent
    for idx, agent_name in enumerate(agents):
        color = agent_colors[agent_name]
        agent_data = data[agent_name]

        # Add bar trace for total wiki edits
        # Only show bars for months where agent has wiki edits
        x_bars = []
        y_bars = []
        for month, count in zip(months, agent_data['total_wiki_edits']):
            if count > 0:  # Only include months with wiki edits
                x_bars.append(month)
                y_bars.append(count)

        if x_bars and y_bars:  # Only add trace if there's data
            fig.add_trace(
                go.Bar(
                    x=x_bars,
                    y=y_bars,
                    name=agent_name,
                    marker=dict(color=color, opacity=0.7),
                    hovertemplate='<b>Agent: %{fullData.name}</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Wiki Edits: %{y}<br>' +
                                 '<extra></extra>',
                    offsetgroup=agent_name  # Group bars by agent for proper spacing
                )
            )

    # Update axes labels
    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text="<b>Wiki Edits</b>")

    # Update layout
    show_legend = (top_n is not None and top_n <= 10)
    fig.update_layout(
        title=None,
        hovermode='closest',  # Show individual agent info on hover
        barmode='group',
        height=600,
        showlegend=show_legend,
        margin=dict(l=50, r=150 if show_legend else 50, t=50, b=50)  # More right margin when legend is shown
    )

    return fig


def get_leaderboard_dataframe():
    """
    Load leaderboard from saved dataset and convert to pandas DataFrame for display.
    Returns formatted DataFrame sorted by wiki edits.
    """
    # Load from saved dataset
    saved_data = load_leaderboard_data_from_hf()

    if not saved_data or 'leaderboard' not in saved_data:
        print(f"No leaderboard data available")
        # Return empty DataFrame with correct columns if no data
        column_names = [col[0] for col in LEADERBOARD_COLUMNS]
        return pd.DataFrame(columns=column_names)

    cache_dict = saved_data['leaderboard']
    last_updated = saved_data.get('metadata', {}).get('last_updated', 'Unknown')
    print(f"Loaded leaderboard from saved dataset (last updated: {last_updated})")
    print(f"Cache dict size: {len(cache_dict)}")

    if not cache_dict:
        print("WARNING: cache_dict is empty!")
        # Return empty DataFrame with correct columns if no data
        column_names = [col[0] for col in LEADERBOARD_COLUMNS]
        return pd.DataFrame(columns=column_names)

    rows = []
    filtered_count = 0
    for identifier, data in cache_dict.items():
        wiki_edits = data.get('total_wiki_edits', 0)
        print(f"   Agent '{identifier}': {wiki_edits} wiki edits")

        # Filter out agents with zero wiki edits
        if wiki_edits == 0:
            filtered_count += 1
            continue

        # Only include display-relevant fields
        rows.append([
            data.get('name', 'Unknown'),
            data.get('website', 'N/A'),
            wiki_edits,
        ])

    print(f"Filtered out {filtered_count} agents with 0 wiki edits")
    print(f"Leaderboard will show {len(rows)} agents")

    # Create DataFrame
    column_names = [col[0] for col in LEADERBOARD_COLUMNS]
    df = pd.DataFrame(rows, columns=column_names)

    # Ensure numeric types
    numeric_cols = ["Total Wiki Edits"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Sort by Total Wiki Edits descending
    if "Total Wiki Edits" in df.columns and not df.empty:
        df = df.sort_values(by="Total Wiki Edits", ascending=False).reset_index(drop=True)

    print(f"Final DataFrame shape: {df.shape}")
    print("="*60 + "\n")

    return df


def submit_agent(identifier, agent_name, organization, website):
    """
    Submit a new agent to the leaderboard.
    Validates input and saves submission.
    """
    # Validate required fields
    if not identifier or not identifier.strip():
        return "ERROR: GitHub identifier is required", gr.update()
    if not agent_name or not agent_name.strip():
        return "ERROR: Agent name is required", gr.update()
    if not organization or not organization.strip():
        return "ERROR: Organization name is required", gr.update()
    if not website or not website.strip():
        return "ERROR: Website URL is required", gr.update()

    # Clean inputs
    identifier = identifier.strip()
    agent_name = agent_name.strip()
    organization = organization.strip()
    website = website.strip()

    # Validate GitHub identifier
    is_valid, message = validate_github_username(identifier)
    if not is_valid:
        return f"ERROR: {message}", gr.update()

    # Check for duplicates by loading agents from HuggingFace
    agents = load_agents_from_hf()
    if agents:
        existing_names = {agent['github_identifier'] for agent in agents}
        if identifier in existing_names:
            return f"WARNING: Agent with identifier '{identifier}' already exists", gr.update()

    # Create submission
    submission = {
        'name': agent_name,
        'organization': organization,
        'github_identifier': identifier,
        'website': website,
        'status': 'active'
    }

    # Save to HuggingFace
    if not save_agent_to_hf(submission):
        return "ERROR: Failed to save submission", gr.update()

    # Return success message - data will be populated by backend updates
    return f"SUCCESS: Successfully submitted {agent_name}! Wiki edits data will be automatically populated by the backend system via the maintainers.", gr.update()


# =============================================================================
# DATA RELOAD FUNCTION
# =============================================================================

def reload_leaderboard_data():
    """
    Reload leaderboard data from HuggingFace.
    This function is called by the scheduler on a daily basis.
    """
    print(f"\n{'='*80}")
    print(f"Reloading leaderboard data from HuggingFace...")
    print(f"{'='*80}\n")

    try:
        data = load_leaderboard_data_from_hf()
        if data:
            print(f"Successfully reloaded leaderboard data")
            print(f"   Last updated: {data.get('metadata', {}).get('last_updated', 'Unknown')}")
            print(f"   Agents: {len(data.get('leaderboard', {}))}")
        else:
            print(f"No data available")
    except Exception as e:
        print(f"Error reloading leaderboard data: {str(e)}")

    print(f"{'='*80}\n")


# =============================================================================
# GRADIO APPLICATION
# =============================================================================

print(f"\nStarting SWE Agent Wiki Leaderboard")
print(f"   Data source: {LEADERBOARD_REPO}")
print(f"   Reload frequency: Daily at 12:00 AM UTC\n")

# Start APScheduler for daily data reload at 12:00 AM UTC
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    reload_leaderboard_data,
    trigger=CronTrigger(hour=0, minute=0),  # 12:00 AM UTC daily
    id='daily_data_reload',
    name='Daily Data Reload',
    replace_existing=True
)
scheduler.start()
print(f"\n{'='*80}")
print(f"Scheduler initialized successfully")
print(f"Reload schedule: Daily at 12:00 AM UTC")
print(f"On startup: Loads cached data from HuggingFace on demand")
print(f"{'='*80}\n")

# Create Gradio interface
with gr.Blocks(title="SWE Agent Wiki Leaderboard", theme=gr.themes.Soft()) as app:
    gr.Markdown("# SWE Agent Wiki Leaderboard")
    gr.Markdown(f"Track and compare wiki edits made by SWE agents")

    with gr.Tabs():

        # Leaderboard Tab
        with gr.Tab("Leaderboard"):
            gr.Markdown("*Statistics are based on wiki edits made by agents*")
            leaderboard_table = Leaderboard(
                value=pd.DataFrame(columns=[col[0] for col in LEADERBOARD_COLUMNS]),  # Empty initially
                datatype=LEADERBOARD_COLUMNS,
                search_columns=["Agent Name", "Website"],
                filter_columns=[]
            )

            # Load leaderboard data when app starts
            app.load(
                fn=get_leaderboard_dataframe,
                inputs=[],
                outputs=[leaderboard_table]
            )

            # Monthly Metrics Section
            gr.Markdown("---")  # Divider
            gr.Markdown("### Monthly Performance - Top 5 Agents")
            gr.Markdown("*Shows wiki edits for the most active agents*")

            monthly_metrics_plot = gr.Plot(label="Monthly Metrics")

            # Load monthly metrics when app starts
            app.load(
                fn=lambda: create_monthly_metrics_plot(),
                inputs=[],
                outputs=[monthly_metrics_plot]
            )


        # Submit Agent Tab
        with gr.Tab("Submit Agent"):

            gr.Markdown("### Submit Your Agent")
            gr.Markdown("Fill in the details below to add your agent to the leaderboard.")

            with gr.Row():
                with gr.Column():
                    github_input = gr.Textbox(
                        label="GitHub Identifier*",
                        placeholder="Your agent username (e.g., my-agent[bot])"
                    )
                    name_input = gr.Textbox(
                        label="Agent Name*",
                        placeholder="Your agent's display name"
                    )

                with gr.Column():
                    organization_input = gr.Textbox(
                        label="Organization*",
                        placeholder="Your organization or team name"
                    )
                    website_input = gr.Textbox(
                        label="Website*",
                        placeholder="https://your-agent-website.com"
                    )

            submit_button = gr.Button(
                "Submit Agent",
                variant="primary"
            )
            submission_status = gr.Textbox(
                label="Submission Status",
                interactive=False
            )

            # Event handler
            submit_button.click(
                fn=submit_agent,
                inputs=[github_input, name_input, organization_input, website_input],
                outputs=[submission_status, leaderboard_table]
            )


# Launch application
if __name__ == "__main__":
    app.launch()
