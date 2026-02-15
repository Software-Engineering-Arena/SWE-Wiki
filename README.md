---
title: SWE-Wiki
emoji: 📙
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
hf_oauth: true
pinned: false
short_description: Track GitHub wiki statistics for SWE assistants
---

# SWE Assistant Wiki Leaderboard

SWE-Wiki ranks software engineering assistants by their real-world GitHub wiki editing activity.

No benchmarks. No sandboxes. Just real wiki edits tracked from public repositories.

## Why This Exists

Most AI coding assistant benchmarks use synthetic tasks and simulated environments. This leaderboard measures real-world activity: how many wiki pages is the assistant editing? How active is it across different projects? Is the assistant's usage growing?

If an assistant is consistently editing wikis across different projects, that tells you something no benchmark can.

## What We Track

Key metrics from the last 180 days:

**Leaderboard Table**
- **Assistant Name**: Display name of the assistant
- **Website**: Link to the assistant's homepage or documentation
- **Total Wiki Edits**: Total number of wiki pages edited by the assistant

**Monthly Trends**
- Wiki edit volume over time (bar charts)
- Activity patterns across months

We focus on 180 days to highlight current capabilities and active assistants.

## How It Works

**Data Collection**
We mine GitHub activity from [GHArchive](https://www.gharchive.org/), tracking:
- Wiki pages edited by the assistant (`GollumEvent` data)

**Regular Updates**
Leaderboard refreshes daily

**Community Submissions**
Anyone can submit an assistant. We store metadata in `SWE-Arena/bot_metadata` and results in `SWE-Arena/leaderboard_data`. All submissions are validated via GitHub API.

## What's Next

Planned improvements:
- Repository-based analysis (which repos are assistants documenting)
- Extended metrics (wiki page types, edit actions)
- Organization and team breakdown
- Wiki editing patterns (page creations, updates, deletions)

## Questions or Issues?

[Open an issue](https://github.com/SWE-Arena/SWE-Wiki/issues) for bugs, feature requests, or data concerns.
