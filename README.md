# Git Review Tool

This project is an automated code review tool that integrates with GitLab and an OpenAI-based API (via OpenWebUI). It retrieves merge request changes, analyzes code diffs using a large language model (LLM), and posts inline comments or an overall summary directly on the merge request.

## Features

- **GitLab Integration:** Fetches file content and diffs from GitLab merge requests.
- **Context Extraction:** Extracts methods and dependencies from Python files to build context.
- **Diff Analysis:** Splits code diffs into chunks and gathers surrounding code context.
- **LLM-Powered Review:** Uses an OpenAI-based API (powered by [OpenWebUI](https://openwebui.com)) to evaluate code quality and identify potential issues.
- **Automated Comments:** Posts inline comments and an overall review summary to the merge request.

## Setup

1. **Environment Variables**

   Create a `.env` file in the project root. A template is provided in [`.env.example`](.env.example).

2. **Install Dependencies**

   Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt

3. **RUN**

   Run `main.py`