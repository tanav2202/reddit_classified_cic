# Reddit Classified CIC

## Overview

Reddit Classified CIC is a small end-to-end pipeline for collecting recent posts from a subreddit and categorising them with a large language model (LLM). The project consists of two core steps:

- **Data collection** – `reddit_fetcher.py` pulls posts (and their comments) from Reddit during a configurable date window and stores them as Parquet files.
- **LLM-based categorisation** – `main.py` loads the collected data and applies a structured prompt (stored in `prompts/classify_post.jinja`) through an LLM client defined in `llm_classifier.py`, adding a `category` column to the dataset.

Both steps can be run together through `main.py`, which will fetch new data when it is missing and then classify each post.

## Project Structure

```
reddit_classified_cic/
├── main.py                       # Orchestrates fetching + classification workflow
├── reddit_fetcher.py             # Reddit ingestion utilities (PRAW based)
├── llm_classifier.py             # Helpers for Ollama / Bedrock / OpenAI classification
├── prompts/
│   └── classify_post.jinja       # Jinja template defining the classification prompt
├── reddit_data/                  # Raw Parquet exports (created at runtime)
├── reddit_data_classified/       # Classified Parquet outputs (created at runtime)
└── requirements.txt              # Minimal dependency list for the fetcher
```

## Prerequisites

- Python 3.10 or newer
- Reddit API credentials (client id, client secret, user agent)
- Local Ollama installation (for the default `mistral:latest` model), or AWS Bedrock credentials if you choose that backend
- Optional: An OpenAI API key if you intend to use GPT models via the helper in `llm_classifier.py`

### Python Dependencies

Install the core dependencies:

```bash
pip install -r requirements.txt
```

The classification helpers additionally rely on several LangChain integrations and utilities. Install them before running `main.py`:

```bash
pip install "langchain>=0.2" "langchain-core>=0.2" langchain-community langchain-aws langchain-ollama langchain-openai jinja2 tqdm
```

> **Tip:** Bundle the extra packages into a virtual environment so they stay isolated from your system Python.

## Configure Reddit API Access

Set the credentials as environment variables so `reddit_fetcher.py` can authenticate:

```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
export REDDIT_USER_AGENT="YourAppName/1.0 by YourRedditUsername"
```

On Windows (PowerShell):

```powershell
$env:REDDIT_CLIENT_ID="your_client_id"
$env:REDDIT_CLIENT_SECRET="your_client_secret"
$env:REDDIT_USER_AGENT="YourAppName/1.0 by YourRedditUsername"
```

Persist them in your shell profile (e.g., `~/.zshrc`, `~/.bashrc`) if you plan to use the script frequently. Never commit these secrets to source control.

## Configure the LLM Backend

### Ollama (default)

1. Install and start [Ollama](https://ollama.com/).
2. Pull a suitable model – the project defaults to `mistral:latest` but any text-capable model will work:

   ```bash
   ollama pull mistral:latest
   ```

3. Ensure the Ollama service is running before invoking `main.py`.

### AWS Bedrock (optional)

If you prefer Bedrock:

- Configure AWS credentials with permission to invoke the desired model (e.g., Claude 3 Sonnet).
- Uncomment `llm = get_bedrock_model()` in `main.py` and adjust the `model_id`/region inside `llm_classifier.py` if needed.

### OpenAI (optional)

`llm_classifier.py` also exposes `get_openai_model`. Supply an API key through the standard `OPENAI_API_KEY` environment variable and switch the call in `main.py` accordingly.

## Running the Pipeline

1. Activate your virtual environment (optional but recommended).
2. Ensure Reddit credentials and the chosen LLM backend are configured and running.
3. Execute the main workflow:

   ```bash
   python main.py
   ```

What happens:

- The script computes a seven-day window ending “now”.
- It checks `reddit_data/<subreddit>_<today>_<start>.parquet`; if missing, it fetches posts from the `subreddit` (default `ubc`) within that window and saves the raw Parquet file.
- It loads the Parquet data, combines `Title` and `Post_Text`, and calls the LLM using the structured prompt.
- A `category` column is added and the classified dataset is saved to `reddit_data_classified/` with the same filename convention.

Progress updates are displayed via `tqdm` while classifying.

## Customisation

- **Subreddit** – change the `subreddit` variable in `main.py`.
- **Date range** – adjust `today_date` and `old_date` or the `days_back` calculation.
- **Prompt / Categories** – edit `prompts/classify_post.jinja` to redefine the allowed categories or instructions to the model.
- **LLM model** – change the call to `get_ollama_model("mistral:latest")` (or use `get_bedrock_model` / `get_openai_model`).
- **Output locations** – modify the `reddit_data` and `reddit_data_classified` folder paths.

## Output Files

- **Raw data:** `reddit_data/<subreddit>_<today>_<start>.parquet`
  - Columns include `Title`, `Post_Text`, `Post_URL`, `Comments`, `Created_UTC`.
- **Classified data:** `reddit_data_classified/<subreddit>_<today>_<start>.parquet`
  - Adds a `category` column produced by the LLM.

## Troubleshooting

- **Zero rows fetched** – ensure your date window is sensible and that the subreddit has recent activity. The script only inspects the newest ~1000 posts.
- **“Missing required environment variables”** – confirm the Reddit credentials are exported in your shell session.
- **Ollama model errors** – verify the model name matches one you have pulled locally. Use `ollama list` to inspect installed models.
- **LangChain import errors** – double-check that the optional dependencies listed above are installed.
- **LLM produces unexpected categories** – refine `prompts/classify_post.jinja`, add explicit examples, or adjust the model/temperature settings in `llm_classifier.py`.

