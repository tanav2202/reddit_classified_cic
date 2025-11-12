# Reddit API Fetcher

A Python script to scrape posts and comments from Reddit subreddits using the Reddit API (PRAW).

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Reddit API Credentials

1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app" or "create app"
3. Choose "script" as the app type
4. Fill in the details:
   - **Name**: Your app name (e.g., "MyRedditScraper")
   - **Description**: Brief description
   - **Redirect URI**: `http://localhost:8080` (or any valid URL)
5. Click "create app"
6. Note down:
   - **Client ID**: The string under your app name (looks like: `abc123def456`)
   - **Client Secret**: The "secret" field (looks like: `xyz789_secret_key`)
   - **User Agent**: Format: `"YourAppName/1.0 by YourRedditUsername"`

### 3. Set Environment Variables

You have several options to set environment variables:

#### Option A: Set in Terminal (Temporary - Current Session Only)

**For macOS/Linux (zsh/bash):**
```bash
export REDDIT_CLIENT_ID="your_client_id_here"
export REDDIT_CLIENT_SECRET="your_client_secret_here"
export REDDIT_USER_AGENT="YourAppName/1.0 by YourRedditUsername"
```

**For Windows (PowerShell):**
```powershell
$env:REDDIT_CLIENT_ID="your_client_id_here"
$env:REDDIT_CLIENT_SECRET="your_client_secret_here"
$env:REDDIT_USER_AGENT="YourAppName/1.0 by YourRedditUsername"
```

**For Windows (Command Prompt):**
```cmd
set REDDIT_CLIENT_ID=your_client_id_here
set REDDIT_CLIENT_SECRET=your_client_secret_here
set REDDIT_USER_AGENT=YourAppName/1.0 by YourRedditUsername
```

#### Option B: Set Permanently in Shell Config (Recommended)

**For macOS/Linux (zsh - default on macOS):**
Add to `~/.zshrc`:
```bash
export REDDIT_CLIENT_ID="your_client_id_here"
export REDDIT_CLIENT_SECRET="your_client_secret_here"
export REDDIT_USER_AGENT="YourAppName/1.0 by YourRedditUsername"
```

Then reload:
```bash
source ~/.zshrc
```

**For macOS/Linux (bash):**
Add to `~/.bashrc` or `~/.bash_profile`:
```bash
export REDDIT_CLIENT_ID="your_client_id_here"
export REDDIT_CLIENT_SECRET="your_client_secret_here"
export REDDIT_USER_AGENT="YourAppName/1.0 by YourRedditUsername"
```

Then reload:
```bash
source ~/.bashrc
```

**For Windows:**
1. Open System Properties → Environment Variables
2. Add new User variables:
   - `REDDIT_CLIENT_ID` = `your_client_id_here`
   - `REDDIT_CLIENT_SECRET` = `your_client_secret_here`
   - `REDDIT_USER_AGENT` = `YourAppName/1.0 by YourRedditUsername`
3. Restart your terminal/IDE

#### Option C: Use a .env File (Alternative - Requires python-dotenv)

1. Create a `.env` file in the project directory:
```bash
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=YourAppName/1.0 by YourRedditUsername
```

2. Install python-dotenv:
```bash
pip install python-dotenv
```

3. Add to the top of your script:
```python
from dotenv import load_dotenv
load_dotenv()
```

**Important**: Add `.env` to `.gitignore` to avoid committing secrets!

## Usage

### Run the Script

```bash
python reddit_fetcher.py
```

The script will:
- Scrape new posts from the last 7 days from each subreddit
- Fetch all comments for each post
- Save to CSV files with date range in filename
- Output format: `{subreddit_name}_{today_date}_{end_date}.csv`

### Customize Subreddits

Edit the `subreddit_names` list in `reddit_fetcher.py`:

```python
subreddit_names = ['ubc', 'python', 'programming']
```

### Customize Date Range

Edit the date logic in the script:

```python
manual_date = None  # Set to dt.datetime(2025, 1, 15) to scrape from a specific date
current_date = manual_date or dt.datetime.utcnow()
end_date = current_date - dt.timedelta(days=7)  # Change 7 to adjust range
```

## Output Format

Each CSV file contains:
- **Title**: Post title
- **Post_Text**: Full text of the post
- **Post_URL**: URL of the post
- **Comments**: List of all comments (as a list in CSV)
- **Created_UTC**: Post creation timestamp

## Troubleshooting

### "Missing required environment variables" Error

Make sure you've set all three environment variables:
- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`

Verify they're set:
```bash
echo $REDDIT_CLIENT_ID
echo $REDDIT_CLIENT_SECRET
echo $REDDIT_USER_AGENT
```

### API Rate Limits

Reddit's API has rate limits:
- 60 requests per minute for authenticated requests
- 30 requests per minute for unauthenticated requests

If you hit rate limits, the script will pause automatically.

## Security Notes

- **Never commit your credentials to git**
- Use environment variables or `.env` files (and add `.env` to `.gitignore`)
- Keep your `client_secret` private
- Don't share your Reddit API credentials

