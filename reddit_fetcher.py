import praw
import pandas as pd
import datetime as dt
import os


def fetch_reddit_posts(subreddit_names, days_back=7, manual_date=None, output_folder="reddit_data"):
    """
    Fetches Reddit posts from specified subreddits within a date range and saves them to Parquet files.
    Parquet format handles newlines and special characters in text without issues.
    
    Args:
        subreddit_names (list): List of subreddit names to fetch posts from
        days_back (int): Number of days to look back from the current date (default: 7)
        manual_date (datetime.datetime, optional): Manual date to use instead of current date.
                                                   Should be timezone-aware UTC datetime.
        output_folder (str): Folder path to save the parquet files (default: "reddit_data")
    
    Returns:
        dict: Dictionary mapping subreddit names to their DataFrames and file paths
    
    Raises:
        ValueError: If required environment variables are not set
    """
    # Get credentials from environment variables
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT')

    # Check if all required environment variables are set
    if not all([client_id, client_secret, user_agent]):
        raise ValueError(
            "Missing required environment variables. Please set:\n"
            "  - REDDIT_CLIENT_ID\n"
            "  - REDDIT_CLIENT_SECRET\n"
            "  - REDDIT_USER_AGENT\n"
        )

    reddit_read_only = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    results = {}
    
    for subreddit_name in subreddit_names:
        subreddit = reddit_read_only.subreddit(subreddit_name)
        # Use timezone-aware datetime (UTC)
        if manual_date is None:
            current_date = dt.datetime.now(dt.timezone.utc)
        else:
            # Ensure manual_date is timezone-aware UTC
            if manual_date.tzinfo is None:
                current_date = manual_date.replace(tzinfo=dt.timezone.utc)
            else:
                current_date = manual_date.astimezone(dt.timezone.utc)
        
        end_date = current_date - dt.timedelta(days=days_back)
        today_day = current_date.strftime('%Y_%m_%d')
        old_date = end_date.strftime('%Y_%m_%d')

        print(f"Fetching posts from r/{subreddit_name} between {end_date.date()} and {current_date.date()}")

        posts_dict = {
            "Title": [],
            "Post_Text": [],
            "Post_URL": [],
            "Comments": [],
            "Created_UTC": []
        }

        # Fetch newest posts
        # Use a limit to avoid fetching too many posts
        post_count = 0
        max_posts_to_check = 1000  # Limit how many posts we check
        
        for post in subreddit.new(limit=max_posts_to_check):
            post_count += 1
            # Use timezone-aware datetime.fromtimestamp instead of deprecated utcfromtimestamp
            post_time = dt.datetime.fromtimestamp(post.created_utc, tz=dt.timezone.utc)
            
            # Check if post is within date range
            if end_date <= post_time <= current_date:
                # Post is within range, add it
                title = post.title
                text = post.selftext
                url = post.url

                comments = []
                try:
                    post.comments.replace_more(limit=None)
                    comments = [comment.body for comment in post.comments.list()]
                except Exception as e:
                    comments = []

                # Even if no comments, append an empty list
                posts_dict["Title"].append(title)
                posts_dict["Post_Text"].append(text)
                posts_dict["Post_URL"].append(url)
                posts_dict["Comments"].append(comments if comments else [])
                posts_dict["Created_UTC"].append(post_time)
            elif post_time < end_date:
                # Post is older than end_date, we can break since posts are sorted by newest first
                # Once we hit posts older than our range, all subsequent posts will also be older
                break
            # If post_time > current_date, it's in the future, skip it but continue checking

        df = pd.DataFrame(posts_dict)
        df["Comments"] = df["Comments"].apply(lambda x: x if isinstance(x, list) else [])

        output_filename = os.path.join(output_folder, f"{subreddit_name}_{today_day}_{old_date}.parquet")
        df.to_parquet(output_filename, index=False, engine='pyarrow')
        print(f"Saved {len(df)} posts from r/{subreddit_name} -> {output_filename}")
        
        results[subreddit_name] = {
            'dataframe': df,
            'file_path': output_filename,
            'today_date': today_day,
            'old_date': old_date
        }
    
    return results
