from reddit_fetcher import fetch_reddit_posts
from llm_classifier import classify_text, get_ollama_model, get_bedrock_model
import datetime as dt
import pandas as pd
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()  # Enable progress_apply on pandas

if __name__ == "__main__":
    # Dates for data fetching and file naming
    today_date = dt.datetime.now(dt.timezone.utc)
    old_date = today_date - dt.timedelta(days=7)

    today_date_str = today_date.strftime('%Y_%m_%d')
    old_date_str = old_date.strftime('%Y_%m_%d')

    subreddit = 'ubc'

    # Define folders
    output_folder = Path("reddit_data")
    output_folder.mkdir(parents=True, exist_ok=True)

    classified_folder = Path("reddit_data_classified")
    classified_folder.mkdir(parents=True, exist_ok=True)

    # Parquet file path based on naming convention
    parquet_file = output_folder / f"{subreddit}_{today_date_str}_{old_date_str}.parquet"
    print(f"Input parquet file: {parquet_file}")

    # Step 1: Fetch data only if file does not exist
    if parquet_file.exists():
        print(f"Data file exists: {parquet_file}. Loading existing data...")
        df = pd.read_parquet(parquet_file)
    else:
        print(f"Data file not found. Fetching new data for subreddit '{subreddit}'...")
        days_back = (today_date - old_date).days
        df = fetch_reddit_posts(
            subreddit_names=[subreddit],
            days_back=days_back,
            manual_date=today_date,
            output_folder=str(output_folder)
        )
        print(f"Fetched data for date range: {old_date.date()} to {today_date.date()}")

    # Step 2: Initialize LLM and prompt
    llm = get_ollama_model("mistral:latest")
    # llm = get_bedrock_model()
    prompt_file = "prompts/classify_post.jinja"

    # Step 3: Define columns containing text
    title_col = "Title"
    content_col = "Post_Text"

    # Step 4: Define classification function
    def classify_row_text(row_text: str) -> str:
        try:
            response = classify_text(content=row_text, llm_model=llm, prompt_file=prompt_file)
            return response.category
        except Exception as e:
            print(f"Error classifying text: {e}")
            return "unknown"

    # Step 5: Combine Title and Content, then classify if needed
    if 'category' not in df.columns or df['category'].isnull().all():
        print("Classifying posts to add 'category' column...")

        combined_texts = (df[title_col].fillna('') + ". " + df[content_col].fillna('')).str.strip()
        df['category'] = combined_texts.progress_apply(classify_row_text)

        # Step 6: Save classified dataframe in classified_folder
        classified_file = classified_folder / parquet_file.name
        df.to_parquet(classified_file, index=False)
        print(f"Saved classified data to {classified_file}")

    else:
        print("'category' column already exists with data. Skipping classification.")
