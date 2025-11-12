import json
import pandas as pd
import numpy as np
from jinja2 import Template
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM
from langchain_aws import BedrockLLM
from pathlib import Path
from typing import Optional, Union


# Define structured output for summarization
class SummaryResponse(BaseModel):
    summary: str = Field(description="A concise summary of the posts for the given category.")


def format_comments(comments) -> str:
    """
    Format comments list into a readable string.
    Handles different comment formats (list, string, numpy array, etc.)
    """
    # Handle None or NaN
    if comments is None:
        return "No comments"
    
    # Handle numpy arrays
    if isinstance(comments, np.ndarray):
        if comments.size == 0:
            return "No comments"
        comments = comments.tolist()
    
    # Handle lists
    if isinstance(comments, list):
        if len(comments) == 0:
            return "No comments"
        
        # Format each comment with a bullet point
        formatted = []
        for i, comment in enumerate(comments[:10], 1):  # Limit to first 10 comments
            if isinstance(comment, dict):
                # If comment is a dict, extract text field
                comment_text = comment.get('text', comment.get('body', str(comment)))
            else:
                comment_text = str(comment)
            
            # Clean and truncate comment
            comment_text = comment_text.strip()
            if len(comment_text) > 200:
                comment_text = comment_text[:200] + "..."
            
            formatted.append(f"  - {comment_text}")
        
        if len(comments) > 10:
            formatted.append(f"  ... and {len(comments) - 10} more comments")
        
        return "\n".join(formatted)
    
    # Handle string or other types
    try:
        if pd.isna(comments):
            return "No comments"
    except (ValueError, TypeError):
        pass
    
    return str(comments)[:500]  # Fallback for other formats


def format_posts_for_prompt(df: pd.DataFrame, category: str, title_col: str = "Title", 
                           content_col: str = "Post_Text", comments_col: str = "Comments") -> str:
    """
    Format posts from a dataframe for a given category into a string for the prompt.
    
    Args:
        df: DataFrame containing the posts
        category: Category name to filter posts
        title_col: Name of the title column
        content_col: Name of the content column
        comments_col: Name of the comments column
    
    Returns:
        Formatted string of posts ready for the prompt
    """
    # Filter posts for this category
    category_posts = df[df['category'] == category].copy()
    
    if len(category_posts) == 0:
        return ""
    
    # Format posts data for the prompt (with comments included)
    posts_list = []
    for idx, row in category_posts.iterrows():
        title = row[title_col] if title_col in row else "No title"
        content = row[content_col] if pd.notna(row.get(content_col)) else "No content"
        comments = format_comments(row.get(comments_col))
        
        post_text = f"""Post {len(posts_list) + 1}:
Title: {title}
Content: {content}
Comments:
{comments}"""
        
        posts_list.append(post_text)
    
    posts_text = "\n\n" + "\n\n---\n\n".join(posts_list)
    return posts_text


def render_prompt(prompt_file: str, posts_data: str) -> str:
    """
    Load and render Jinja prompt template.
    
    Args:
        prompt_file: Path to the Jinja2 prompt template file
        posts_data: Formatted string of posts to include in the prompt
    
    Returns:
        Rendered prompt string
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        template = Template(f.read())
    return template.render(posts=posts_data)


def summarize_posts(posts_data: str, llm_model, prompt_file: str, 
                   verbose: bool = False) -> SummaryResponse:
    """
    Summarize posts using an LLM model.
    
    Args:
        posts_data: Formatted string of posts to summarize
        llm_model: An LLM instance (Ollama or Bedrock)
        prompt_file: Path to Jinja2 prompt file
        verbose: Whether to print debug information
    
    Returns:
        SummaryResponse object containing the summary
    """
    # Prepare output parser
    parser = PydanticOutputParser(pydantic_object=SummaryResponse)

    # Render the prompt from file
    rendered_prompt = render_prompt(prompt_file, posts_data)

    # Add parser instructions so the model outputs valid JSON
    format_instructions = parser.get_format_instructions()
    full_prompt = f"{rendered_prompt}\n\nFollow the format below:\n{format_instructions}"

    if verbose:
        print("=== PROMPT PREVIEW ===")
        print(full_prompt[:500] + "...\n")

    # Run the LLM and get output
    response = llm_model.invoke(full_prompt)
    if isinstance(response, dict):
        output_text = response.get("content", "")
    elif hasattr(response, "content"):
        output_text = response.content
    else:
        output_text = str(response)

    if verbose:
        print("=== RAW MODEL OUTPUT ===")
        print(output_text[:500] + "...\n")

    # Try parsing structured output
    try:
        return parser.parse(output_text)
    except Exception as e:
        if verbose:
            print(f"Error parsing output: {e}")
        # fallback — return raw output as summary
        return SummaryResponse(summary=output_text.strip())


def summarize_category(
    df: pd.DataFrame,
    category: str,
    prompt_file: str,
    llm_model,
    title_col: str = "Title",
    content_col: str = "Post_Text",
    comments_col: str = "Comments",
    verbose: bool = False
) -> str:
    """
    Main function to summarize posts for a given category.
    
    Args:
        df: DataFrame containing the posts with a 'category' column
        category: Category name to filter and summarize
        prompt_file: Path to the Jinja2 prompt template file
        llm_model: LLM model instance (from get_ollama_model() or get_bedrock_model())
        title_col: Name of the title column (default: "Title")
        content_col: Name of the content column (default: "Post_Text")
        comments_col: Name of the comments column (default: "Comments")
        verbose: Whether to print debug information (default: False)
    
    Returns:
        Summary string for the category
    
    Raises:
        ValueError: If category column is missing or category not found
    """
    # Validate inputs
    if 'category' not in df.columns:
        raise ValueError("DataFrame must contain a 'category' column")
    
    category_posts = df[df['category'] == category]
    if len(category_posts) == 0:
        raise ValueError(f"No posts found for category '{category}'")
    
    if verbose:
        print(f"Processing category: {category}")
        print(f"Found {len(category_posts)} posts\n")
    
    # Format posts for prompt
    posts_text = format_posts_for_prompt(
        df, category, title_col, content_col, comments_col
    )
    
    if not posts_text:
        return "No posts to summarize."
    
    if verbose:
        print(f"Total posts text length: {len(posts_text)} characters\n")
    
    # Summarize using LLM
    summary_response = summarize_posts(posts_text, llm_model, prompt_file, verbose)
    
    return summary_response.summary


# Functions to create model clients
def get_ollama_model(model_name: str = "mistral:latest"):
    """Get an Ollama LLM model instance."""
    return OllamaLLM(model=model_name)


def get_bedrock_model(model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
    """Get an AWS Bedrock LLM model instance."""
    return BedrockLLM(model_id=model_id)


def process_all_prompts(
    df: pd.DataFrame,
    category_prompt_map: dict,
    prompts_dir: str = "prompts",
    llm_model=None,
    output_dir: str = "summaries",
    verbose: bool = False
) -> dict:
    """
    Process all categories with their corresponding prompt files.
    
    Args:
        df: DataFrame containing the posts with a 'category' column
        category_prompt_map: Dictionary mapping category names to prompt filenames
                           Example: {"Computer Science": "comp_sci.jinja", 
                                    "Social": "social.jinja"}
        prompts_dir: Directory containing prompt files
        llm_model: LLM model instance (if None, will use default Ollama model)
        output_dir: Directory to save summary files
        verbose: Whether to print debug information
    
    Returns:
        Dictionary mapping category names to summaries
    """
    from pathlib import Path
    
    # Initialize LLM if not provided
    if llm_model is None:
        llm_model = get_ollama_model("mistral:latest")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Validate inputs
    if not category_prompt_map:
        print("No categories provided in category_prompt_map")
        return {}
    
    prompts_path = Path(prompts_dir)
    
    print(f"Processing {len(category_prompt_map)} categories\n")
    
    all_summaries = {}
    successful = 0
    skipped = 0
    errors = 0
    
    # Process each category
    for category, prompt_filename in category_prompt_map.items():
        
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"Prompt file: {prompt_filename}")
        print(f"{'='*60}")
        
        # Check if prompt file exists
        prompt_file = prompts_path / prompt_filename
        if not prompt_file.exists():
            print(f"Prompt file '{prompt_filename}' not found. Skipping...")
            errors += 1
            continue
        
        # Check if category exists in dataframe
        if 'category' not in df.columns:
            print(f"Error: 'category' column not found in dataframe. Skipping...")
            errors += 1
            continue
        
        category_posts = df[df['category'] == category]
        
        if len(category_posts) == 0:
            print(f"Category '{category}' not found in data. Skipping...")
            skipped += 1
            continue
        
        print(f"Found {len(category_posts)} posts for '{category}'\n")
        
        # Summarize the category
        try:
            summary = summarize_category(
                df=df,
                category=category,
                prompt_file=str(prompt_file),
                llm_model=llm_model,
                verbose=verbose
            )
            
            all_summaries[category] = summary
            
            print(f"\n=== SUMMARY FOR {category} ===")
            print(summary[:500] + "..." if len(summary) > 500 else summary)
            
            # Save individual summary to file
            safe_category_name = "".join(c for c in category if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_category_name = safe_category_name.replace(' ', '_')
            summary_file = output_path / f"{safe_category_name}_summary.txt"
            
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(f"Category: {category}\n")
                f.write(f"Number of posts: {len(category_posts)}\n")
                f.write(f"Prompt file: {prompt_filename}\n")
                f.write(f"{'='*60}\n\n")
                f.write(summary)
            
            print(f"\nSummary saved to {summary_file}")
            successful += 1
            
        except Exception as e:
            print(f"Error processing '{category}': {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            all_summaries[category] = f"Error: {str(e)}"
            errors += 1
            continue
    
    # Save combined summary file
    if all_summaries:
        combined_file = output_path / "all_categories_summary.txt"
        with open(combined_file, "w", encoding="utf-8") as f:
            f.write("REDDIT POSTS SUMMARY - ALL CATEGORIES\n")
            f.write("="*60 + "\n\n")
            
            for category, summary in all_summaries.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"CATEGORY: {category}\n")
                f.write(f"{'='*60}\n\n")
                f.write(summary)
                f.write("\n\n")
        
        print(f"\nCombined summary saved to {combined_file}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Skipped (category not found): {skipped}")
    print(f"  Errors: {errors}")
    print(f"{'='*60}\n")
    
    return all_summaries


# Example usage
if __name__ == "__main__":
    # Load dataframe
    data_file = "reddit_data_classified/ubc_2025_11_12_2025_11_05.parquet"
    df = pd.read_parquet(data_file)
    
    # Initialize LLM
    llm = get_ollama_model("mistral:latest")
    
    # Define category to prompt file mapping
    # Format: {"Category Name": "prompt_filename.jinja"}
    category_prompt_map = {
        "Computer Science": "Computer_Science.jinja",
        "Social": "Social_Events.jinja",
        "General Academics": "General_Academics.jinja",
        "General Sciences": "General_Sciences.jinja",
        "Mental Health and Wellbeing": "Mental_Health_and_Wellbeing.jinja",
        "Math and Statistics": "Math_and_Statistics.jinja",
        "Campus Spaces": "Campus_Spaces.jinja",
        "Career": "Career.jinja",
        "Business and Econ": "Business_and_Econ.jinja",
        "Housing and Residence": "Housing_and_Residence.jinja",
        "Admin and Logistics": "Admin_and_Logistics.jinja",
        "Arts and Humanities": "Arts_and_Humanities.jinja",
        "Rants": "Rants_and_Complaints.jinja",
        "Advice and Tips": "Advice_and_Tips.jinja",
    }
    
    # Process all categories
    summaries = process_all_prompts(
        df=df,
        category_prompt_map=category_prompt_map,
        prompts_dir="prompts",
        llm_model=llm,
        output_dir="summaries",
        verbose=False  # Set to True for debug output
    )
    
    print(f"\nProcessed {len(summaries)} categories successfully!")
