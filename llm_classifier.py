import json
from jinja2 import Template
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.llms import Ollama
from langchain_aws import BedrockLLM  # make sure `langchain-aws` is installed

# Define structured output
class CategoryResponse(BaseModel):
    category: str = Field(description="Predicted category name from the predefined list.")

# Function to load and render Jinja prompt
def render_prompt(prompt_file: str, content: str) -> str:
    with open(prompt_file, "r") as f:
        template = Template(f.read())
    return template.render(content=content)

# Functions to create model clients
def get_ollama_model(model_name: str = "llama2:latest"):
    return Ollama(model=model_name)

def get_bedrock_model(model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
    return BedrockLLM(model_id=model_id)

# Main classification function
def classify_text(content: str, llm_model, prompt_file: str) -> CategoryResponse:
    """
    content: text to classify
    llm_model: an LLM instance (Ollama or Bedrock)
    prompt_file: path to Jinja2 prompt file
    """
    # Prepare output parser
    parser = PydanticOutputParser(pydantic_object=CategoryResponse)

    # Render the prompt from file
    rendered_prompt = render_prompt(prompt_file, content)

    # Add parser instructions so the model outputs valid JSON
    format_instructions = parser.get_format_instructions()
    full_prompt = f"{rendered_prompt}\n\nFollow the format below:\n{format_instructions}"

    # Run the LLM and get output
    response = llm_model.invoke(full_prompt)
    if isinstance(response, dict):
        output_text = response.get("content", "")
    elif hasattr(response, "content"):
        output_text = response.content
    else:
        output_text = str(response)

    # Try parsing structured output
    try:
        return parser.parse(output_text)
    except Exception:
        # fallback — extract category heuristically
        cleaned = output_text.strip().split("\n")[0]
        return CategoryResponse(category=cleaned)

# ---------------------------------------------------------------------------
# Optional: Factories for different model providers
# ---------------------------------------------------------------------------
def get_ollama_model(model_name: str):
    """Return an Ollama model."""
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model=model_name)


def get_bedrock_model(
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    region: str = "us-east-1",
):
    """Return an AWS Bedrock model."""
    from langchain_aws import ChatBedrock
    return ChatBedrock(
        model_id=model_id,
        region_name=region,
        model_kwargs={"temperature": 0},
    )


def get_openai_model(model_name: str = "gpt-4o-mini"):
    """Return an OpenAI model."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_name, temperature=0)



