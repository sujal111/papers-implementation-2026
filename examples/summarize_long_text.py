import os
from pathlib import Path
from dotenv import load_dotenv
import sys

# Add the parent directory to the path so we can import the RLM
sys.path.append(str(Path(__file__).parent.parent))
from rlm.core import RLM, RLMOptions

def load_long_text(file_path: str) -> str:
    """Load a long text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize the RLM with custom options
    options = RLMOptions(
        model="gpt-4",  # or any other supported model
        max_tokens=4000,
        temperature=0.2,
        max_recursion_depth=3,
        verbose=True
    )
    rlm = RLM(options)
    
    # Example: Summarize a long text
    print("Loading long text...")
    long_text = """
    [Your long text here. This could be any lengthy document or article.
    For demonstration, we'll use a placeholder. In a real scenario, you would
    load this from a file or another source.]
    
    The field of artificial intelligence has made significant progress in recent years.
    Large language models like GPT-4 have demonstrated remarkable capabilities in
    understanding and generating human-like text. However, one of the key challenges
    remains the effective processing of very long contexts.
    
    [Additional paragraphs of text...]
    """
    
    # Create a prompt for summarization
    prompt = f"""
    Please analyze the following text and provide a concise summary.
    The text is quite long, so you may want to process it in chunks.
    
    TEXT TO SUMMARIZE:
    {long_text}
    
    INSTRUCTIONS:
    1. First, analyze the structure and key points of the text.
    2. Break down the text into logical sections if needed.
    3. Generate a summary that captures the main ideas and key details.
    4. Store your final summary in the 'output' variable.
    
    You can use Python code to help with the analysis and processing.
    """
    
    print("Processing with RLM...")
    result = rlm.process(prompt)
    
    if result['success']:
        print("\n=== SUMMARY ===")
        print(result['output'])
        
        if 'execution_results' in result and result['execution_results']:
            print("\n=== EXECUTION STEPS ===")
            for i, step in enumerate(result['execution_results'], 1):
                print(f"\nStep {i}:")
                print(f"Code executed:\n{step['code']}")
                print(f"Output: {step['result']['output']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
