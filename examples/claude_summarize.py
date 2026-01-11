import os
from pathlib import Path
from dotenv import load_dotenv
import sys

# Add the parent directory to the path so we can import the RLM
sys.path.append(str(Path(__file__).parent.parent))
from rlm.openai_rlm import OpenAIRLM, OpenAIRLMOptions

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize the OpenAI RLM with custom options
    options = OpenAIRLMOptions(
        model="gpt-4-turbo",  # or "gpt-3.5-turbo" for faster, less expensive processing
        max_tokens=4000,
        temperature=0.2,
        max_recursion_depth=3,
        verbose=True
    )
    rlm = OpenAIRLM(options)
    
    # Example: Summarize a long text
    print("Loading long text...")
    long_text = """
    The field of artificial intelligence has made significant progress in recent years. 
    Large language models like GPT-4 have demonstrated remarkable capabilities in 
    understanding and generating human-like text. However, one of the key challenges 
    remains the effective processing of very long contexts.
    
    [Additional paragraphs of text...]
    
    Recursive Language Models (RLMs) offer a promising approach to this challenge by 
    breaking down long contexts into manageable chunks and processing them recursively.
    This allows for more effective handling of documents that exceed standard context windows.
    
    [More text about RLMs and their applications...]
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
    
    print("Processing with OpenAI RLM...")
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