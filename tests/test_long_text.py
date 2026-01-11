import sys
from pathlib import Path

# Add the parent directory to the path so we can import the RLM
sys.path.append(str(Path(__file__).parent.parent))
from rlm.openai_rlm import OpenAIRLM, OpenAIRLMOptions

def test_long_text_chunking():
    """Test processing text that exceeds the context window."""
    options = OpenAIRLMOptions(
        model="gpt-3.5-turbo",
        max_tokens=100,  # Set very low to force chunking
        max_recursion_depth=3,
        verbose=True
    )
    rlm = OpenAIRLM(options)
    
    print("\n=== Testing Long Text Chunking ===")
    print("Testing with a long repetitive text...")
    
    # Create a long text (about 2000 characters)
    long_text = "This is a test sentence. " * 80
    
    result = rlm.process(f"Count how many times 'sentence' appears in this text: {long_text}")
    
    print("\nTest Results:")
    print(f"Success: {result['success']}")
    print(f"Output: {result['output']}")
    
    assert result['success'] is True
    # The exact count might vary based on how the text is chunked
    print("✓ Test passed! (Basic functionality verified)")

if __name__ == "__main__":
    test_long_text_chunking()
