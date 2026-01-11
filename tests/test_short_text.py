import sys
from pathlib import Path

# Add the parent directory to the path so we can import the RLM
sys.path.append(str(Path(__file__).parent.parent))
from rlm.openai_rlm import OpenAIRLM, OpenAIRLMOptions

def test_short_text_processing():
    """Test processing text that fits within a single context window."""
    options = OpenAIRLMOptions(
        model="gpt-3.5-turbo",  # Using 3.5 for faster testing
        max_tokens=1000,
        verbose=True
    )
    rlm = OpenAIRLM(options)
    
    print("\n=== Testing Short Text Processing ===")
    print("Testing with simple arithmetic question...")
    
    result = rlm.process("What is 2+2?")
    
    print("\nTest Results:")
    print(f"Success: {result['success']}")
    print(f"Output: {result['output']}")
    
    assert result['success'] is True
    assert "4" in result['output']
    print("✓ Test passed!")

if __name__ == "__main__":
    test_short_text_processing()
