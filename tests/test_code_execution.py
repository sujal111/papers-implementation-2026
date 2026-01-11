import sys
from pathlib import Path

# Add the parent directory to the path so we can import the RLM
sys.path.append(str(Path(__file__).parent.parent))
from rlm.openai_rlm import OpenAIRLM, OpenAIRLMOptions

def test_safe_code_execution():
    """Test that code execution works with safe operations."""
    options = OpenAIRLMOptions(
        model="gpt-3.5-turbo",
        verbose=True
    )
    rlm = OpenAIRLM(options)
    
    print("\n=== Testing Safe Code Execution ===")
    print("Testing with a factorial calculation...")
    
    result = rlm.process("""
    Calculate the factorial of 5 using Python code and store it in the output variable.
    You can use a loop or recursion to implement the factorial function.
    """)
    
    print("\nTest Results:")
    print(f"Success: {result['success']}")
    print(f"Output: {result['output']}")
    
    assert result['success'] is True
    assert "120" in result['output']  # 5! = 120
    print("✓ Test passed!")

if __name__ == "__main__":
    test_safe_code_execution()
