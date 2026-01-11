import sys
from pathlib import Path

# Add the parent directory to the path so we can import the RLM
sys.path.append(str(Path(__file__).parent.parent))
from rlm.openai_rlm import OpenAIRLM, OpenAIRLMOptions

def test_restricted_code_execution():
    """Test that unsafe operations are properly restricted."""
    options = OpenAIRLMOptions(
        model="gpt-3.5-turbo",
        verbose=True
    )
    rlm = OpenAIRLM(options)
    
    print("\n=== Testing Restricted Code Execution ===")
    print("Testing with restricted import...")
    
    result = rlm.process("""
    Try to import the 'os' module and list files in the current directory.
    This should fail due to security restrictions.
    """)
    
    print("\nTest Results:")
    print(f"Success: {result['success']}")
    print(f"Error: {result.get('error', 'No error')}")
    
    assert result['success'] is False
    assert any(keyword in result.get('error', '').lower() 
              for keyword in ['import', 'restricted', 'security', 'not allowed'])
    print("✓ Test passed! (Security restriction enforced)")

if __name__ == "__main__":
    test_restricted_code_execution()
