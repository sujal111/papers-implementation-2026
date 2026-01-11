import os
import re
import json
import openai
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class RLMOptions:
    """Configuration options for the RLM."""
    model: str = "gpt-4"
    max_tokens: int = 4000
    temperature: float = 0.2
    max_recursion_depth: int = 5
    verbose: bool = True

class RLM:
    """Recursive Language Model implementation."""
    
    def __init__(self, options: Optional[RLMOptions] = None):
        """Initialize the RLM with the given options."""
        self.options = options or RLMOptions()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.context = ""
        self.execution_history = []
    
    def _call_llm(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Make a call to the underlying LLM."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.options.model,
                messages=messages,
                temperature=self.options.temperature,
                max_tokens=self.options.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {str(e)}")
            raise
    
    def _execute_code(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Safely execute Python code in a restricted environment."""
        # Create a safe execution environment
        safe_globals = {
            '__builtins__': {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'sum': sum,
                'min': min,
                'max': max,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'isinstance': isinstance,
                'type': type,
                'print': print,
            },
            'context': context.copy(),
        }
        
        try:
            # Execute the code in the safe environment
            exec(code, safe_globals, {})
            return {
                'success': True,
                'context': safe_globals.get('context', context),
                'output': safe_globals.get('output', None),
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'context': context,
                'output': None,
                'error': str(e)
            }
    
    def process(self, prompt: str, context: Optional[Dict[str, Any]] = None, depth: int = 0) -> Dict[str, Any]:
        """Process a prompt using the RLM approach."""
        if depth >= self.options.max_recursion_depth:
            return {
                'success': False,
                'output': None,
                'error': 'Maximum recursion depth reached',
                'context': context or {}
            }
        
        # Initialize context if not provided
        context = context or {}
        
        # Create system message
        system_message = """You are a Recursive Language Model (RLM) that can process long contexts by breaking them down into smaller, manageable pieces.
        You have access to a Python environment where you can execute code to process and analyze the context.
        
        Your task is to:
        1. Analyze the user's request and the current context
        2. If needed, write Python code to process the context in chunks
        3. Use the 'output' variable to store your final response
        4. Keep your code focused and efficient
        
        You can access the context dictionary to store and retrieve information between code executions.
        """
        
        # Call the LLM to generate code
        llm_response = self._call_llm(prompt, system_message)
        
        # Extract code blocks from the response
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', llm_response, re.DOTALL)
        
        if not code_blocks:
            # No code blocks found, return the raw response
            return {
                'success': True,
                'output': llm_response,
                'context': context,
                'code_executed': None
            }
        
        # Execute each code block
        execution_results = []
        for code in code_blocks:
            result = self._execute_code(code, context)
            execution_results.append({
                'code': code,
                'result': result
            })
            
            if not result['success']:
                # If execution failed, return the error
                return {
                    'success': False,
                    'output': f"Error executing code: {result['error']}",
                    'context': result['context'],
                    'execution_results': execution_results
                }
            
            # Update context for next iteration
            context = result['context']
        
        # Check if we need to make a recursive call
        if 'next_prompt' in context:
            next_prompt = context.pop('next_prompt')
            return self.process(next_prompt, context, depth + 1)
        
        # Return the final result
        return {
            'success': True,
            'output': context.get('output', llm_response),
            'context': context,
            'execution_results': execution_results
        }
