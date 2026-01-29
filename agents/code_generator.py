# Code Generator Agent - Generates code using AI
import json
import requests
from .base import BaseAgent, GROQ_BASE_URL, MODEL


class CodeGeneratorAgent(BaseAgent):
    """Agent that generates code using AI"""
    
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.name = "Code Generator"
        self.description = "Generates clean, production-ready code from natural language prompts"
        
        self.system_prompt = """You are an expert code generator. Generate clean, well-documented, 
production-ready code based on the user's request. Include helpful comments and follow best practices.
Only output the code, no explanations before or after. Do not wrap code in markdown code blocks."""
        
    def generate(self, prompt):
        """Non-streaming code generation"""
        result = self.call_api(self.system_prompt, prompt)
        if result:
            # Remove markdown code blocks if present
            if result.startswith('```'):
                lines = result.split('\n')
                result = '\n'.join(lines[1:-1]) if lines[-1].strip() == '```' else '\n'.join(lines[1:])
            return result
        return self._mock_generate(prompt)
    
    def generate_stream(self, prompt):
        """Streaming code generation - yields chunks"""
        try:
            response = requests.post(
                GROQ_BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "stream": True
                },
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
            else:
                print(f"Stream Error: {response.status_code}")
                for chunk in self._mock_stream(prompt):
                    yield chunk
        except Exception as e:
            print(f"Stream Error: {e}")
            for chunk in self._mock_stream(prompt):
                yield chunk
    
    def _mock_stream(self, prompt):
        """Mock streaming for fallback"""
        code = self._mock_generate(prompt)
        chunk_size = 20
        for i in range(0, len(code), chunk_size):
            yield code[i:i+chunk_size]
    
    def _mock_generate(self, prompt):
        """Fallback mock generation"""
        return f'''def solution(param1, param2):
    """Generated function for: {prompt[:30]}..."""
    result = param1 + param2
    return result

if __name__ == "__main__":
    print(solution(10, 20))'''
