# Base Agent Module - Shared utilities for all agents
import os
import sys
import requests
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

# API Configuration from settings
GROQ_API_KEY = settings.GROQ_API_KEY
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = settings.GROQ_MODEL


def call_groq_api(api_key, system_prompt, user_prompt, max_tokens=2000):
    """Helper function to call Groq API"""
    # Use provided key or fall back to settings
    key = api_key if api_key and len(api_key) > 10 else GROQ_API_KEY
    
    # Check if API key is valid
    if not settings.is_valid_key(key):
        print(f"⚠️ Invalid or missing API key (length: {len(key) if key else 0})")
        return None
    
    try:
        print(f"🔄 Calling Groq API with model {MODEL}...")
        response = requests.post(
            GROQ_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": max_tokens
            },
            timeout=90  # Increased timeout for complex analysis
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"✅ API call successful, received {len(content)} chars")
            return content
        else:
            print(f"❌ API Error: {response.status_code} - {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        print(f"❌ API Timeout after 90 seconds")
        return None
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None


class BaseAgent:
    """Base class for all AI agents"""
    
    def __init__(self, api_key=None):
        # Use provided key, fall back to env var, but validate it
        self.api_key = api_key if api_key and len(api_key) > 10 else GROQ_API_KEY
        self.name = "Base Agent"
        self.description = "Base agent class"
        print(f"🤖 {self.__class__.__name__} initialized with API key: {'✓ Valid' if self.api_key and len(self.api_key) > 10 else '✗ Missing'}")
    
    def call_api(self, system_prompt, user_prompt, max_tokens=2000):
        """Call the Groq API with given prompts"""
        return call_groq_api(self.api_key, system_prompt, user_prompt, max_tokens)
    
    def parse_json_response(self, result):
        """Parse JSON from API response with robust error handling"""
        if result:
            try:
                # Try to find JSON in the response
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = result[json_start:json_end]
                    parsed = json.loads(json_str)
                    print(f"✅ Successfully parsed JSON response")
                    return parsed
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parse error: {e}")
                # Try multiple cleanup strategies
                try:
                    # Strategy 1: Remove markdown code blocks
                    clean = result.replace('```json', '').replace('```python', '').replace('```', '')
                    json_start = clean.find('{')
                    json_end = clean.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = clean[json_start:json_end]
                        return json.loads(json_str)
                except:
                    pass
                
                try:
                    # Strategy 2: Fix common issues - escape control chars
                    json_str = result[result.find('{'):result.rfind('}')+1]
                    # Remove the report field if it's causing issues (we'll generate it from fixes)
                    import re
                    # Remove problematic report field
                    json_str = re.sub(r',\s*"report"\s*:\s*"[^"]*(?:\\.[^"]*)*"', '', json_str)
                    json_str = re.sub(r'"report"\s*:\s*"[^"]*(?:\\.[^"]*)*"\s*,?', '', json_str)
                    return json.loads(json_str)
                except:
                    pass
                
                try:
                    # Strategy 3: Extract key fields manually using regex
                    import re
                    extracted = {}
                    
                    # Extract status
                    status_match = re.search(r'"status"\s*:\s*"([^"]+)"', result)
                    if status_match:
                        extracted['status'] = status_match.group(1)
                    
                    # Extract fixed_code
                    code_match = re.search(r'"fixed_code"\s*:\s*"((?:[^"\\]|\\.)*)"', result, re.DOTALL)
                    if code_match:
                        extracted['fixed_code'] = code_match.group(1).encode().decode('unicode_escape')
                    
                    # Extract fixes_applied as array
                    fixes_match = re.search(r'"fixes_applied"\s*:\s*\[(.*?)\]', result, re.DOTALL)
                    if fixes_match:
                        fixes_str = fixes_match.group(1)
                        fixes = re.findall(r'"([^"]+)"', fixes_str)
                        extracted['fixes_applied'] = fixes
                    
                    # Extract issues
                    issues_match = re.search(r'"issues"\s*:\s*\[(.*?)\]', result, re.DOTALL)
                    if issues_match:
                        issues_str = issues_match.group(1)
                        issues = re.findall(r'"([^"]+)"', issues_str)
                        extracted['issues'] = issues
                    
                    if extracted.get('status') or extracted.get('fixed_code'):
                        print(f"✅ Extracted partial JSON using regex")
                        return extracted
                except Exception as e2:
                    print(f"⚠️ Regex extraction failed: {e2}")
                    pass
        return None
