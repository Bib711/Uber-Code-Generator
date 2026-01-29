# Validation Agent - Validates code and auto-fixes issues
from .base import BaseAgent


class ValidationAgent(BaseAgent):
    """AI-Powered Agent that validates code and auto-fixes issues"""
    
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.name = "Validation Agent"
        self.description = "Validates code for style, syntax, and best practices with auto-fix capability"
        
        self.system_prompt = """You are an expert Python code reviewer and validator. Your job is to:
1. Analyze the code for syntax errors, style issues, and best practice violations
2. Check for PEP8 compliance, proper indentation, naming conventions
3. Look for missing docstrings, type hints, and code organization issues
4. **ALWAYS FIX any issues you find** - don't just report them!

IMPORTANT: If you find ANY issues, you MUST:
- Fix them in the code
- Provide the complete fixed code in "fixed_code"
- List each fix with clear before/after examples

Respond in this EXACT JSON format (no markdown, just valid JSON):
{
    "status": "passed" or "warnings" or "failed",
    "issues": ["issue 1", "issue 2"],
    "warnings": ["warning 1"],
    "suggestions": ["suggestion 1"],
    "stats": {
        "functions": 5,
        "classes": 2,
        "lines": 100,
        "has_docstrings": false,
        "has_type_hints": false
    },
    "fixes_applied": [
        {
            "description": "Added docstring to function calculate()",
            "before": "def calculate(x, y):",
            "after": "def calculate(x, y):\\n    \\\"\\\"\\\"Calculate sum of two numbers.\\\"\\\"\\\"",
            "line": 10
        },
        {
            "description": "Added type hints to process()",
            "before": "def process(data):",
            "after": "def process(data: dict) -> bool:",
            "line": 25
        }
    ],
    "fixed_code": "THE COMPLETE FIXED CODE HERE"
}

CRITICAL RULES:
1. fixes_applied must be an array of objects with description, before, after, and line
2. Escape all quotes and newlines properly in strings
3. fixed_code must contain the COMPLETE fixed code
4. Do NOT include markdown or code blocks - just pure JSON"""
    
    def validate(self, code):
        """Validate code and return analysis with potential fixes"""
        user_prompt = f"Please validate and fix if needed:\n\n```python\n{code}\n```"
        
        print(f"🔍 {self.name}: Analyzing code...")
        result = self.call_api(self.system_prompt, user_prompt, max_tokens=3000)
        parsed = self.parse_json_response(result)
        
        if parsed:
            parsed['ai_powered'] = True
            parsed['message'] = f"{'✅ Validation passed' if parsed.get('status') == 'passed' else '⚠️ Found issues'} - {len(parsed.get('issues', []))} issues, {len(parsed.get('fixes_applied', []))} fixes"
            print(f"✅ {self.name}: AI analysis complete - {len(parsed.get('fixes_applied', []))} fixes applied")
            return parsed
        
        # Fallback to basic validation
        print(f"⚠️ {self.name}: Using basic validation (API unavailable)")
        basic = self._basic_validate(code)
        basic['ai_powered'] = False
        basic['message'] = '⚠️ Basic validation (API key required for AI fixes)'
        return basic
    
    def _basic_validate(self, code):
        """Fallback basic validation"""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(f"Line {i}: exceeds 120 characters")
            if '\t' in line:
                issues.append(f"Line {i}: uses tabs instead of spaces")
        
        if "import *" in code:
            issues.append("Wildcard imports detected - avoid 'from x import *'")
        
        return {
            "status": "passed" if len(issues) == 0 else "warnings",
            "issues": issues,
            "warnings": [],
            "suggestions": [],
            "stats": {"functions": code.count('def '), "classes": code.count('class '), "lines": len(lines)},
            "fixes_applied": [],
            "fixed_code": None,
            "description": f"📊 Basic validation: {len(issues)} issues found",
            "message": f"Found {len(issues)} issue(s)"
        }
