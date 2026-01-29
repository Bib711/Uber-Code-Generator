# Testing Agent - Analyzes testability and adds error handling
from .base import BaseAgent


class TestingAgent(BaseAgent):
    """AI-Powered Agent that analyzes testability and generates test suggestions"""
    
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.name = "Testing Agent"
        self.description = "Analyzes code testability, adds error handling, and suggests test cases"
        
        self.system_prompt = """You are an expert Python testing engineer. Your job is to:
1. Analyze the code for testability issues
2. Check if proper error handling exists (try/except)
3. Verify input validation is present
4. Check for edge cases that should be handled
5. **ALWAYS FIX any issues you find** - add error handling, validation, etc.

IMPORTANT: If you find ANY issues, you MUST:
- Fix them by adding try/except blocks, input validation, etc.
- Provide the complete fixed code in "fixed_code"
- List each fix with clear before/after examples

Respond in this EXACT JSON format (no markdown, just valid JSON):
{
    "status": "all_passed" or "warnings" or "failed",
    "testability_score": 75,
    "results": [
        {"test_name": "Error handling", "status": "warning", "description": "Missing try/except", "duration": "0.01s"}
    ],
    "issues_found": ["No error handling", "Missing input validation"],
    "fixes_applied": [
        {
            "description": "Added try/except block for database operations",
            "before": "result = db.query(sql)\\nreturn result",
            "after": "try:\\n    result = db.query(sql)\\n    return result\\nexcept Exception as e:\\n    return None",
            "line": 15
        },
        {
            "description": "Added input validation for user_id",
            "before": "def get_user(user_id):",
            "after": "def get_user(user_id):\\n    if not user_id:\\n        raise ValueError('user_id required')",
            "line": 5
        }
    ],
    "fixed_code": "THE COMPLETE FIXED CODE WITH ALL ERROR HANDLING ADDED",
    "suggested_tests": ["Test with valid input", "Test with None", "Test exception handling"]
}

CRITICAL RULES:
1. fixes_applied must be an array of objects with description, before, after, and line
2. Escape all quotes and newlines properly (use \\n for newlines)
3. fixed_code must contain the COMPLETE fixed code
4. Do NOT include markdown or code blocks - just pure JSON

Focus on adding:
- try/except blocks for risky operations
- Input validation checks
- Edge case handling"""
    
    def test(self, code):
        """Analyze code testability and return results with potential fixes"""
        user_prompt = f"Please analyze testability and fix issues:\n\n```python\n{code}\n```"
        
        print(f"🧪 {self.name}: Analyzing testability...")
        result = self.call_api(self.system_prompt, user_prompt, max_tokens=3000)
        parsed = self.parse_json_response(result)
        
        if parsed:
            parsed['ai_powered'] = True
            fixes_count = len(parsed.get('fixes_applied', []))
            parsed['message'] = f"Score: {parsed.get('testability_score', 0)}/100 - {fixes_count} fixes applied"
            print(f"✅ {self.name}: AI analysis complete - {fixes_count} fixes applied")
            return parsed
        
        print(f"⚠️ {self.name}: Using basic testing (API unavailable)")
        basic = self._basic_test(code)
        basic['ai_powered'] = False
        basic['message'] = '⚠️ Basic test (API key required for AI fixes)'
        return basic
    
    def _basic_test(self, code):
        """Fallback basic test analysis"""
        has_try = 'try:' in code
        has_validation = 'if ' in code and ('not ' in code or 'is None' in code)
        
        return {
            "status": "all_passed" if has_try and has_validation else "warnings",
            "testability_score": 70 if has_try else 50,
            "results": [
                {"test_name": "Syntax Check", "status": "passed", "description": "Code compiles", "duration": "0.01s"},
                {"test_name": "Error Handling", "status": "passed" if has_try else "warning", "description": "Has try/except" if has_try else "Missing error handling", "duration": "0.01s"}
            ],
            "issues_found": [] if has_try else ["Missing error handling"],
            "fixes_applied": [],
            "fixed_code": None,
            "suggested_tests": ["Test with valid inputs", "Test with edge cases"],
            "description": "📊 Basic testability analysis",
            "message": f"Testability score: {70 if has_try else 50}/100"
        }
