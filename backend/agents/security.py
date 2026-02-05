# Security Agent - Scans for vulnerabilities and fixes them
from .base import BaseAgent


class SecurityAgent(BaseAgent):
    """AI-Powered Agent that scans for security vulnerabilities and fixes them"""
    
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.name = "Security Agent"
        self.description = "Scans for security vulnerabilities and automatically applies fixes"
        
        self.system_prompt = """You are an expert cybersecurity engineer specializing in Python security. Your job is to:
1. Scan for ALL security vulnerabilities (injection, XSS, CSRF, etc.)
2. Check for dangerous functions (eval, exec, pickle, os.system, etc.)
3. Look for hardcoded secrets, credentials, API keys
4. Check for insecure cryptographic practices
5. Identify SQL injection, command injection vulnerabilities
6. **ALWAYS FIX any vulnerabilities you find** - don't just report them!

Common fixes you should apply:
- eval() → ast.literal_eval() or proper parsing
- exec() → safer alternatives or remove entirely
- pickle → json for data serialization
- os.system() → subprocess.run() with shell=False
- Hardcoded passwords → os.environ.get('PASSWORD')
- SQL concatenation → parameterized queries

Respond in this EXACT JSON format (no markdown, just valid JSON):
{
    "status": "secure" or "vulnerabilities_found",
    "risk_level": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
    "risk_score": 25,
    "vulnerabilities": [
        {"pattern": "eval(user_input)", "severity": "CRITICAL", "type": "Code Injection", "description": "Allows arbitrary code execution", "line": 15}
    ],
    "warnings": [],
    "fixes_applied": [
        {
            "description": "Replaced eval() with ast.literal_eval() for safe parsing",
            "severity": "CRITICAL",
            "before": "result = eval(user_input)",
            "after": "import ast\\nresult = ast.literal_eval(user_input)",
            "line": 15
        },
        {
            "description": "Replaced hardcoded password with environment variable",
            "severity": "HIGH",
            "before": "password = 'admin123'",
            "after": "password = os.environ.get('DB_PASSWORD', '')",
            "line": 8
        }
    ],
    "fixed_code": "THE COMPLETE SECURE CODE WITH ALL VULNERABILITIES FIXED",
    "recommendations": ["Add input validation", "Implement rate limiting"]
}

CRITICAL RULES:
1. fixes_applied must be an array of objects with description, severity, before, after, and line
2. Escape all quotes and newlines properly (use \\n for newlines)
3. fixed_code must contain the COMPLETE fixed code
4. Do NOT include markdown or code blocks - just pure JSON
5. Include severity level (CRITICAL/HIGH/MEDIUM/LOW) for each fix"""
    
    def scan(self, code):
        """Scan code for security vulnerabilities and return results with fixes"""
        user_prompt = f"Please perform a security audit and fix vulnerabilities:\n\n```python\n{code}\n```"
        
        print(f"🛡️ {self.name}: Scanning for vulnerabilities...")
        result = self.call_api(self.system_prompt, user_prompt, max_tokens=3000)
        parsed = self.parse_json_response(result)
        
        if parsed:
            parsed['ai_powered'] = True
            vuln_count = len(parsed.get('vulnerabilities', []))
            fixes_count = len(parsed.get('fixes_applied', []))
            parsed['message'] = f"{'✅ Secure' if vuln_count == 0 else f'🚨 {vuln_count} vulnerabilities'} - {fixes_count} fixes applied"
            print(f"✅ {self.name}: AI scan complete - {fixes_count} security fixes applied")
            return parsed
        
        print(f"⚠️ {self.name}: Using basic scan (API unavailable)")
        basic = self._basic_scan(code)
        basic['ai_powered'] = False
        basic['message'] = '⚠️ Basic scan only - API key required for AI auto-fix'
        return basic
    
    def _basic_scan(self, code):
        """Fallback basic security scan"""
        vulnerabilities = []
        
        dangerous_patterns = [
            ("eval(", "CRITICAL", "Code Injection", "eval() can execute arbitrary code"),
            ("exec(", "CRITICAL", "Code Injection", "exec() can execute arbitrary code"),
            ("pickle.loads(", "HIGH", "Deserialization", "pickle.loads() can execute arbitrary code"),
            ("os.system(", "HIGH", "Command Injection", "os.system() is vulnerable to command injection"),
            ("subprocess.call(", "MEDIUM", "Command Injection", "Consider using subprocess.run with shell=False"),
            ("shell=True", "HIGH", "Command Injection", "shell=True enables command injection"),
            ("password =", "MEDIUM", "Hardcoded Secret", "Possible hardcoded password"),
            ("api_key =", "MEDIUM", "Hardcoded Secret", "Possible hardcoded API key"),
            ("secret =", "MEDIUM", "Hardcoded Secret", "Possible hardcoded secret"),
        ]
        
        for pattern, severity, vuln_type, description in dangerous_patterns:
            if pattern.lower() in code.lower():
                vulnerabilities.append({
                    "pattern": pattern,
                    "severity": severity,
                    "type": vuln_type,
                    "description": description
                })
        
        risk_score = min(len(vulnerabilities) * 25, 100)
        risk_level = "LOW" if risk_score < 25 else "MEDIUM" if risk_score < 50 else "HIGH" if risk_score < 75 else "CRITICAL"
        
        return {
            "status": "secure" if not vulnerabilities else "vulnerabilities_found",
            "risk_level": risk_level,
            "risk_score": risk_score,
            "vulnerabilities": vulnerabilities,
            "warnings": [],
            "fixes_applied": [],
            "fixed_code": None,
            "recommendations": ["Review code manually for additional security issues"],
            "description": f"🛡️ Basic security scan: {len(vulnerabilities)} issues found",
            "message": f"Found {len(vulnerabilities)} vulnerability(ies)"
        }
