# Orchestrator for Uber Code Generator
# Coordinates all AI-powered agents using LangGraph-style workflow

import networkx as nx
from datetime import datetime

# Import all agents from the agents package
from agents import (
    CodeGeneratorAgent,
    ValidationAgent,
    TestingAgent,
    SecurityAgent,
    AGENT_INFO
)


class Orchestrator:
    """LangGraph-style orchestrator that coordinates all AI-powered agents"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        
        # Initialize all agents
        self.code_generator = CodeGeneratorAgent(api_key)
        self.validator = ValidationAgent(api_key)
        self.tester = TestingAgent(api_key)
        self.security = SecurityAgent(api_key)
        
        # LangGraph-style workflow graph
        self.workflow_graph = nx.DiGraph()
        self._build_workflow_graph()
        
        # Memory for storing intermediate results
        self.memory = {}
        self.workflow_history = []
        self.all_fixes = []  # Track all fixes made by agents

    def _build_workflow_graph(self):
        """Build the LangGraph workflow DAG"""
        # Add nodes (agents)
        self.workflow_graph.add_node("code_generator", agent=self.code_generator)
        self.workflow_graph.add_node("validator", agent=self.validator)
        self.workflow_graph.add_node("tester", agent=self.tester)
        self.workflow_graph.add_node("security", agent=self.security)
        
        # Add edges (workflow connections)
        # Code flows: Generator -> Validator -> Tester -> Security
        self.workflow_graph.add_edge("code_generator", "validator")
        self.workflow_graph.add_edge("validator", "tester")
        self.workflow_graph.add_edge("tester", "security")

    def get_workflow_info(self):
        """Return workflow graph information"""
        return {
            "nodes": list(self.workflow_graph.nodes()),
            "edges": list(self.workflow_graph.edges()),
            "execution_order": list(nx.topological_sort(self.workflow_graph)),
            "agent_info": AGENT_INFO
        }

    def generate_code_stream(self, prompt):
        """Stream code generation - yields chunks for real-time display"""
        self.memory['latest_prompt'] = prompt
        for chunk in self.code_generator.generate_stream(prompt):
            yield chunk

    def validate_code(self, code):
        """Run validation agent on code"""
        result = self.validator.validate(code)
        self.memory['latest_validation'] = result
        step = {
            "agent": "Validator",
            "agent_name": self.validator.name,
            "status": "completed",
            "output": result,
            "timestamp": datetime.now().isoformat()
        }
        self.workflow_history.append(step)
        return result

    def test_code(self, code):
        """Run testing agent on code"""
        result = self.tester.test(code)
        self.memory['latest_test'] = result
        step = {
            "agent": "Testing",
            "agent_name": self.tester.name,
            "status": "completed",
            "output": result,
            "timestamp": datetime.now().isoformat()
        }
        self.workflow_history.append(step)
        return result

    def secure_code(self, code):
        """Run security agent on code"""
        result = self.security.scan(code)
        self.memory['latest_security'] = result
        step = {
            "agent": "Security",
            "agent_name": self.security.name,
            "status": "completed",
            "output": result,
            "timestamp": datetime.now().isoformat()
        }
        self.workflow_history.append(step)
        return result

    def run_workflow(self, prompt):
        """
        Execute the full LangGraph workflow with auto-fixing
        
        Flow:
        1. Code Generator -> Creates initial code
        2. Validator -> Checks style/syntax, fixes issues
        3. Tester -> Adds error handling, improves testability
        4. Security -> Scans vulnerabilities, applies security fixes
        
        Each agent can modify the code, and fixes are chained together.
        """
        self.workflow_history = []
        self.all_fixes = []
        
        # Step 1: Code Generation
        print(f"🚀 Running Code Generator Agent...")
        code = self.code_generator.generate(prompt)
        original_code = code
        self.memory['original_code'] = code
        self.memory['latest_code'] = code
        self.memory['latest_prompt'] = prompt
        
        self.workflow_history.append({
            "agent": "Code Generator",
            "agent_name": self.code_generator.name,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "description": f"Generated {len(code.split(chr(10)))} lines of code"
        })
        
        # Step 2: Validation - analyze and potentially fix
        print(f"✅ Running Validation Agent...")
        validation = self.validate_code(code)
        if validation.get('fixed_code') and validation.get('fixes_applied'):
            code = validation['fixed_code']
            self.memory['latest_code'] = code
            self.all_fixes.append({
                "agent": "Validator",
                "agent_name": self.validator.name,
                "icon": "✅",
                "fixes": validation['fixes_applied']
            })
            print(f"  → Applied {len(validation['fixes_applied'])} fixes")
        
        # Step 3: Testing - analyze and potentially fix
        print(f"🧪 Running Testing Agent...")
        tests = self.test_code(code)
        if tests.get('fixed_code') and tests.get('fixes_applied'):
            code = tests['fixed_code']
            self.memory['latest_code'] = code
            self.all_fixes.append({
                "agent": "Testing",
                "agent_name": self.tester.name,
                "icon": "🧪",
                "fixes": tests['fixes_applied']
            })
            print(f"  → Applied {len(tests['fixes_applied'])} fixes")
        
        # Step 4: Security - analyze and potentially fix
        print(f"🛡️ Running Security Agent...")
        security = self.secure_code(code)
        if security.get('fixed_code') and security.get('fixes_applied'):
            code = security['fixed_code']
            self.memory['latest_code'] = code
            self.all_fixes.append({
                "agent": "Security",
                "agent_name": self.security.name,
                "icon": "🛡️",
                "fixes": security['fixes_applied']
            })
            print(f"  → Applied {len(security['fixes_applied'])} fixes")
        
        # Determine final code (with all fixes applied)
        final_code = code
        code_was_fixed = final_code != original_code
        total_fixes = sum(len(f['fixes']) for f in self.all_fixes)
        
        print(f"✨ Workflow complete! {total_fixes} total fixes applied.")
        
        return {
            "code": final_code,
            "original_code": original_code if code_was_fixed else None,
            "prompt": prompt,
            "validation": validation,
            "tests": tests,
            "security": security,
            "workflow": self.workflow_history,
            "workflow_graph": self.get_workflow_info(),
            "all_fixes": self.all_fixes,
            "code_was_fixed": code_was_fixed,
            "total_fixes": total_fixes
        }

    def run_agents_on_code(self, code):
        """Run validation, testing, security on existing code (without generation)"""
        self.workflow_history = []
        self.all_fixes = []
        original_code = code
        
        # Run all analysis agents
        validation = self.validate_code(code)
        if validation.get('fixed_code') and validation.get('fixes_applied'):
            code = validation['fixed_code']
            self.all_fixes.append({
                "agent": "Validator",
                "fixes": validation['fixes_applied']
            })
        
        tests = self.test_code(code)
        if tests.get('fixed_code') and tests.get('fixes_applied'):
            code = tests['fixed_code']
            self.all_fixes.append({
                "agent": "Testing",
                "fixes": tests['fixes_applied']
            })
        
        security = self.secure_code(code)
        if security.get('fixed_code') and security.get('fixes_applied'):
            code = security['fixed_code']
            self.all_fixes.append({
                "agent": "Security",
                "fixes": security['fixes_applied']
            })
        
        code_was_fixed = code != original_code
        
        return {
            "validation": validation,
            "tests": tests,
            "security": security,
            "workflow": self.workflow_history,
            "code": code,
            "original_code": original_code if code_was_fixed else None,
            "all_fixes": self.all_fixes,
            "code_was_fixed": code_was_fixed,
            "total_fixes": sum(len(f['fixes']) for f in self.all_fixes)
        }

    def get_agent_info(self):
        """Get information about all agents"""
        return {
            "agents": [
                {
                    "id": "code_generator",
                    "name": self.code_generator.name,
                    "description": self.code_generator.description,
                    "icon": "🚀"
                },
                {
                    "id": "validator",
                    "name": self.validator.name,
                    "description": self.validator.description,
                    "icon": "✅"
                },
                {
                    "id": "tester",
                    "name": self.tester.name,
                    "description": self.tester.description,
                    "icon": "🧪"
                },
                {
                    "id": "security",
                    "name": self.security.name,
                    "description": self.security.description,
                    "icon": "🛡️"
                }
            ],
            "workflow": self.get_workflow_info()
        }
