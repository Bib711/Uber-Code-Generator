#!/usr/bin/env python
"""FastAPI server runner with uvicorn"""
import os
import sys

# Ensure current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == '__main__':
    print("🚀 Starting Uber Code Generator Backend (FastAPI)...")
    print("📚 API Docs available at http://localhost:5000/docs")
    print("📖 ReDoc available at http://localhost:5000/redoc")
    print("🔗 API available at http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,  # Auto-reload on code changes (dev mode)
        workers=1
    )
