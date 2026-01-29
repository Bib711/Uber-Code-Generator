# Entry point for Uber Code Generator API with Streaming - FastAPI Version
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List
import json

from config import settings
from database import connect_to_mongo, close_mongo_connection, SessionDB, MessageDB
from auth import router as auth_router, get_current_user, get_optional_user
from orchestrator import Orchestrator


# Lifespan for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    yield
    # Shutdown
    await close_mongo_connection()


# FastAPI app
app = FastAPI(
    title="Uber Code Generator API",
    description="AI-powered code generation with multi-agent validation",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth router
app.include_router(auth_router)


# Pydantic Schemas (Request/Response models with automatic validation)
class GenerateRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    api_key: Optional[str] = None


class EditRequest(BaseModel):
    original_code: str
    updates: List[dict] = []


class RegenerateRequest(BaseModel):
    original_prompt: str = ""
    edit_instructions: str
    current_code: str = ""
    api_key: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    created_at: str


class HealthResponse(BaseModel):
    status: str
    message: str


# API Routes
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Uber Code Generator API (FastAPI + MongoDB)"}


@app.post("/api/generate")
async def generate_code(request: GenerateRequest, user: Optional[dict] = Depends(get_optional_user)):
    """Generate code from a prompt (non-streaming)"""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Use API key from request or from settings
    api_key = request.api_key or settings.GROQ_API_KEY
    orchestrator = Orchestrator(api_key)
    result = orchestrator.run_workflow(request.prompt)
    
    # Save to session if provided
    if request.session_id:
        try:
            await MessageDB.add_message(
                session_id=request.session_id,
                role='user',
                content=request.prompt
            )
            await MessageDB.add_message(
                session_id=request.session_id,
                role='assistant',
                content="Code generated",
                code_output=result.get('code'),
                workflow_data=result.get('workflow')
            )
        except Exception as e:
            print(f"Session save error: {e}")
    
    return result


@app.post("/api/generate/stream")
async def generate_code_stream(request: GenerateRequest):
    """Streaming endpoint for real-time code generation with agent fixes"""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    orchestrator = Orchestrator(request.api_key)
    
    async def generate():
        full_code = ""
        current_code = ""
        original_code = ""
        all_fixes = []
        
        # Stream code generation
        yield "data: " + json.dumps({'type': 'start', 'agent': 'Code Generator', 'message': '🚀 Generating code...'}) + "\n\n"
        
        for chunk in orchestrator.generate_code_stream(request.prompt):
            full_code += chunk
            yield "data: " + json.dumps({'type': 'chunk', 'content': chunk}) + "\n\n"
        
        original_code = full_code
        current_code = full_code
        lines_count = len(full_code.splitlines())
        yield "data: " + json.dumps({'type': 'agent_complete', 'agent': 'Code Generator', 'message': f'Generated {lines_count} lines'}) + "\n\n"
        
        # Validator Agent - analyzes and fixes
        yield "data: " + json.dumps({'type': 'start', 'agent': 'Validator', 'message': '✅ Analyzing code quality...'}) + "\n\n"
        validation = orchestrator.validate_code(current_code)
        
        if validation.get('fixed_code') and validation.get('fixes_applied'):
            current_code = validation['fixed_code']
            fix_count = len(validation['fixes_applied'])
            all_fixes.append({'agent': 'Validator', 'fixes': validation['fixes_applied']})
            yield "data: " + json.dumps({'type': 'code_update', 'agent': 'Validator', 'code': current_code, 'fixes': validation['fixes_applied'], 'message': f'Applied {fix_count} fixes'}) + "\n\n"
        
        yield "data: " + json.dumps({'type': 'result', 'agent': 'Validator', 'data': validation}) + "\n\n"
        
        # Testing Agent - analyzes and fixes
        yield "data: " + json.dumps({'type': 'start', 'agent': 'Testing', 'message': '🧪 Checking testability & error handling...'}) + "\n\n"
        tests = orchestrator.test_code(current_code)
        
        if tests.get('fixed_code') and tests.get('fixes_applied'):
            current_code = tests['fixed_code']
            fix_count = len(tests['fixes_applied'])
            all_fixes.append({'agent': 'Testing', 'fixes': tests['fixes_applied']})
            yield "data: " + json.dumps({'type': 'code_update', 'agent': 'Testing', 'code': current_code, 'fixes': tests['fixes_applied'], 'message': f'Applied {fix_count} fixes'}) + "\n\n"
        
        yield "data: " + json.dumps({'type': 'result', 'agent': 'Testing', 'data': tests}) + "\n\n"
        
        # Security Agent - analyzes and fixes
        yield "data: " + json.dumps({'type': 'start', 'agent': 'Security', 'message': '🛡️ Scanning for vulnerabilities...'}) + "\n\n"
        security = orchestrator.secure_code(current_code)
        
        if security.get('fixed_code') and security.get('fixes_applied'):
            current_code = security['fixed_code']
            fix_count = len(security['fixes_applied'])
            all_fixes.append({'agent': 'Security', 'fixes': security['fixes_applied']})
            yield "data: " + json.dumps({'type': 'code_update', 'agent': 'Security', 'code': current_code, 'fixes': security['fixes_applied'], 'message': f'Applied {fix_count} security fixes'}) + "\n\n"
        
        yield "data: " + json.dumps({'type': 'result', 'agent': 'Security', 'data': security}) + "\n\n"
        
        # Calculate totals
        total_fixes = sum(len(f['fixes']) for f in all_fixes)
        code_was_fixed = current_code != original_code
        
        # Final result
        final_data = {
            'type': 'complete',
            'code': current_code,
            'original_code': original_code if code_was_fixed else None,
            'prompt': request.prompt,
            'all_fixes': all_fixes,
            'total_fixes': total_fixes,
            'code_was_fixed': code_was_fixed
        }
        yield "data: " + json.dumps(final_data) + "\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/edit")
async def edit_code(request: EditRequest):
    """Edit existing code and re-run agents"""
    if not request.original_code:
        raise HTTPException(status_code=400, detail="Original code is required")
    
    updated_code = request.original_code
    for update in request.updates:
        old = update.get('old', '')
        new = update.get('new', '')
        if old:
            updated_code = updated_code.replace(old, new)
    
    orchestrator = Orchestrator()
    result = orchestrator.run_agents_on_code(updated_code)
    
    return {"updated_code": updated_code, **result}


@app.post("/api/regenerate")
async def regenerate_with_edit(request: RegenerateRequest):
    """Regenerate code with edited prompt"""
    if not request.edit_instructions:
        raise HTTPException(status_code=400, detail="Edit instructions required")
    
    new_prompt = f"""Original request: {request.original_prompt}

Current code:
```
{request.current_code}
```

Please modify the code based on these instructions: {request.edit_instructions}

Only make the requested changes, keep the rest of the code intact."""
    
    orchestrator = Orchestrator(request.api_key)
    result = orchestrator.run_workflow(new_prompt)
    
    return result


@app.post("/api/regenerate/stream")
async def regenerate_stream(request: RegenerateRequest):
    """Streaming regeneration with edited prompt"""
    new_prompt = f"""Original request: {request.original_prompt}

Current code:
```
{request.current_code}
```

Modify based on: {request.edit_instructions}
Only make requested changes."""
    
    orchestrator = Orchestrator(request.api_key)
    
    async def generate():
        full_code = ""
        yield f"data: {json.dumps({'type': 'start', 'agent': 'Code Generator'})}\n\n"
        
        for chunk in orchestrator.generate_code_stream(new_prompt):
            full_code += chunk
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
        
        yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Code Generator'})}\n\n"
        
        yield f"data: {json.dumps({'type': 'start', 'agent': 'Validator'})}\n\n"
        validation = orchestrator.validate_code(full_code)
        yield f"data: {json.dumps({'type': 'result', 'agent': 'Validator', 'data': validation})}\n\n"
        
        yield f"data: {json.dumps({'type': 'start', 'agent': 'Testing'})}\n\n"
        tests = orchestrator.test_code(full_code)
        yield f"data: {json.dumps({'type': 'result', 'agent': 'Testing', 'data': tests})}\n\n"
        
        yield f"data: {json.dumps({'type': 'start', 'agent': 'Security'})}\n\n"
        security = orchestrator.secure_code(full_code)
        yield f"data: {json.dumps({'type': 'result', 'agent': 'Security', 'data': security})}\n\n"
        
        yield f"data: {json.dumps({'type': 'complete', 'code': full_code, 'prompt': request.edit_instructions})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# ============== Session Management (MongoDB) ==============

@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(user: dict = Depends(get_current_user)):
    """Create a new chat session (requires authentication)"""
    session = await SessionDB.create_session(user_id=user["_id"])
    return {"session_id": session["_id"], "created_at": session["created_at"].isoformat()}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str, user: dict = Depends(get_current_user)):
    """Get a specific chat session with messages"""
    session = await SessionDB.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check ownership
    if session["user_id"] != user["_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    messages = await MessageDB.get_session_messages(session_id)
    
    return {
        "session_id": session["_id"],
        "title": session.get("title", "New Chat"),
        "created_at": session["created_at"].isoformat(),
        "messages": [{
            "id": msg["_id"],
            "role": msg["role"],
            "content": msg["content"],
            "code_output": msg.get("code_output"),
            "workflow_data": msg.get("workflow_data"),
            "created_at": msg["created_at"].isoformat()
        } for msg in messages]
    }


@app.get("/api/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    """List user's chat sessions"""
    sessions = await SessionDB.get_user_sessions(user["_id"])
    return [{
        "session_id": s["_id"],
        "title": s.get("title", "New Chat"),
        "created_at": s["created_at"].isoformat(),
        "updated_at": s["updated_at"].isoformat(),
        "message_count": s.get("message_count", 0)
    } for s in sessions]


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    """Delete a chat session"""
    session = await SessionDB.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session["user_id"] != user["_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    await SessionDB.delete_session(session_id)
    return {"message": "Session deleted"}


class AddMessageRequest(BaseModel):
    role: str
    content: str
    code_output: Optional[str] = None
    workflow_data: Optional[dict] = None


@app.post("/api/sessions/{session_id}/messages")
async def add_message_to_session(
    session_id: str, 
    request: AddMessageRequest,
    user: dict = Depends(get_current_user)
):
    """Add a message to a session and auto-generate title if first user message"""
    session = await SessionDB.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session["user_id"] != user["_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Add the message
    message = await MessageDB.add_message(
        session_id=session_id,
        role=request.role,
        content=request.content,
        code_output=request.code_output,
        workflow_data=request.workflow_data
    )
    
    # Auto-generate title on first user message if title is still "New Chat"
    if request.role == "user" and session.get("title") == "New Chat":
        # Get message count
        messages = await MessageDB.get_session_messages(session_id)
        user_messages = [m for m in messages if m["role"] == "user"]
        
        if len(user_messages) == 1:  # First user message
            title = await SessionDB.generate_title_from_chat(request.content)
            await SessionDB.update_session_title(session_id, title)
            return {
                "message_id": message["_id"],
                "new_title": title
            }
    
    return {"message_id": message["_id"]}


@app.patch("/api/sessions/{session_id}/title")
async def update_session_title_endpoint(
    session_id: str,
    title: str,
    user: dict = Depends(get_current_user)
):
    """Manually update session title"""
    session = await SessionDB.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session["user_id"] != user["_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    await SessionDB.update_session_title(session_id, title)
    return {"message": "Title updated", "title": title}


class BulkDeleteRequest(BaseModel):
    session_ids: List[str]


@app.post("/api/sessions/bulk-delete")
async def bulk_delete_sessions(
    request: BulkDeleteRequest,
    user: dict = Depends(get_current_user)
):
    """Delete multiple sessions at once"""
    deleted_count = 0
    
    for session_id in request.session_ids:
        try:
            session = await SessionDB.get_session(session_id)
            if session and session["user_id"] == user["_id"]:
                await SessionDB.delete_session(session_id)
                deleted_count += 1
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            continue
    
    return {"message": f"Deleted {deleted_count} sessions", "deleted_count": deleted_count}


@app.post("/api/sessions/cleanup-empty")
async def cleanup_empty_sessions(user: dict = Depends(get_current_user)):
    """Delete all sessions with 0 messages"""
    sessions = await SessionDB.get_user_sessions(user["_id"], limit=100)
    deleted_count = 0
    
    for session in sessions:
        if session.get("message_count", 0) == 0:
            try:
                await SessionDB.delete_session(session["_id"])
                deleted_count += 1
            except Exception as e:
                print(f"Error cleaning up session {session['_id']}: {e}")
    
    return {"message": f"Cleaned up {deleted_count} empty sessions", "deleted_count": deleted_count}


# Run with: python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=True)
