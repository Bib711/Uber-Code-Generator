# Uber Code Generator

A multi-agent AI code generation system with a modern React frontend and FastAPI backend.

## Features

- 🤖 **Multi-Agent System** - Code generation, validation, testing, and security agents
- 💬 **Generative UI** - Real-time streaming with interactive components
- 🔐 **Google OAuth** - Secure authentication
- 📝 **Context-Aware** - Edit and iterate on previously generated code

## Tech Stack

| Frontend | Backend |
|----------|---------|
| React 18 | FastAPI |
| Framer Motion | MongoDB |
| React Router | Groq API  |

## Quick Start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
uvicorn main:app --reload
```

### 2. Frontend

```bash
cd frontend
npm install
npm start
```

## Environment Variables

Create `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key
MONGODB_URL=your_mongodb_connection_string
GOOGLE_CLIENT_ID=your_google_oauth_client_id
JWT_SECRET=your_jwt_secret
```

## Project Structure

```
uber_code_generator/
├── backend/
│   ├── main.py           # FastAPI app
│   ├── orchestrator.py   # Agent orchestration
│   ├── agents/           # AI agents
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── components/   # React components
    │   ├── pages/        # Page components
    │   └── context/      # Auth context
    └── package.json
```

## License

MIT