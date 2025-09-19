Taru Chatbot Flask API

A lightweight Flask service exposing Taru chatbot and burnout risk endpoints for integration with the Next.js app.

Endpoints

- GET /health — health check
- POST /chat — body: { "message": string } → { answer, actions, sources, confidence }
- GET /risk — → { risk_probability, reason }

Quickstart

1. Python 3.10+ recommended.
2. Create and activate a virtualenv.
3. Install deps:
   pip install -r flask_api/requirements.txt
4. Create .env in flask_api/ using .env.example and set GOOGLE_API_KEY.
5. Run locally:
   python -m flask_api.app
   Server runs on http://localhost:8000 by default.

CORS

Set CORS_ORIGINS in .env (comma-separated) to your Next.js origin(s), e.g. http://localhost:3000.

Example curl

curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"I feel stressed about exams"}'


