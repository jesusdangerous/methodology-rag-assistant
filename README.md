# Minimal Go LLM Backend (via Python Inference API)

This project provides a minimal, production-ready Go foundation for LLM integration via interface-driven design.

## Architecture

- `llm`: provider-agnostic interface and HTTP inference implementation
- `api`: CLI adapter for user input/output
- `config`: environment-based configuration

`main.go` wires dependencies using dependency injection.

## Environment variables

- `INFERENCE_BASE_URL` (required): Base URL of Python inference server (example: `http://127.0.0.1:8000`)
- `INFERENCE_TIMEOUT_SECONDS` (optional, default `600`): timeout for Go -> inference `/generate` calls
- `APP_HTTP_ADDR` (optional, default `:8090`)

The app auto-loads variables from `.env` (if the file exists).

Model serving is delegated to `inference_server` (Python FastAPI service), so Go backend is not coupled to model/provider details.

## Stages

### Stage 1 (now): no Docker for model

- Go backend runs locally
- Python inference server runs locally (`inference_server`)
- Docker image should not contain model/data/weights

### Stage 2: Docker only for Go API

- Go API runs in Docker
- Inference server remains local or separate service

## Run

```

Request example:

```bash
curl -X POST http://localhost:8090/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Explain Scrum roles\"}"
```

If this fails, start `inference_server` first, then run this backend.

If first `POST /generate` times out, increase `INFERENCE_TIMEOUT_SECONDS` (initial model download/load can take several minutes).

## Docker Compose (Go + Inference in one command)

Prepare env file for inference service:

Start both services:

```bash
docker compose up --build
```

Services:

- Go API: `http://localhost:8090`
- Inference server: `http://localhost:8000`