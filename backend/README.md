# AgenticRAG Backend

This is the backend for the AgenticRAG system, which provides a FastAPI service with agentic RAG capabilities.

## Architecture

The backend consists of the following components:

- **FastAPI Service**: Provides the REST API endpoints for chat, feedback, and health checks.
- **Agent System**: Routes user messages to specialized agents (application, decision, history).
- **Retrieval System**: Extracts keywords and performs vector search to find relevant information.
- **Memory System**: Stores conversation history in Redis.
- **Azure Functions**: Handles document ingestion and processing.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Azure OpenAI API key or OpenAI API key
- Azure Cognitive Search instance (optional)
- Redis (included in Docker Compose)

### Environment Variables

Copy the `.env.example` file to `.env` and fill in the required values:

```bash
cp .env.example .env
```

The most important variables to set are:

- `USE_AZURE_OPENAI`: Set to `true` to use Azure OpenAI API, or `false` to use OpenAI API.
- `OPENAI_API_KEY`: Your OpenAI API key (if using OpenAI API).
- `AZURE_OPENAI_KEY`: Your Azure OpenAI API key (if using Azure OpenAI API).
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint (if using Azure OpenAI API).
- `AZURE_OPENAI_CHAT_DEPLOYMENT`: Your Azure OpenAI chat model deployment name (if using Azure OpenAI API).
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`: Your Azure OpenAI embedding model deployment name (if using Azure OpenAI API).

### Running with Docker Compose

To run the entire system with Docker Compose:

```bash
cd backend/docker
docker-compose up -d
```

This will start the following services:

- API service on port 8000
- Celery worker
- Redis on port 6379
- Azure Functions on port 7071

### Running Locally

To run the API service locally:

```bash
cd backend/app
pip install -r requirements.txt
uvicorn main:app --reload
```

To run the Celery worker locally:

```bash
cd backend/app
celery -A tasks.async_tasks.celery_app worker --loglevel=info
```

To run the Azure Functions locally:

```bash
cd backend/functions
pip install -r requirements.txt
func start
```

## API Endpoints

- `POST /api/v1/chat`: Send a message to the chat system.
- `POST /api/v1/chat/stream`: Send a message to the chat system and receive a streaming response.
- `POST /api/v1/feedback`: Submit feedback for a message.
- `GET /api/v1/healthz`: Health check endpoint.
- `POST /api/v1/ingest`: Manually trigger document ingestion.

## Development

### Adding a New Agent

To add a new agent:

1. Create a new file in `app/agents/` that extends `BaseAgent`.
2. Implement the `handle` method.
3. Update the `RouterAgent` to route messages to the new agent.

### Adding a New Endpoint

To add a new endpoint:

1. Create a new file in `app/api/v1/` or add to an existing file.
2. Define the endpoint using FastAPI decorators.
3. Update the `main.py` file to include the new router if needed.

## Deployment

The system can be deployed to Azure using Azure Container Apps, Azure Container Instances, or Azure Kubernetes Service.

### Azure Container Apps

```bash
az containerapp up --name agentic-rag-api --resource-group my-resource-group --location eastus --environment my-environment --image agentic-rag-api:latest --target-port 8000 --ingress external --env-vars "USE_AZURE_OPENAI=true" "AZURE_OPENAI_KEY=your-key" "AZURE_OPENAI_ENDPOINT=your-endpoint"
```

### Azure Container Instances

```bash
az container create --name agentic-rag-api --resource-group my-resource-group --image agentic-rag-api:latest --ports 8000 --environment-variables "USE_AZURE_OPENAI=true" "AZURE_OPENAI_KEY=your-key" "AZURE_OPENAI_ENDPOINT=your-endpoint"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
