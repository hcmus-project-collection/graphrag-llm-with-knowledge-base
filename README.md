# Large Language Model with Knowledge Base

This is a project of large language model associated with knowledge base, or in some sense, this is
an RAG (Retrieval-Augmented Generation) application.

## Project Overview

This project implements a RAG (Retrieval-Augmented Generation) system that combines a Large Language Model with a knowledge base. The system uses Milvus as a vector database for efficient similarity search and includes a knowledge graph component for enhanced information retrieval.

## Architecture

### Backend Components
- **FastAPI Server** (`server.py`): Main application server handling API requests
- **Core Components** (`app/`):
  - `api.py`: API route definitions
  - `handlers.py`: Core business logic and data processing
  - `graph_handlers.py`: Knowledge graph operations
  - `models.py`: Data models and schemas
  - `embedding.py`: Text embedding functionality
  - `utils.py`: Utility functions
  - `constants.py`: System constants and configurations

### Frontend Components
- **React Application** (`frontend/`):
  - Modern web interface for interacting with the RAG system
  - Built with React and TypeScript

### Infrastructure
- **Milvus Vector Database**: For efficient vector similarity search
- **Docker Containers**: For easy deployment and scaling
  - `docker-compose.yml`: Main service orchestration
  - `milvus-docker-compose.yml`: Milvus database setup
  - `models-docker-compose.yml`: ML models containerization

## Tech Stack
- Backend:
  - FastAPI
  - Milvus vector database
  - Knowledge Graph
  - Python 3.10+
- Frontend:
  - ReactJS
  - TypeScript
- Infrastructure:
  - Docker
  - Docker Compose

## Getting Started

### System Requirements
- Python 3.10 or later
- Node.js (for frontend)
- Docker and Docker Compose
- Virtual environment (conda or uv recommended)

### Installation Steps

1. **Set up Python Environment**
```bash
conda create -n llmkb python==3.12 -y
conda activate llmkb
```

2. **Configure Environment Variables**
```bash
cp .env.template .env
# Edit .env with your configuration
```

3. **Start Milvus Database**
```bash
docker-compose -f milvus-docker-compose.yml up -d
```

4. **Install Backend Dependencies**
```bash
pip install -r requirements.txt
```

5. **Start Backend Server**
```bash
python -O server.py
```

6. **Set up Frontend**
```bash
cd frontend
cp .env.template .env
npm install
npm start
```

## System Flow

1. **Data Ingestion**
   - Documents are processed and embedded
   - Vectors are stored in Milvus
   - Knowledge graph relationships are extracted

2. **Query Processing**
   - User queries are received through the API
   - Query is embedded and matched against stored vectors
   - Relevant context is retrieved from Milvus
   - Knowledge graph provides additional context

3. **Response Generation**
   - LLM generates response using retrieved context
   - Response is returned to the frontend

## Development

The project includes several Docker configurations for different components:
- `docker-compose.yml`: Main application services
- `milvus-docker-compose.yml`: Vector database setup
- `models-docker-compose.yml`: ML models deployment

## License

See the [LICENSE](LICENSE) file for details.
