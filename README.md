# GraphRAG - Large Language Model with Knowledge Base

This is GraphRAG, a project of large language model associated with knowledge base, or in some sense, this is
an RAG (Retrieval-Augmented Generation) application with graph-based knowledge representation.

## 📋 Project Overview

GraphRAG implements a RAG (Retrieval-Augmented Generation) system that combines a Large Language Model with a knowledge base. The system uses Milvus as a vector database for efficient similarity search and includes a knowledge graph component for enhanced information retrieval.

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
  - Modern web interface for interacting with the GraphRAG system
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
REACT_APP_BACKEND_URL=http://localhost:8000
REACT_APP_CLOUDINARY_PRESET=your-cloudinary-preset
REACT_APP_CLOUDINARY_CLOUD_NAME=your-cloudinary-cloud-name
```

## 🚀 Getting Started

### ✅ Prerequisites
- **🐍 Python 3.12+**: Required for backend
- **📗 Node.js 16+**: Required for frontend
- **🐳 Docker & Docker Compose**: For infrastructure services
- **🌍 Virtual Environment**: Recommended (conda, venv, or uv)

### 📦 Installation

#### 1️⃣ Environment Setup
```bash
# 🐍 Create Python environment
conda create -n llmkb python=3.12 -y
conda activate llmkb

# 📥 Clone and navigate to project
git clone <repository-url>
cd llm-with-knowledge-base
```

#### 2️⃣ Infrastructure Services
```bash
# 🚀 Start Milvus vector database
docker-compose -f milvus-docker-compose.yml up -d

# ✅ Verify services are running
docker-compose -f milvus-docker-compose.yml ps
```

#### 3️⃣ Backend Setup
```bash
# 📦 Install dependencies
pip install -r requirements.txt

# ⚙️ Configure environment
cp .env.template .env
# Edit .env with your configuration

# 🚀 Start backend server
python -O server.py
```

#### 4️⃣ Frontend Setup
```bash
cd frontend

# 📦 Install dependencies
npm install

# ⚙️ Configure environment
cp .env.template .env
# Edit .env with your configuration

# 🚀 Start frontend development server
npm start
```

## 📖 Usage Workflow

### 1️⃣ Document Ingestion
1. **📤 Upload Documents**: Use the web interface to upload PDFs or input text
2. **⚙️ Automatic Processing**: System chunks documents and extracts knowledge graphs
3. **🔢 Vector Generation**: Creates embeddings for semantic search
4. **💾 Storage**: Stores vectors in Milvus and relationships in knowledge graph

### 2️⃣ Knowledge Graph Construction
1. **🔍 Chunk Analysis**: Each document chunk is analyzed for relationships
2. **🔗 Triplet Extraction**: LLM identifies (subject, relation, object) triplets
3. **🏷️ Entity Recognition**: Extracts named entities and key concepts
4. **🗄️ Graph Storage**: Stores entities and relations with dedicated collections

### 3️⃣ Query Processing
1. **📝 Query Analysis**: Input query is processed and refined
2. **🏷️ Entity Extraction**: Identifies relevant entities in the query
3. **🔍 Vector Search**: Performs similarity search in Milvus
4. **🕸️ Graph Enhancement**: Leverages knowledge graph for context
5. **📊 Result Ranking**: Returns ranked results with similarity scores

## 🚀 Advanced Features

### 🔄 Background Processing
- **⚡ Asynchronous Document Processing**: Non-blocking document ingestion
- **🧹 Scheduled Deduplication**: Automatic cleanup of duplicate entries (every 300 minutes)
- **⚡ Concurrent Embedding**: Parallel processing for faster throughput

### 📈 Performance Optimization
- **⚡ Redis Caching**: Query results cached for 60 seconds
- **📦 Batch Processing**: Configurable batch sizes for embeddings and insertions
- **🔗 Connection Pooling**: Reusable database connections
- **🚦 Concurrent Limits**: Configurable concurrency for external API calls

### 📊 Data Management
- **🏠 Knowledge Base Isolation**: Separate collections per knowledge base
- **📥 Export Functionality**: Admin endpoint for data export
- **🧹 Cleanup Utilities**: Scripts for resetting Milvus and Redis

## 👨‍💻 Development

### ✨ Code Quality
- **🔧 Ruff**: Comprehensive linting and formatting (configured in `pyproject.toml`)
- **📝 Type Hints**: Full type annotation coverage
- **✅ Pydantic Validation**: Strong data validation and serialization
- **⚡ Async/Await**: Modern async programming patterns

### 🛠️ Utility Scripts
```bash
# 🔄 Reset Milvus database
./scripts/reset_milvus.sh

# 🔄 Reset Redis cache
./scripts/reset_redis.sh

# 📥 Export collection data (programmatic)
python scripts/export_collection_data.py
```

### 🧪 Testing
```bash
# ⚛️ Frontend tests
cd frontend
npm test

# 🐍 Backend tests (if available)
python -m pytest
```

## 📊 Monitoring and Logging

- **❤️ Health Check Endpoint**: `GET /` for service monitoring
- **📝 Structured Logging**: Configurable log levels with filtered endpoints
- **⏱️ Performance Metrics**: Execution time logging for key operations
- **🚨 Error Handling**: Comprehensive exception handling and logging

## 🐳 Docker Deployment

### 🚀 Full Stack Deployment
```bash
# 🚀 Start all services
docker-compose up -d

# 📝 Monitor logs
docker-compose logs -f
```

### 🔧 Individual Services
```bash
# 🗄️ Milvus only
docker-compose -f milvus-docker-compose.yml up -d

# 📱 Application services
docker-compose up app frontend
```

## 🔧 Troubleshooting

### ⚠️ Common Issues
1. **🗄️ Milvus Connection**: Ensure Milvus is running on correct port (19530)
2. **🤖 Embedding Service**: Verify embedding API endpoint and credentials
3. **🧠 LLM API**: Check OpenAI-compatible API configuration
4. **💾 Memory Usage**: Monitor system resources during large document processing

### 📝 Logs and Debugging
- **🐍 Backend logs**: Check console output or configure file logging
- **⚛️ Frontend logs**: Browser developer console
- **🗄️ Milvus logs**: `docker-compose -f milvus-docker-compose.yml logs milvus-standalone`

## 📄 License

See the [LICENSE](LICENSE) file for details.

## 🏷️ Version

Current version: **v1.0.0** ✨

## About us
We are Van-Loc Nguyen and Dinh-Khoi Vo, Master's students from [University of Science, VNUHCM](https://hcmus.edu.vn/).
This is the project for the subject MTH088 - Advanced Mathematics for AI.

## Acknowledgments
Special thanks for Do Tran Ngoc for his help on debugging.
