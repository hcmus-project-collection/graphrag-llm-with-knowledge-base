# 🧠 Large Language Model with Knowledge Base

An advanced RAG (Retrieval-Augmented Generation) system that combines Large Language Models with a sophisticated knowledge base featuring vector similarity search and knowledge graph extraction.

## 📋 Project Overview

This project implements a comprehensive RAG system with the following key features:

- **🕸️ Knowledge Graph Extraction**: Automatically extracts structured relationships (triplets) from documents
- **🔍 Vector Similarity Search**: Uses Milvus vector database for efficient semantic search
- **📄 Multi-modal Document Processing**: Supports PDF files and text input with advanced chunking
- **⚡ Real-time Query Processing**: Fast retrieval with Redis caching and background processing
- **💻 Modern Web Interface**: React-based frontend for document upload and querying

## 🏗️ Architecture

### 🔧 Core Components

#### 🐍 Backend (`app/`)
- **`server.py`**: FastAPI application server with CORS support and health monitoring
- **`api.py`**: REST API endpoints for document insertion, querying, and knowledge base management
- **`handlers.py`**: Core business logic including document processing, embedding generation, and query execution
- **`graph_handlers.py`**: Knowledge graph extraction using LLM-powered triplet extraction
- **`models.py`**: Pydantic data models and API schemas
- **`embedding.py`**: Text embedding model configuration and management
- **`constants.py`**: System configuration and environment variable handling
- **`utils.py`**: Utility functions for async processing, batching, and file operations
- **`state.py`**: Application state management
- **`io.py`**: File I/O operations and external service integrations

#### 🔄 Wrappers (`app/wrappers/`)
- **`milvus_kit.py`**: Milvus vector database connection management
- **`redis_kit.py`**: Redis caching layer with decorators
- **`log_decorators.py`**: Logging and performance monitoring decorators

#### ⚛️ Frontend (`frontend/`)
- **React TypeScript Application**: Modern web interface
- **Components**:
  - `UploadForm.tsx`: Text document upload
  - `PDFUploader.tsx`: PDF file upload with Cloudinary integration
  - `QueryForm.tsx`: Search interface
  - `Output.tsx`: Results display

#### 🏢 Infrastructure
- **🗄️ Milvus Vector Database**: High-performance vector similarity search
- **⚡ Redis**: Caching layer for query results and performance optimization
- **📖 Docling Server**: PDF processing and intelligent chunking
- **🪣 MinIO**: Object storage for Milvus
- **🔑 etcd**: Metadata storage for Milvus cluster

## 🛠️ Tech Stack

### 🐍 Backend
- **⚡ FastAPI** (v0.115.5): High-performance async web framework
- **🗄️ Milvus** (v2.5.0): Vector database for similarity search
- **🔴 Redis**: In-memory caching and session storage
- **🐍 PyMilvus** (v2.5.3): Python client for Milvus
- **✅ Pydantic** (v2.9.2): Data validation and serialization
- **📄 Docling**: Advanced PDF processing and chunking
- **⏰ Schedule**: Background task scheduling
- **🌐 HTTPX**: Async HTTP client for external API calls

### ⚛️ Frontend
- **⚛️ React** (v19.0.0): Modern UI framework
- **📘 TypeScript** (v4.9.5): Type-safe JavaScript
- **🔗 Axios** (v1.8.2): HTTP client for API communication

### 🐳 Infrastructure
- **🐳 Docker & Docker Compose**: Containerized deployment
- **🦄 Uvicorn**: ASGI server for FastAPI
- **🪣 MinIO**: S3-compatible object storage
- **🔑 etcd**: Distributed key-value store

## 🚀 API Endpoints

### 📄 Document Management
- **📤 POST `/api/insert`**: Insert documents (text or file URLs) into knowledge base
- **🔄 POST `/api/update`**: Update existing documents in knowledge base
- **🗑️ DELETE `/api/delete`**: Delete all documents from a knowledge base

### 🔍 Query Operations
- **🔎 POST `/api/query`**: Semantic search with configurable parameters
- **📋 GET `/api/sample`**: Get sample documents from knowledge base

### 🛠️ Utilities
- **📥 GET `/api/export`**: Export collection data (admin endpoint)
- **❤️ GET `/`**: Health check endpoint

### ⚙️ Query Parameters
- `top_k`: Number of similar documents to retrieve
- `threshold`: Similarity threshold for filtering results
- `kb`: Knowledge base identifier(s)

## 🕸️ Knowledge Graph Features

The system implements sophisticated knowledge graph extraction:

### 🔗 Triplet Extraction
- **🔍 Automatic Relationship Discovery**: Extracts (subject, relation, object) triplets from text
- **🤖 LLM-Powered Analysis**: Uses OpenAI-compatible APIs for intelligent relationship identification
- **🗄️ Structured Storage**: Stores entities and relations separately with dedicated suffixes
- **📈 Query Enhancement**: Improves search relevance through entity and relationship context

### 🏷️ Named Entity Recognition
- **👤 Entity Extraction**: Identifies persons, locations, organizations, and key concepts
- **✨ Query Refinement**: Enhances search queries with extracted entities
- **🎯 Contextual Analysis**: Preserves meaningful phrases and multi-word expressions

## ⚙️ Configuration

### 🔐 Required Environment Variables

#### 🐍 Backend Configuration (`.env`)
```bash
# 🤖 Embedding Model Configuration
EMBEDDING_URL=your-embedding-service-url
EMBEDDING_MODEL_ID=your-embedding-model-id
TOKENIZER_MODEL_ID=your-tokenizer-model-id
MODEL_DIMENSION=4096

# 🧠 LLM Configuration (for Knowledge Graph)
OPENAI_API_KEY=your-llm-api-key
OPENAI_API_BASE=your-llm-api-base-url
OPENAI_MODEL_ID=your-model-name

# 🗄️ Database Configuration
MILVUS_HOST=http://localhost:19530

# ⚡ Optional Performance Tuning
DEFAULT_BATCH_SIZE=8
DEFAULT_TOP_K=1
DEFAULT_CONCURRENT_EMBEDDING_REQUESTS_LIMIT=64
DEFAULT_INSERT_BATCH_SIZE=128
```

#### ⚛️ Frontend Configuration (`frontend/.env`)
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
