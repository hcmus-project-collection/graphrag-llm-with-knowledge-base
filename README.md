# Large Language Model with Knowledge Base

This is a project of large language model associated with knowledge base, or in some sense, this is
an RAG (Retrieval-Augmented Generation) application.

## Tech Stacks:
- FastAPI
- Milvus vector database
- Knowledge Graph
- Frontend: ReactJS

## Running the server
### System Requirements
First of all, you need a Python (version 3.10 or later) installed. It is recommended to use a virtual environment, such as [conda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [uv](https://astral.sh/blog/uv).
You will also need Docker installed on your system.

### Install `conda` environment
Check [this docs](https://www.anaconda.com/docs/getting-started/miniconda/main) for `conda` installation guide.
After `conda` is installed, let's create a virtual environment for this project.

```bash
conda create -n llmkb python==3.12 -y
```
```bash
conda activate llmkb
```

### Setup Docker for Milvus
```bash

# for the latest version, checkout https://milvus.io/docs/install_standalone-docker-compose.md
wget https://github.com/milvus-io/milvus/releases/download/v2.5.0-beta/milvus-standalone-docker-compose.yml -O docker-compose.yml
```
Build the Docker container
```bash
docker-compose -f docker-compose.yml up -d
```
### Install requirement packages
```bash
pip install -r requirements.txt
```
### Run the server
```bash
python -O server.py
```

## Running the frontend
### System Requirements
You need to have Node.js installed on your system. You can download it from [here](https://nodejs.org/en/download/).
### Install dependencies
```bash
cd frontend
npm install
```
### Run the frontend
```bash
npm start
```
