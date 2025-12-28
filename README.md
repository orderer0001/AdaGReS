# AdaGReS API Guide

## Overview

This repository provides a Flask-based REST API (`AdaGReS_api.py`) for invoking AdaGReS (Adaptive Greedy Context Selection via Redundancy-Aware Scoring for Token-Budgeted RAG) to select the most suitable passages from **candidate chunks**.

## What you will / will not get (please read first)

- **This repository provides**
  - **API server code**: `AdaGReS_api.py`
  - **AdaGReS algorithm & candidate retrieval logic**: `AdaGReS.py` (internally retrieves top-N candidates from a vector database)
  - **Request examples**: `test_api.py` / `curl` examples

- **This repository does NOT provide (you must set up/prepare these yourself)**
  - **Vector database deployment & operations** (e.g., Milvus installation, startup, monitoring, backups)
  - **Collection creation / indexing / data ingestion** (your collection schema, fields, and vector dimension must be prepared by you)
  - **A ready-to-use embedding service / model path** (the current `my_milvus.py` contains hard-coded paths tied to the author's environment and must be adapted)

> In other words: this repo provides **“request-serving + algorithm interface code”**, not a one-click full-stack solution including the vector database.

## Requirements

- Python 3.7+
- Dependencies: `requirements.txt`
- **You must prepare a vector database first** (Milvus 2.x recommended; see the [Milvus official docs](https://milvus.io/docs))

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Set up and initialize your vector database

Make sure that:

- **Milvus is reachable** (URI/port/authentication are up to you)
- A collection has been created (e.g., `entities`) and includes at least these fields:
  - **Vector field**: e.g., `vector` (FloatVector; its dimension must match your embedding output)
  - **Text field**: e.g., `description` (returned to the caller)
- Your data has been ingested and vector search returns results

> This repository will not create collections/indexes or ingest data for you. If these are not ready, the API will fail at runtime (usually HTTP 500).

### 3) Update local configuration

There are hard-coded values in the code; you must modify them for your environment:

- **Milvus connection settings**
  - `AdaGReS.py`: `MilvusClient(uri=..., db_name=...)`
  - `my_milvus.py`: `MilvusClient(uri=..., db_name=...)`

- **Collection / field names / filters**
  - In `AdaGReS.py` → `get_top_n_candidates()`:
    - `collection_name` (default: `entities`)
    - `field_name` (default: `vector`)
    - `output_fields` (default: `["description"]`)
    - `filter="knowledge_base_id in [...]"` (example/business-specific filter from the author; remove or replace it if your schema does not have this field)

- **Embedding generation logic (vector dimension must match your Milvus collection)**
  - `my_milvus.py`: `generate_vector_from_text()` depends on a local embedding model and import paths
  - Replace the model path mapping/import paths, or the entire generation logic, with your own embedding implementation

### 4) Start the service

```bash
python AdaGReS_api.py
```

By default it listens on `http://localhost:5000` (the code uses `0.0.0.0:5000`).

## API Endpoints

### 1) Select chunks (main endpoint)

- **Endpoint**: `POST /api/select_chunks`
- **Content-Type**: `application/json`

**Request parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---:|---:|---|
| query_text | string | Yes | - | Query text |
| top_n_candidates | int | No | 3000 | Number of candidates retrieved from the vector DB (top-N) |
| selection_method | string | No | `"greedy"` | `"greedy"` or `"simple"` |
| k | int | No | - | Target number of chunks to return; required when `selection_method="simple"` |
| beta | number | No | - | Fixed beta; if provided, beta will no longer be computed dynamically |
| Tmax | int | No | 1500 | Used to estimate beta dynamically when beta is not provided (max tokens/budget) |
| sample_n | int | No | 500 | Sample size used during dynamic estimation |

**Example (greedy + dynamic beta):**

```bash
curl -X POST http://localhost:5000/api/select_chunks \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What are the adverse reactions of metronidazole?",
    "top_n_candidates": 1000,
    "selection_method": "greedy",
    "Tmax": 1500,
    "sample_n": 500
  }'
```

**Example (simple):**

```bash
curl -X POST http://localhost:5000/api/select_chunks \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What are the adverse reactions of metronidazole?",
    "top_n_candidates": 1000,
    "k": 15,
    "selection_method": "simple"
  }'
```

**Response example:**

```json
{
  "success": true,
  "message": "Chunks selected successfully",
  "code": 200,
  "data": {
    "selected_chunks": [],
    "beta": 0.123,
    "k": 18,
    "total_selected": 18,
    "beta_source": "dynamic"
  }
}
```

### 2) Health check

- **Endpoint**: `GET /api/health`

### 3) API help

- **Endpoint**: `GET /api/select_chunks/help`

## Testing

```bash
python test_api.py
```

Note: `test_api.py` contains `input("按Enter键开始测试...")`. If you want to run it in an automated environment, remove this interactive prompt.

## FAQ

### 1) HTTP 500 / Internal Server Error

This is usually caused by one of the following:

- **Milvus is not running / unreachable**: wrong `uri`/port/authentication
- **Collection/schema mismatch**: the code queries `entities` with fields `vector/description`, but your DB does not have them
- **Vector dimension mismatch**: embedding output dimension ≠ Milvus vector field dimension
- **Filter field does not exist**: `AdaGReS.py` has a `knowledge_base_id` filter (it will error immediately if your schema lacks this field)
- **Embedding generation fails**: `my_milvus.py` depends on author-specific model/import paths; replace with your own implementation

### 2) Can I “run it end-to-end with one click” using this repository?

If you do not yet have a vector database + data, and you do not have an embedding pipeline: **no**. You must first prepare “vector DB + data ingestion + embedding generation”, then adjust the connection settings and field names in this repo to match your environment.

## Production deployment suggestions (optional)

1. Tune `top_n_candidates` based on your workload and latency requirements
2. Consider using a connection pool under high concurrency
3. Add a caching layer to improve response latency