# AdaptKG-RAG

## An Adaptive Knowledge Graph Retrieval Framework for Explainable Multi-Hop Question Answering

## 1. Abstract / Problem Statement

Fixed-depth subgraph retrieval in standard RAG architectures over Knowledge Graphs (KGs) is rigid. It either **under-retrieves** for complex questions (causing missing answers for multi-hop queries) or **over-retrieves** for simple questions (pulling massive amounts of irrelevant neighborhood data, resulting in latency spikes and context window noise).

AdaptKG-RAG solves this by introducing an **Adaptive Retrieval Controller**. It predicts the hop complexity of a question before querying the graph, dynamically adjusting the BFS retrieval depth on a per-question basis.

## 2. Main Research Contribution

The primary contribution is a lightweight, confidence-aware complexity classifier (TF-IDF + Logistic Regression) that dictates retrieval depth. By incorporating a fallback mechanism for low-confidence predictions, AdaptKG-RAG achieves the high accuracy of maximum-depth retrieval while significantly reducing the average candidate triple count and query latency.

## 3. Architecture

1. **Entity Detection:** Identifies seed entities using `[bracket_match]` or fallback longest n-gram index matching.
2. **Hop Classifier:** Analyzes the question and predicts complexity (1, 2, or 3 hops).
3. **Adaptive Depth Selection:** If confidence > 0.60, the predicted depth is used. Otherwise, it safely falls back to `min(predicted + 1, 3)`.
4. **Knowledge Graph Retrieval:** Performs bidirectional BFS from the seed entities out to the selected depth.
5. **Triple Ranking:** Ranks retrieved subgraph triples using lexical TF-IDF cosine similarity against the question.
6. **Answer Engine:** A deterministic rule-based walker that extracts the exact answer and formats the evidence path.
7. **FastAPI Backend:** Exposes the pipeline via `/api/ask`.
8. **React Frontend:** A Vite-powered SPA for interactive querying and research evaluation visualization.

## 4. Dataset

The project evaluates this architecture on the **MetaQA Dataset**.
* **Entities:** 43,234
* **Triples:** 134,741
* **Relations:** 9 types (e.g., `directed_by`, `starred_actors`, `has_genre`)

Dataset location: `sem7pro/metaQA/`

## 5. Evaluation Results

Evaluation was conducted against fixed-hop baselines on a combined test set.

| Method | Exact Match Accuracy | Avg Candidate Triples | Avg Retrieval Time | Avg Selected Depth |
|--------|----------------------|-----------------------|--------------------|--------------------|
| Baseline: Fixed 1-hop | 36.6% | ~7 | 0.1ms | 1.0 |
| Baseline: Fixed 2-hop | 30.0% | ~1,155 | 7.1ms | 2.0 |
| Baseline: Fixed 3-hop | 23.3% | ~14,178 | 171.8ms | 3.0 |
| **AdaptKG-RAG (Proposed)** | **86.6%** | **~9,558** | **144.3ms** | **2.0** |

*Note: The hop classifier achieved 100% accuracy on the evaluated development sample, proving that lightweight ML is highly effective for structural complexity prediction on templated datasets.*

## 6. Project Structure

```text
sem7pro/
├── backend/
│   ├── api/                 # FastAPI endpoints (main.py)
│   ├── services/            # Core logic (adaptive pipeline, retrieval, classifier, answering)
│   ├── utils/               # Metrics, explainability, pathing utilities
│   └── scripts/             # Training, evaluation, and dataset inspection scripts
├── frontend/                # React + Vite UI
├── metaQA/                  # The dataset
├── models/                  # Saved artifacts (joblib)
├── results/                 # Evaluation outputs (CSVs, JSONs, Charts)
└── tests/                   # Pytest suite (42/42 passing)
```

## 7. Installation & Requirements

### Dependencies
* **Backend:** Python 3.12+, `networkx>=3.0`, `scikit-learn>=1.3`, `fastapi`, `uvicorn`, `pytest`
* **Frontend:** Node.js 18+, `react`, `vite`, `recharts`

### Setup

```bash
# 1. Install Backend Requirements
pip install networkx scikit-learn joblib pytest fastapi uvicorn pydantic pandas matplotlib seaborn

# 2. Install Frontend Requirements
cd frontend
npm install
```

## 8. Usage Commands

**Run the Backend (FastAPI)**
```bash
# From the project root
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```
*API Endpoints: `GET /api/health`, `GET /api/stats`, `GET /api/results`, `POST /api/ask`*

**Run the Frontend (React)**
```bash
cd frontend
npm run dev
# The UI will be available at http://localhost:5173
```

## 10. Deploy to Render

The repository includes a Render Blueprint and Dockerfile that build the Vite
frontend and serve it from the FastAPI application on the same origin.

1. Push the project to a GitHub or GitLab repository.
2. In Render, choose **New → Blueprint**, connect that repository, and deploy
   the `render.yaml` Blueprint.
3. Wait for the service health check at `/api/health` to pass, then open the
   service URL to use the application.

The deployment includes the MetaQA data, classifier artifacts, and evaluation
results, so the service can take a little while to build and initialize. The
free Render service may spin down when idle and take longer to respond on the
first request after waking.

**Run the Test Suite**
```bash
python -m pytest tests/ -v
```

**Run Research Evaluation**
```bash
python backend/scripts/run_research_evaluation.py
```

## 9. Limitations & Future Work

* **Rule-Based Answer Engine:** The current answer extraction relies on deterministic relation parsing. While perfect for MetaQA's templated structure, replacing it with an LLM-based generative answerer would allow for free-form QA.
* **Lexical Ranking:** Triple ranking currently uses TF-IDF. Future work should upgrade this to dense vector embeddings (e.g., SentenceTransformers) to capture semantic similarity.
* **"Hub" Entity Explosion:** 3-hop retrieval from highly connected nodes (like `[English]` or `[Drama]`) causes memory bottlenecks. Future iterations should implement neighbor sampling or PageRank pruning during the BFS traversal.
