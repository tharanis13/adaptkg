import sys
from pathlib import Path
import csv
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.services.knowledge_graph import load_graph
from backend.services.hop_classifier import load_artifacts
from backend.services.adaptive_pipeline import run_adaptive_pipeline
from backend.scripts.run_research_evaluation import main as run_eval

class AppState:
    kg = None
    vectorizer = None
    model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Knowledge Graph and Classifier models...")
    AppState.kg = load_graph()
    AppState.vectorizer, AppState.model = load_artifacts()
    print("Models loaded successfully.")
    yield
    print("Shutting down...")

app = FastAPI(title="AdaptKG-RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/stats")
def stats():
    if not AppState.kg:
        raise HTTPException(status_code=503, detail="KG not loaded yet")
    return AppState.kg.get_graph_statistics()

@app.get("/api/model-info")
def model_info():
    report_path = _PROJECT_ROOT / "results" / "hop_classifier_metrics.json"
    if report_path.exists():
        with open(report_path, "r") as f:
            return json.load(f)
    return {"info": "Model trained but detailed metrics not found."}

@app.get("/api/results")
def get_results():
    results_dir = _PROJECT_ROOT / "results"
    summary_path = results_dir / "large_scale" / "phase10_evaluation_summary_3000.csv"
    if not summary_path.exists():
        summary_path = results_dir / "evaluation_summary.csv"
    
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not found")
    
    records = []
    with open(summary_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                method_val = row.get("Method") or row.get("Hop") or "Unknown"
                row["Method"] = method_val
                records.append(row)
    
    clf_path = results_dir / "classifier_report.json"
    clf_data = {}
    if clf_path.exists():
        with open(clf_path, "r", encoding="utf-8") as f:
            clf_data = json.load(f)
            
    return {
        "evaluation_summary": records,
        "classifier_report": clf_data
    }

@app.post("/api/ask")
def ask(req: AskRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
        
    try:
        result = run_adaptive_pipeline(
            question=req.question,
            kg=AppState.kg,
            vectorizer=AppState.vectorizer,
            model=AppState.model
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate")
def evaluate(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_eval)
    return {"status": "Evaluation triggered in the background. Check /api/results later."}

_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
