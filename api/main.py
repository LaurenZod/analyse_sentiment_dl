import os
import uuid
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

import mlflow
import yaml
import mlflow.transformers
import json
import time

# --- HF transformers imports & CPU/offline defaults ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch

# --- Runtime defaults (CPU-friendly & offline-safe) ---
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(1)

# ---------- Config via variables d'environnement ----------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "tweet_prediction")
MODEL_ALIAS         = os.getenv("MODEL_ALIAS")          
ENV_MODEL_VERSION   = os.getenv("MODEL_VERSION")        
MODEL_STAGE         = os.getenv("MODEL_STAGE")          
API_KEY             = os.getenv("API_KEY", "")
PRODUCTION_LOG_PATH = os.getenv("PRODUCTION_LOG_PATH", "production_inferences.jsonl") 

# ---------- Modèles Pydantic ----------
class PredictIn(BaseModel):
    text: str
    @field_validator("text")
    def non_empty(cls, v: str):
        if not v or not v.strip():
            raise ValueError("text cannot be empty")
        return v.strip()

class PredictBatchIn(BaseModel):
    texts: List[str]
    @field_validator("texts")
    def non_empty_list(cls, v: List[str]):
        if not v:
            raise ValueError("texts cannot be empty")
        return [t.strip() for t in v if t and t.strip()]

class PredictOut(BaseModel):
    label: str
    score: float
    raw_label: str
    model_name: str
    model_version: Optional[str] = None
    model_stage: Optional[str] = None
    prediction_id: Optional[str] = None
    timestamp: Optional[float] = None

class HealthOut(BaseModel):
    status: str
    model_uri: str
    model_name: str
    model_stage: str
    model_version: Optional[str] = None

# --- NOUVEAU MODÈLE POUR LE FEEDBACK ---
class FeedbackIn(BaseModel):
    prediction_id: str
    ground_truth: str
    
    @field_validator("ground_truth")
    def validate_sentiment(cls, v: str):
        v = v.strip().lower()
        if v not in {"positive", "negative"}:
            raise ValueError("ground_truth must be 'positive' or 'negative'")
        return v
# ------------------------------------

# ---------- Chargement du modèle au démarrage ----------
PIPELINE = None
MODEL_URI = None
MODEL_VERSION = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global PIPELINE, MODEL_URI, MODEL_VERSION
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # 1) Alias prioritaire
    if MODEL_ALIAS:
        MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            MODEL_VERSION = mv.version
        except Exception:
            MODEL_VERSION = None
    # 2) Version explicite
    elif ENV_MODEL_VERSION:
        MODEL_URI = f"models:/{MODEL_NAME}/{ENV_MODEL_VERSION}"
        MODEL_VERSION = ENV_MODEL_VERSION
    # 3) Stage (legacy)
    elif MODEL_STAGE and MODEL_STAGE.lower() != "none":
        MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        try:
            versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            if versions:
                MODEL_VERSION = versions[0].version
        except Exception:
            MODEL_VERSION = None
    else:
        raise RuntimeError(
            "Aucune stratégie de sélection de modèle n'est fournie. "
            "Définis au choix MODEL_ALIAS, MODEL_VERSION ou MODEL_STAGE."
        )

    # --- Load model artifacts locally and build a plain HF pipeline (bypass mlflow.transformers loader) ---
    local_root = mlflow.artifacts.download_artifacts(MODEL_URI)

    # --- Resolve exact HF component paths from MLflow MLmodel (robust to layout) ---
    mlmodel_path = os.path.join(local_root, "MLmodel")
    if not os.path.isfile(mlmodel_path):
        raise RuntimeError(f"Fichier MLmodel introuvable sous {local_root}")

    try:
        with open(mlmodel_path, "r", encoding="utf-8") as f:
            mlmodel = yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Impossible de parser MLmodel: {e}")

    flavors = (mlmodel.get("flavors") or {})
    tflavor = (flavors.get("transformers") or {})
    components = (tflavor.get("components") or {})

    # Composants possibles selon mlflow.transformers
    # CORRECTION: On neutralise l'accès direct (qui cause l'AttributeError)
    model_rel = None
    tok_rel = None

    # Fallback: certains logs mettent tout sous 'components'
    if not model_rel:
        # Essaye quelques chemins classiques
        for cand in ("components/model", "data/model", "model"):
            if os.path.isdir(os.path.join(local_root, cand)):
                model_rel = cand
                break

    if not tok_rel:
        for cand in ("components/tokenizer", "components/tokenizer_fast", "components/tokenizer_json", "data/tokenizer", "tokenizer"):
            if os.path.isdir(os.path.join(local_root, cand)):
                tok_rel = cand
                break

    if not model_rel:
        raise RuntimeError(f"Impossible de localiser le dossier modèle HF dans {local_root}; components={components}")

    model_dir = os.path.join(local_root, model_rel)
    tok_dir = os.path.join(local_root, tok_rel) if tok_rel else model_dir

    # Vérifications minimales
    model_cfg = os.path.join(model_dir, "config.json")
    if not os.path.isfile(model_cfg):
        # Parfois le modèle est directement dans un sous-dossier unique
        subdirs = [d for d in (os.listdir(model_dir) if os.path.isdir(model_dir) else []) if os.path.isdir(os.path.join(model_dir, d))]
        for sd in subdirs:
            cfg = os.path.join(model_dir, sd, "config.json")
            if os.path.isfile(cfg):
                model_dir = os.path.join(model_dir, sd)
                model_cfg = cfg
                break
    if not os.path.isfile(model_cfg):
        raise RuntimeError(f"config.json introuvable dans {model_dir}")

    # --- Load strictly from disk (no Hub), on CPU ---
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

    PIPELINE = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=-1,   # CPU
        top_k=None
    )
    if not hasattr(PIPELINE, "__call__"):
        raise RuntimeError("Loaded object is not a transformers Pipeline.")

    try:
        yield
    finally:
        PIPELINE = None

# ---------- App & CORS ----------
app = FastAPI(title="Tweet Sentiment API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthOut)
def health():
    return HealthOut(
        status="ok",
        model_uri=MODEL_URI or "",
        model_name=MODEL_NAME,
        model_stage=(MODEL_ALIAS or MODEL_STAGE or ""),
        model_version=MODEL_VERSION,
    )

# ---------- Sécurité simple par clé API (facultatif) ----------
def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ---------- Helpers ----------
def _normalize_label(raw_label: str, id2label: Optional[Dict[int, str]] = None) -> str:
    raw = raw_label.upper()
    if raw in {"NEGATIVE", "POSITIVE"}:
        return raw.lower()
    if raw.startswith("LABEL_"):
        try:
            idx = int(raw.split("_")[-1])
        except Exception:
            return raw_label
        if idx == 0:
            return "negative"
        if idx == 1:
            return "positive"
    return raw_label

def _run_pipeline_single(text: str) -> Dict[str, Any]:
    out = PIPELINE(text, truncation=True)
    
    # CORRECTION: Le pipeline Hugging Face peut renvoyer [[{...}]] ou [{...}]
    if isinstance(out, list) and out:
        # Cas 1: Si c'est [[{...}]] (sortie batch d'un seul élément)
        if isinstance(out[0], list) and out[0]:
            result = out[0][0] # Prend l'objet de prédiction
        # Cas 2: Si c'est [{...}]
        elif isinstance(out[0], dict):
            result = out[0] # Prend l'objet de prédiction
        else:
            raise HTTPException(status_code=500, detail="Unexpected pipeline output format")
    else:
        # Cas où 'out' n'est pas une liste valide (ou vide)
        raise HTTPException(status_code=500, detail="Unexpected pipeline output format")
        
    raw_label = result["label"]
    score = float(result["score"])

    id2label = getattr(getattr(PIPELINE, "model", None), "config", None)
    id2label = getattr(id2label, "id2label", None)

    label = _normalize_label(raw_label, id2label=id2label)
    return {"label": label, "score": score, "raw_label": raw_label}

# --- NOUVELLE FONCTION POUR METTRE À JOUR LE LOG AVEC LE GROUND TRUTH ---
def _update_log_with_feedback(prediction_id: str, ground_truth: str):
    """Lit le log, met à jour l'entrée correspondante, et réécrit le fichier."""
    
    lines = []
    found = False
    
    try:
        # Lire toutes les lignes
        with open(PRODUCTION_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Trouver la ligne correspondante au prediction_id (clé unique)
                    if data.get("prediction_id") == prediction_id:
                        data["ground_truth"] = ground_truth.lower()
                        found = True
                    lines.append(json.dumps(data))
                except json.JSONDecodeError:
                    # Gérer le cas où une ligne n'est pas un JSON valide
                    lines.append(line.strip()) 

    except FileNotFoundError:
        # Si le fichier n'existe pas, rien à faire
        raise HTTPException(status_code=404, detail=f"Log file not found or empty.")

    if not found:
        # S'assurer que l'inférence a été trouvée
        raise HTTPException(status_code=404, detail=f"Inference with prediction_id {prediction_id} not found.")

    # Réécrire l'intégralité du fichier avec la ligne mise à jour
    try:
        # Utiliser 'w' (write) pour écraser et réécrire toutes les lignes
        with open(PRODUCTION_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rewriting log file: {str(e)}")

    return {"message": "Ground truth updated successfully"}
# -------------------------------------------------------------------


# ---------- Endpoints ----------
@app.post("/predict", response_model=PredictOut, dependencies=[Depends(require_api_key)])
def predict(body: PredictIn):
    start_time = time.time()
    
    try:
        prediction_timestamp = time.time()
        prediction_id = str(uuid.uuid4())
        res = _run_pipeline_single(body.text)
        
        # --- LOGGING DE LA PRODUCTION ---
        log_data = {
            "timestamp": prediction_timestamp,
            "prediction_id": prediction_id,
            "model_version": MODEL_VERSION,
            "input_text": body.text,
            "prediction_label": res["label"],
            "prediction_score": res["score"],
            "latency_ms": (time.time() - start_time) * 1000,
            "ground_truth": None # Laissé à None pour être rempli via /feedback
        }
        
        # Le mode "a" permet d'ajouter à la fin du fichier (append)
        with open(PRODUCTION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data) + "\n")
        # -----------------------------------------------

        return PredictOut(
            label=res["label"],
            score=res["score"],
            raw_label=res["raw_label"],
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            model_stage=MODEL_STAGE,
            prediction_id=prediction_id,
            timestamp=prediction_timestamp,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=List[PredictOut], dependencies=[Depends(require_api_key)])
def predict_batch(body: PredictBatchIn):
    try:
        outs = []
        for txt in body.texts:
            r = _run_pipeline_single(txt)
            outs.append(PredictOut(
                label=r["label"],
                score=r["score"],
                raw_label=r["raw_label"],
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                model_stage=MODEL_STAGE,
            ))
        return outs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NOUVEL ENDPOINT POUR LE GROUND TRUTH ---
@app.post("/feedback", dependencies=[Depends(require_api_key)])
def submit_feedback(body: FeedbackIn):
    """Permet de fournir le ground truth pour une inférence précédente en utilisant prediction_id comme ID."""
    try:
        res = _update_log_with_feedback(
            prediction_id=body.prediction_id, 
            ground_truth=body.ground_truth
        )
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# -------------------------------------------