from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os, json
import numpy as np
from openai import OpenAI
from alignment import split_claims, split_charter_articles

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.60"))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

CORPUS_PATH = os.getenv("CORPUS_PATH", "./storage/corpus.json")

app = FastAPI(title="Verificador ONU/OEA", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UploadCorpusRequest(BaseModel):
    org: str
    text: str

class ClaimResult(BaseModel):
    claim: str
    aligned: bool
    score: float
    org: Optional[str] = None
    articulo: Optional[str] = None
    citation: Optional[str] = None

class EvaluateRequest(BaseModel):
    texto: str
    org: Optional[str] = "ambas"
    threshold: Optional[float] = None

class EvaluateResponse(BaseModel):
    porcentaje_alineacion: float
    porcentaje_onu: Optional[float] = None
    porcentaje_oea: Optional[float] = None
    total_claims: int
    total_alineados: int
    detalle: List[ClaimResult]

def _load_corpus():
    if not os.path.exists(CORPUS_PATH):
        return {"ONU": [], "OEA": []}
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_corpus(data):
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def _embed(texts: List[str]):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurado.")
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

@app.get("/health")
def health():
    corpus = _load_corpus()
    return {"status": "ok", "onu_articulos": len(corpus.get("ONU", [])), "oea_articulos": len(corpus.get("OEA", []))}

@app.post("/upload_corpus")
def upload_corpus(req: UploadCorpusRequest):
    org = req.org.strip().upper()
    if org not in ("ONU", "OEA"):
        raise HTTPException(400, "org debe ser 'ONU' u 'OEA'")
    articles = split_charter_articles(req.text)
    embs = _embed([a["text"] for a in articles])
    for i, a in enumerate(articles):
        a["embedding"] = embs[i]
    corpus = _load_corpus()
    corpus[org] = articles
    _save_corpus(corpus)
    return {"ok": True, "org": org, "articulos": len(articles)}

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    texto = req.texto.strip()
    if not texto:
        raise HTTPException(400, "texto vacío")
    org = (req.org or "ambas").lower()
    threshold = float(req.threshold) if req.threshold is not None else DEFAULT_THRESHOLD

    corpus = _load_corpus()
    if (org in ("onu","ambas") and not corpus.get("ONU")) or (org in ("oea","ambas") and not corpus.get("OEA")):
        raise HTTPException(400, "El corpus no está cargado. Usa /upload_corpus antes.")

    claims = split_claims(texto)
    if not claims:
        return EvaluateResponse(porcentaje_alineacion=0.0,total_claims=0,total_alineados=0,detalle=[])

    claim_embs = _embed(claims)

    detalle = []
    alineados_global = 0
    aligned_counts = {"ONU":0,"OEA":0}
    total_counts = {"ONU":0,"OEA":0}

    def best_match(c_vec, articles):
        best=None; best_score=-1.0
        for art in articles:
            art_vec = np.array(art["embedding"], dtype=float)
            s=float(np.dot(c_vec, art_vec)/((np.linalg.norm(c_vec)+1e-9)*(np.linalg.norm(art_vec)+1e-9)))
            if s>best_score: best_score=s; best=art
        return best, best_score

    for c_text, c_vec in zip(claims, claim_embs):
        c_vec=np.array(c_vec,dtype=float)
        best_records=[]
        if org in ("onu","ambas"):
            art_onu,sc_onu=best_match(c_vec, corpus["ONU"]); total_counts["ONU"]+=1
            best_records.append(("ONU",art_onu,sc_onu))
        if org in ("oea","ambas"):
            art_oea,sc_oea=best_match(c_vec, corpus["OEA"]); total_counts["OEA"]+=1
            best_records.append(("OEA",art_oea,sc_oea))
        best_org,best_art,best_score=max(best_records,key=lambda x:x[2])
        aligned=best_score>=threshold
        if aligned:
            alineados_global+=1; aligned_counts[best_org]+=1
        articulo_label=best_art.get("article") if best_art else None
        citation=f"{best_org} {articulo_label}" if (best_org and articulo_label) else None
        detalle.append({"claim":c_text,"aligned":aligned,"score":round(best_score,4),
                        "org":best_org,"articulo":articulo_label,"citation":citation})

    pct_onu=(aligned_counts["ONU"]/total_counts["ONU"]) if total_counts["ONU"] else None
    pct_oea=(aligned_counts["OEA"]/total_counts["OEA"]) if total_counts["OEA"] else None

    if org=="ambas":
        parts=[p for p in (pct_onu,pct_oea) if p is not None]; pct_global=sum(parts)/len(parts) if parts else 0.0
    elif org=="onu":
        pct_global=pct_onu or 0.0
    else:
        pct_global=pct_oea or 0.0

    return EvaluateResponse(
        porcentaje_alineacion=round(pct_global or 0.0,4),
        porcentaje_onu=round(pct_onu,4) if pct_onu is not None else None,
        porcentaje_oea=round(pct_oea,4) if pct_oea is not None else None,
        total_claims=len(claims),
        total_alineados=alineados_global,
        detalle=[ClaimResult(**d) for d in detalle]
    )
