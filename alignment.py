import re
from typing import List, Dict
import numpy as np

def split_claims(texto: str, min_len: int = 20) -> List[str]:
    sents = re.split(r'(?<=[\.!?])\s+', texto.strip())
    claims = [s.strip() for s in sents if len(s.strip()) >= min_len]
    return claims if claims else ([texto.strip()] if texto.strip() else [])

def split_charter_articles(texto: str) -> List[Dict[str, str]]:
    t = re.sub(r'\r\n?', '\n', texto)
    pattern = re.compile(r'(?i)(?:art[íi]?culo|art\.|article)\s*([0-9IVXLC]+)')
    parts = []
    idxs = [(m.start(), m.group(0), m.group(1)) for m in pattern.finditer(t)]
    if not idxs:
        return [{"article": "GENERAL", "text": t}]
    for i, (start, hdr, num) in enumerate(idxs):
        end = idxs[i+1][0] if i+1 < len(idxs) else len(t)
        chunk = t[start:end].strip()
        parts.append({"article": f"{hdr} {num}".strip(), "text": chunk})
    return parts

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))
