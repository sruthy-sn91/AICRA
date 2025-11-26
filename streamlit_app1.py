from __future__ import annotations

import io, os, re, json, sys, subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ===================== Env guards & defaults =====================
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.90")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.50")

def _ensure_transformers(min_version="4.38.0"):
    try:
        import transformers, packaging.version as _pv  # noqa
        if _pv.parse(transformers.__version__) < _pv.parse(min_version):
            raise ImportError(f"transformers<{min_version}")
    except Exception:
        print(f"[setup] Installing/upgrading 'transformers>={min_version}' …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"transformers>={min_version}"])

_ensure_transformers()

# Lightweight parsers
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    import docx  # python-docx
except Exception:
    docx = None

# ML deps
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

from sentence_transformers import SentenceTransformer
import faiss

# .env (optional) for GROQ_API_KEY
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# ---------------- Presets ----------------
LLM_PRESETS = {
    "Base: Qwen2.5-1.5B (no LoRA)": {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_dir": None,
        "use_adapter": False,
    },
    "LoRA: qwen2.5-1.5B → lora_out_qwen25_1p5b_fast": {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_dir": "lora_out_qwen25_1p5b_fast",
        "use_adapter": True,
    },
    "LoRA: qwen2-7B → lora_out_qwen2_7b_fast": {
        "base_model": "Qwen/Qwen2-7B-Instruct",
        "adapter_dir": "lora_out_qwen2_7b_fast",
        "use_adapter": True,
    },
    "Custom…": {  # Will read from user inputs
        "base_model": "",
        "adapter_dir": "",
        "use_adapter": False,
    },
}

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

# ---------------- Helpers ----------------
def device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEV = device_str()

# Avoid capture groups so re.split doesn't inject None
CLAUSE_SPLIT_HEADING_RE = re.compile(
    r"\n\s*(?:Section|Clause|Article)\s*\d+[\.:)]?\s+|\n\s*[A-Z][A-Z\- ]{3,}\n"
)

# CUAD-style severity mapping
SEVERITY_RULES = {
    "Uncapped Liability": 3,
    "Indemnification": 3,
    "Liquidated Damages": 3,
    "Non-Compete, Exclusivity, No-Solicit of Customers": 3,
    "IP Ownership Assignment": 3,
    "Covenant not to Sue Release of Claims": 3,
    "Termination for Convenience": 2,
    "Warranty Duration": 2,
    "Audit Rights": 2,
    "Insurance": 2,
    "Minimum Commitment": 2,
    "Most Favored Nation": 2,
    "No-Solicit of Employees": 2,
    "Post-Termination Services": 2,
    "Price Restrictions": 2,
    "ROFR-ROFO-ROFN": 2,
    "Revenue-Profit Sharing": 2,
    "Source Code Escrow": 2,
    "Third Party Beneficiary": 2,
    "Unlimited-All-You-Can-Eat License": 2,
    "Volume Restriction": 2,
    "Joint IP Ownership": 2,
    "Licenses": 2,
    "Governing Law": 1,
    "Dates": 1,
    "Document Name": 1,
    "Parties": 1,
    "Anti-assignment, CIC": 2,
}
SEVERITY_LABELS = {3: "High", 2: "Medium", 1: "Low"}

def _head_weight_norm(model) -> Optional[float]:
    try:
        for n, p in model.named_parameters():
            if any(k in n for k in ["classifier.weight", "score.weight", "out_proj.weight"]):
                return float(torch.linalg.norm(p.detach().cpu()))
    except Exception:
        pass
    return None

def arrow_safe_view(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(
                lambda v: (
                    json.dumps(v) if isinstance(v, (list, tuple, dict))
                    else ("" if v is None or (isinstance(v, float) and np.isnan(v)) else v)
                )
            )
    return out[cols]

@st.cache_data(show_spinner=False)
def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[\t\r]", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r" {2,}", " ", s)
    return s.strip()

# ---------------- File parsing ----------------
def parse_pdf(file_bytes: bytes) -> str:
    if fitz is None:
        st.warning("PyMuPDF not installed. Falling back to simple PDF text. Try: pip install pymupdf")
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        txt = "\n\n".join([p.extract_text() or "" for p in reader.pages])
        return clean_text(txt)
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    texts = [page.get_text("text") or "" for page in doc]
    return clean_text("\n\n".join(texts))

def parse_docx(file_bytes: bytes) -> str:
    if docx is None:
        st.warning("python-docx not installed. Try: pip install python-docx")
        return ""
    buf = io.BytesIO(file_bytes)
    d = docx.Document(buf)
    return clean_text("\n".join(p.text for p in d.paragraphs))

def parse_txt(file_bytes: bytes) -> str:
    try:
        return clean_text(file_bytes.decode("utf-8", errors="ignore"))
    except Exception:
        return clean_text(file_bytes.decode("latin-1", errors="ignore"))

SUPPORTED_TYPES = {
    ".pdf": ("PDF", parse_pdf),
    ".docx": ("DOCX", parse_docx),
    ".txt": ("TXT", parse_txt),
}

# ---------------- Clause segmentation ----------------
def segment_clauses(text: str, min_len: int = 40) -> List[Dict[str, Any]]:
    if not text:
        return []
    parts = CLAUSE_SPLIT_HEADING_RE.split("\n" + text)
    chunks: List[str] = []
    if len(parts) > 1:
        for part in parts:
            part = (part or "").strip()
            if part:
                chunks.append(part)
    else:
        sents = re.split(r"(?<=[\.!?])\s+", text)
        buf: List[str] = []
        for s in sents:
            s = (s or "").strip()
            if not s:
                continue
            buf.append(s)
            if len(" ".join(buf)) >= 400:
                chunks.append(" ".join(buf)); buf = []
        if buf:
            chunks.append(" ".join(buf))
    cleaned = [c.strip() for c in chunks if c and len(c.strip()) >= min_len]
    merged: List[str] = []
    i = 0
    while i < len(cleaned):
        cur = cleaned[i]
        if (
            i + 1 < len(cleaned)
            and len(cur) <= 80
            and (cur.isupper() or re.match(r"^(?:Section|Clause|Article)\s*\d", cur, flags=re.I))
        ):
            merged.append((cur + " — " + cleaned[i + 1]).strip())
            i += 2
        else:
            merged.append(cur); i += 1
    return [{"clause_id": j, "text": t} for j, t in enumerate(merged)]

# ---------------- Classifier export discovery (supports safetensors/bin) ----------------
def _has_weights(dir_path: Path) -> Tuple[bool, Optional[Path]]:
    st_path = dir_path / "model.safetensors"
    bin_path = dir_path / "pytorch_model.bin"
    if st_path.exists() and st_path.stat().st_size > 0:
        return True, st_path
    if bin_path.exists() and bin_path.stat().st_size > 0:
        return True, bin_path
    return False, None

def resolve_classifier_dir(path_str: str) -> Tuple[str, Optional[str]]:
    p = Path(path_str).expanduser().resolve()
    note = None
    if p.is_dir() and "checkpoint-" in p.as_posix():
        sib = p.parent / "model"
        if sib.is_dir():
            p = sib.resolve()
            note = f"Detected checkpoint path. Using sibling export: {p}"
    if p.is_dir() and not (p / "config.json").exists() and (p / "model").is_dir():
        p = (p / "model").resolve()
        note = f"Using contained export: {p}"
    cfg = p / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"config.json missing under: {p}")
    ok, _ = _has_weights(p)
    if not ok:
        raise FileNotFoundError(
            f"Neither model.safetensors nor pytorch_model.bin found under: {p}\n"
            f"If you trained recently, the default is model.safetensors."
        )
    return str(p), note

def find_best_export_dir(root: str = "Clause_classifier/outputs_models") -> Optional[str]:
    rootp = Path(root).expanduser()
    if not rootp.exists():
        return None
    csvs = sorted(rootp.glob("models_leaderboard_*.csv"))
    if csvs:
        try:
            df = pd.read_csv(csvs[-1])
            if "out_dir" in df.columns and len(df):
                best = df.sort_values("test_macro_f1", ascending=False).iloc[0]["out_dir"]
                model_dir = Path(str(best)).expanduser() / "model"
                ok, _ = _has_weights(model_dir)
                if model_dir.is_dir() and (model_dir / "config.json").exists() and ok:
                    return str(model_dir.resolve())
        except Exception:
            pass
    candidates = []
    for run in rootp.iterdir():
        if not run.is_dir():
            continue
        model_dir = run / "model"
        if model_dir.is_dir() and (model_dir / "config.json").exists() and _has_weights(model_dir)[0]:
            candidates.append((model_dir.stat().st_mtime, model_dir))
    if candidates:
        candidates.sort(reverse=True)
        return str(candidates[0][1].resolve())
    return None

# ---------------- Cached model loaders ----------------
@st.cache_resource(show_spinner=True)
def load_classifier(model_dir_input: str):
    chosen, note = resolve_classifier_dir(model_dir_input)
    tok = AutoTokenizer.from_pretrained(chosen, use_fast=True, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(chosen, trust_remote_code=True)
    model.eval(); model.to("cpu")
    id2label_preview = list(_normalize_id2label(model).items())[:5]
    head_norm = _head_weight_norm(model)
    return tok, model, chosen, note, id2label_preview, head_norm

@st.cache_resource(show_spinner=True)
def load_embedder(model_name: str = "intfloat/e5-large-v2"):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=True)
def load_chat_model(base_model: str, adapter_dir: Optional[str]):
    dtype = torch.float32  # CPU for stability on macOS
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        trust_remote_code=True,
        device_map="cpu",
    )
    if adapter_dir and PeftModel is not None and Path(adapter_dir).exists():
        base = PeftModel.from_pretrained(base, adapter_dir)
    elif adapter_dir and PeftModel is None:
        st.warning("peft is not installed; loading base model without LoRA.")
    elif adapter_dir and not Path(adapter_dir).exists():
        st.warning(f"LoRA adapter dir not found: {adapter_dir} — loading base model.")
    base.eval()
    return tok, base

# ---------------- Classification ----------------
def _softmax(x: torch.Tensor) -> torch.Tensor:
    return nn.functional.softmax(x, dim=-1)

def _normalize_id2label(model) -> Dict[int, str]:
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and id2label:
        try:
            return {int(k): v for k, v in id2label.items()}
        except Exception:
            pass
    l2i = getattr(model.config, "label2id", None)
    if isinstance(l2i, dict) and l2i:
        return {int(i): lbl for lbl, i in l2i.items()}
    return {i: f"label_{i}" for i in range(model.config.num_labels)}

def _ensure_attention_mask(tok, enc):
    if isinstance(enc, torch.Tensor):
        input_ids = enc
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        attn = (input_ids != pad_id).long()
        return {"input_ids": input_ids.cpu(), "attention_mask": attn.cpu()}
    elif isinstance(enc, dict):
        ids = enc.get("input_ids")
        if ids is None and hasattr(enc, "data"):
            ids = enc.data.get("input_ids")
        if "attention_mask" in enc and enc["attention_mask"] is not None:
            attn = enc["attention_mask"]
        else:
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            attn = (ids != pad_id).long()
        return {"input_ids": ids.cpu(), "attention_mask": attn.cpu()}
    else:
        ids = enc.data["input_ids"]
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        attn = (ids != pad_id).long()
        return {"input_ids": ids.cpu(), "attention_mask": attn.cpu()}

def classify_clauses(
    clauses: List[Dict[str, Any]],
    tok,
    model,
    max_length: int = 512,
    stride: int = 128,
    min_confidence: float = 0.15,
    min_margin: float = 0.10,
    force_label_when_uncertain: bool = False,
    debug_probs: bool = False,
) -> List[Dict[str, Any]]:
    id2label = _normalize_id2label(model)
    model.eval()
    for c in clauses:
        text = (c.get("text") or "").strip()
        if not text:
            c.update({"category": "(Unknown)", "confidence": 0.0, "severity": 1, "severity_label": "Low"})
            continue
        enc = tok(
            text, truncation=True, max_length=max_length,
            return_overflowing_tokens=True, stride=stride,
            padding="max_length", return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to("cpu"),
                attention_mask=enc["attention_mask"].to("cpu"),
            ).logits
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        probs = nn.functional.softmax(logits.mean(dim=0), dim=-1).cpu().numpy()
        order = np.argsort(probs)[::-1]
        top, second = int(order[0]), int(order[1])
        p1, p2 = float(probs[top]), float(probs[second])
        accept = (p1 >= min_confidence) or ((p1 - p2) >= min_margin and p1 >= 0.12)
        label = id2label.get(top, f"label_{top}") if (accept or force_label_when_uncertain) else "(Unknown)"
        c["category"] = label
        c["confidence"] = round(p1, 3)
        sev = SEVERITY_RULES.get(label, 1)
        c["severity"] = sev
        c["severity_label"] = SEVERITY_LABELS.get(sev, "Low")
        if debug_probs:
            c["top3"] = [(id2label.get(int(i), f"id{i}"), float(probs[i])) for i in order[:3]]
    return clauses

# ---------------- FAISS index & search ----------------
def build_doc_index(
    clauses: List[Dict[str, Any]], embedder
) -> Tuple[faiss.IndexFlatIP, np.ndarray, List[Dict[str, Any]]]:
    valid = [c for c in clauses if (c.get("text") or "").strip()]
    if not valid:
        raise ValueError("No non-empty clauses to index. If this is a scanned PDF, add OCR.")
    texts = [c["text"] for c in valid]
    embs = embedder.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    embs = np.asarray(embs, dtype="float32")
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    if embs.size == 0 or embs.shape[0] == 0:
        raise ValueError("Embedding model returned empty vectors.")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, embs, valid

def search_doc(
    index: faiss.IndexFlatIP, embedder, query: str, clauses: List[Dict[str, Any]], k: int = 5
) -> List[Dict[str, Any]]:
    q = embedder.encode([query], normalize_embeddings=True)
    q = np.array(q, dtype="float32")
    scores, ids = index.search(q, k)
    hits = []
    for sc, i in zip(scores[0], ids[0]):
        if i == -1: continue
        item = clauses[int(i)].copy(); item["score"] = float(sc); hits.append(item)
    return hits

# ---------------- RAG chat helpers ----------------
def supports_chat_template(tok) -> bool:
    return getattr(tok, "chat_template", None) not in (None, "")

def format_context(hits: List[Dict[str, Any]], max_chars: int = 1400) -> str:
    lines, used = [], 0
    for h in hits:
        snippet = " ".join((h.get("text") or "").split())
        if not snippet: continue
        remaining = max_chars - used
        if remaining <= 0: break
        if len(snippet) > remaining: snippet = snippet[:remaining] + " …"
        lines.append(f"[clause_id {h['clause_id']}] {snippet}")
        used += len(snippet)
    return "\n\n".join(lines)

PROMPT_TEMPLATE = (
    "You are a contracts analyst. Answer the user based ONLY on the retrieved contract clauses.\n"
    "Cite clause_id(s) you used. Keep answer concise (3-6 sentences).\n\n"
    "### Question\n{question}\n\n### Retrieved Clauses\n{contexts}\n\n### Answer\n"
)

def generate_answer_local(tok, model, question: str, hits: List[Dict[str, Any]], max_new_tokens: int = 256) -> str:
    system = ("You are a contracts analyst. Answer ONLY from the retrieved contract clauses. "
              "Cite clause_id(s) used. Keep answers concise (3–6 sentences).")
    ctx = format_context(hits, max_chars=1400)
    if supports_chat_template(tok):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"### Question\n{question.strip()}\n\n### Retrieved Clauses\n{ctx}\n\n### Answer"}
        ]
        enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    else:
        prompt_text = PROMPT_TEMPLATE.format(question=question.strip(), contexts=ctx)
        enc = tok(prompt_text, return_tensors="pt", truncation=True, max_length=3800)

    inputs = _ensure_attention_mask(tok, enc)
    gen_common = dict(max_new_tokens=max_new_tokens,
                      pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)

    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, **gen_common)
    text = tok.decode(out[0], skip_special_tokens=True)
    answer = text.split("### Answer", 1)[-1].strip() if "### Answer" in text else text.strip()

    if not answer or re.fullmatch(r"[!?.\s]{8,}", answer) or answer.lower().startswith("system "):
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=True, temperature=0.4, top_p=0.9, **gen_common)
        text = tok.decode(out[0], skip_special_tokens=True)
        answer = text.split("### Answer", 1)[-1].strip() if "### Answer" in text else text.strip()
    return answer.strip()

def generate_answer_groq(question: str, hits: List[Dict[str, Any]], model_id: str, max_new_tokens: int = 256) -> str:
    import requests
    ctx = format_context(hits, max_chars=1400)
    sys = "You are a contracts analyst. Answer ONLY from the retrieved contract clauses. Cite clause_id(s) used. Keep answers concise (3–6 sentences)."
    api_key = os.environ.get("GROQ_API_KEY") or (getattr(st, "secrets", {}) or {}).get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": sys},
                     {"role": "user", "content": f"### Question\n{question.strip()}\n\n### Retrieved Clauses\n{ctx}\n\n### Answer"}],
        "temperature": 0.0, "max_tokens": max_new_tokens,
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                      json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# --- Generic Groq chat for non-RAG (used in Compare Versions) ---
def groq_chat(system: str, user: str, model_id: str, max_new_tokens: int = 512) -> str:
    import requests
    api_key = os.environ.get("GROQ_API_KEY") or (getattr(st, "secrets", {}) or {}).get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
    }
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload, timeout=60
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# === Session-state helpers and Compare-Version utilities =====================
from difflib import SequenceMatcher

def _ensure_state(defaults: Dict[str, Any]) -> None:
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _embed_norm(embedder, texts: List[str]) -> np.ndarray:
    embs = embedder.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    embs = np.asarray(embs, dtype="float32")
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    return embs

def align_and_diff(old_clauses: List[Dict[str, Any]], new_clauses: List[Dict[str, Any]],
                   embedder, sim_thresh: float = 0.80) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_texts = [c["text"] for c in old_clauses]
    new_texts = [c["text"] for c in new_clauses]
    if not old_texts or not new_texts:
        return (pd.DataFrame(columns=["new_id","text"]),
                pd.DataFrame(columns=["old_id","text"]),
                pd.DataFrame(columns=["old_id","new_id","sim","diff_ratio","old_text","new_text"]))
    Eo = _embed_norm(embedder, old_texts)
    En = _embed_norm(embedder, new_texts)
    S = np.matmul(Eo, En.T)
    O, N = S.shape
    matched = []
    used_o, used_n = set(), set()
    while True:
        i, j = np.unravel_index(np.argmax(S), S.shape)
        best = float(S[i, j])
        if best < sim_thresh:
            break
        matched.append((i, j, best))
        used_o.add(i); used_n.add(j)
        S[i, :] = -1e9; S[:, j] = -1e9
    removed = [i for i in range(O) if i not in used_o]
    added   = [j for j in range(N) if j not in used_n]
    added_df   = pd.DataFrame([{"new_id": j, "text": new_texts[j]} for j in added])
    removed_df = pd.DataFrame([{"old_id": i, "text": old_texts[i]} for i in removed])
    modified_rows = []
    for i, j, sim in matched:
        old_t, new_t = old_texts[i], new_texts[j]
        diff_ratio = 1.0 - SequenceMatcher(None, old_t, new_t).ratio()
        if diff_ratio > 0.001:
            modified_rows.append({
                "old_id": i, "new_id": j, "sim": round(sim, 3),
                "diff_ratio": round(diff_ratio, 4),
                "old_text": old_t, "new_text": new_t
            })
    modified_df = pd.DataFrame(modified_rows, columns=["old_id","new_id","sim","diff_ratio","old_text","new_text"])
    return added_df, removed_df, modified_df

def _bullets_for_llm(added_df, removed_df, modified_df, max_each=10) -> str:
    lines = []
    if len(added_df):
        lines.append("ADDED:")
        for _, r in added_df.head(max_each).iterrows():
            lines.append(f"- NEW[{int(r['new_id'])}]: {str(r['text'])[:400]}")
    if len(removed_df):
        lines.append("\nREMOVED:")
        for _, r in removed_df.head(max_each).iterrows():
            lines.append(f"- OLD[{int(r['old_id'])}]: {str(r['text'])[:400]}")
    if len(modified_df):
        lines.append("\nMODIFIED:")
        for _, r in modified_df.head(max_each).iterrows():
            lines.append(
                f"- OLD[{int(r['old_id'])}] → NEW[{int(r['new_id'])}] (sim={r['sim']}, Δ={r['diff_ratio']}): "
                f"OLD: {str(r['old_text'])[:240]}  ||  NEW: {str(r['new_text'])[:240]}"
            )
    return "\n".join(lines).strip()

# ---------------- UI ----------------
st.set_page_config(page_title="AI Contract Frontend", page_icon="📄", layout="wide")

def _safe_set_option(name, value):
    try:
        st.set_option(name, value)
    except Exception:
        pass  # ignore unknown/deprecated keys

# (Keep for older Streamlit versions; safely ignored on newer)
_safe_set_option("deprecation.showfileUploaderEncoding", False)

st.title("📄 AI-Powered Contract Frontend — Upload • Classify • Chat")

with st.sidebar:
    st.subheader("⚙️ Model Settings")

    # --- Classifier path (auto-pick best export) ---
    _default_clf = find_best_export_dir() or "Clause_classifier/outputs_models/<choose-a-run>/model"
    clf_dir_input = st.text_input(
        "Classifier model directory",
        value=_default_clf,
    )

    # --- Backend selection ---
    backend = st.radio("Chat backend", ["Groq (API)", "Local (HF)"], index=1, horizontal=True)

    # --- Groq settings ---
    if backend == "Groq (API)":
        groq_model = st.text_input("Groq model id", value=DEFAULT_GROQ_MODEL)
    else:
        groq_model = DEFAULT_GROQ_MODEL  # default placeholder

    # --- Local (HF) presets / custom ---
    local_base_model = None
    local_adapter_dir = None
    use_adapter = False

    if backend == "Local (HF)":
        preset_choice = st.selectbox(
            "Choose local LLM",
            list(LLM_PRESETS.keys()),
            index=1
        )
        preset = LLM_PRESETS[preset_choice]
        if preset_choice == "Custom…":
            local_base_model = st.text_input("Base HF model", value="Qwen/Qwen2.5-1.5B-Instruct")
            local_adapter_dir = st.text_input("LoRA adapter directory (optional)", value="")
            use_adapter = bool(local_adapter_dir.strip())
        else:
            local_base_model = preset["base_model"]
            local_adapter_dir = preset["adapter_dir"]
            use_adapter = preset["use_adapter"]
            st.caption(
                f"Preset selected → Base: `{local_base_model}`  "
                f"{'• LoRA: `' + local_adapter_dir + '`' if use_adapter else '• (No LoRA)'}"
            )

    embed_name = st.text_input("Embedder (Sentence-Transformers)", value="intfloat/e5-large-v2")
    top_k = st.slider("Top-K retrieval", 3, 10, 5)
    max_new = st.slider("Max new tokens", 128, 1024, 256, step=64)

    clf_max_len = st.slider("Classifier max length", 128, 512, 384, step=32)
    min_conf = st.slider("Min confidence for label", 0.0, 0.9, 0.30, 0.05)
    st.markdown("---")
    if st.button("🔄 Reset caches (reload models)"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Caches cleared. Click Rerun.")

st.markdown("---")

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Upload Contract")
    up = st.file_uploader("PDF / DOCX / TXT", type=["pdf", "docx", "txt"], accept_multiple_files=False)
    parse_btn = st.button("Parse & Segment")

with right:
    st.subheader("2) Models")
    ready = False
    try:
        with st.spinner("Loading classifier (CPU)…"):
            clf_tok, clf_model, clf_dir, note, *extras = load_classifier(clf_dir_input)
            if len(extras) >= 2:
                id2label_chk, head_norm = extras[0], extras[1]
            else:
                id2label_chk = list(_normalize_id2label(clf_model).items())[:5]
                head_norm = _head_weight_norm(clf_model)
            st.caption(
                f"Classifier dir: {clf_dir}\n\n"
                f"num_labels: {clf_model.config.num_labels} • "
                f"head_l2_norm: {head_norm if head_norm is not None else 'n/a'} • "
                f"labels preview: {id2label_chk}"
            )
            if note:
                st.info(note)
        with st.spinner("Loading embedder…"):
            embedder = load_embedder(embed_name)

        if backend == "Local (HF)":
            with st.spinner("Loading local chat model…"):
                chat_tok, chat_model = load_chat_model(
                    local_base_model,
                    local_adapter_dir if use_adapter and (local_adapter_dir or "").strip() else None
                )
                st.session_state["chat_tok"] = chat_tok
                st.session_state["chat_model"] = chat_model
                if use_adapter and local_adapter_dir:
                    st.caption(f"LoRA adapter enabled: {local_adapter_dir}")

        ready = True
        st.success("Models loaded")
    except Exception as e:
        st.error(f"Failed to load models: {e}")

st.markdown("---")

# ---------------- Main flow ----------------
if up and parse_btn and ready:
    ext = Path(up.name).suffix.lower()
    parser = SUPPORTED_TYPES.get(ext)
    if not parser:
        st.error(f"Unsupported file type: {ext}"); st.stop()
    kind, fn = parser
    with st.spinner(f"Parsing {kind}…"):
        text = fn(up.read())
    if not text:
        st.error("No text extracted. If it's a scanned PDF, add OCR."); st.stop()

    with st.spinner("Segmenting clauses…"):
        clauses = segment_clauses(text)
        if not clauses:
            st.error("No clauses found after segmentation. Try DOCX/TXT or add OCR."); st.stop()
        if len(clauses) == 1:
            st.warning("Only 1 clause found — retrieval will be limited.")

    with st.spinner("Classifying clauses…"):
        clauses = classify_clauses(
            clauses, clf_tok, clf_model,
            max_length=clf_max_len, stride=max(64, clf_max_len // 4),
            min_confidence=min_conf, min_margin=0.05,
            force_label_when_uncertain=False, debug_probs=False,
        )
        df = pd.DataFrame(clauses)
        unknown_rate = (df["category"] == "(Unknown)").mean() if len(df) else 1.0
        if unknown_rate >= 0.7 and not st.session_state.get("_auto_relaxed"):
            st.warning(f"High unknown rate ({unknown_rate:.0%}). Auto-relaxing thresholds for this document.")
            st.session_state["_auto_relaxed"] = True
            clauses = classify_clauses(
                clauses, clf_tok, clf_model,
                max_length=clf_max_len, stride=max(64, clf_max_len // 4),
                min_confidence=max(0.10, float(min_conf) * 0.5), min_margin=0.05,
                force_label_when_uncertain=False, debug_probs=False,
            )
            df = pd.DataFrame(clauses)

    with st.spinner("Indexing uploaded document…"):
        index, embs, meta = build_doc_index(clauses, embedder)

    st.session_state["clauses_df"] = df
    st.session_state["faiss_index"] = index
    st.session_state["faiss_clauses"] = meta
    st.session_state["embedder"] = embedder
    if backend == "Local (HF)":
        st.session_state["chat_tok"] = st.session_state.get("chat_tok")
        st.session_state["chat_model"] = st.session_state.get("chat_model")

    st.success(f"Parsed {len(text):,} chars → {len(meta)} clauses")

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["Contract Analysis", "Policy Studio", "Compare Versions"])

# === Contract Analysis ===
with tab1:
    if "clauses_df" in st.session_state:
        st.subheader("3) Analysis")
        df = st.session_state["clauses_df"]
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Clauses", len(df))
        with c2: st.metric("Categories", df["category"].nunique())
        with c3:
            risk_score = int(np.clip((df["severity"].sum() / max(len(df), 1)) * 10, 0, 100))
            st.metric("Overall Risk (heuristic)", f"{risk_score}/100")
        with c4:
            top_cat = df["category"].value_counts().idxmax()
            st.metric("Most Common Category", top_cat)
        st.bar_chart(df["category"].value_counts(), use_container_width=True)
        st.dataframe(df[["clause_id", "category", "confidence", "severity_label", "text"]],
             use_container_width=True, height=420)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Clauses CSV", csv, file_name="clauses_classified.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("4) Chat over this Contract (RAG)")
        q = st.text_input("Ask a question about the uploaded contract…",
                          value="Summarize the indemnification obligations and cite relevant clauses.",
                          key="qa_input")
        if st.button("Ask", key="qa_btn") and q.strip():
            idx = st.session_state["faiss_index"]; emb = st.session_state["embedder"]
            clauses_meta = st.session_state["faiss_clauses"]
            with st.spinner("Retrieving…"):
                hits = search_doc(idx, emb, q, clauses_meta, k=top_k)
            try:
                if backend == "Groq (API)":
                    with st.spinner("Calling Groq…"):
                        answer = generate_answer_groq(q, hits, groq_model, max_new_tokens=max_new)
                else:
                    chat_tok = st.session_state["chat_tok"]; chat_model = st.session_state["chat_model"]
                    with st.spinner("Generating answer…"):
                        answer = generate_answer_local(chat_tok, chat_model, q, hits, max_new_tokens=max_new)
            except Exception as e:
                answer = f"Generation failed: {e}"
            st.subheader("Answer"); st.write(answer)
            st.markdown("---")
            st.subheader("Retrieved Clauses")
            with st.expander("Show retrieved clauses", expanded=False):
                for h in hits:
                    st.markdown(
                        f"**clause_id {h['clause_id']}** — _{h.get('category','?')}_ "
                        f"(score={h['score']:.3f})\n\n{h['text']}"
                    )
    else:
        st.info("Upload a contract and click ‘Parse & Segment’ to begin.")

# === Policy Studio (simple JSON rule checks) ===
with tab2:
    st.subheader("🛡️ Policy Studio")
    st.caption("Build policies with dropdowns and fields (and/or paste JSON).")

    example = {
        "requires": ["Indemnification", "Termination for Convenience"],
        "forbids": ["Uncapped Liability"],
        "rules": [
            {"category": "Governing Law", "must_include": ["Delaware"], "must_exclude": ["California"]},
        ]
    }
    st.code(json.dumps(example, indent=2), language="json")

    if "clauses_df" in st.session_state and not st.session_state["clauses_df"].empty:
        _cats_list = sorted(list(set(st.session_state["clauses_df"]["category"].dropna().astype(str).tolist())))
    else:
        _cats_list = ["Indemnification", "Termination for Convenience", "Uncapped Liability", "Governing Law"]

    with st.expander("Build policy with controls", expanded=True):
        colb1, colb2 = st.columns(2)
        with colb1:
            req_sel = st.multiselect("Requires (categories must exist)", _cats_list, [])
            forb_sel = st.multiselect("Forbids (categories must NOT exist)", _cats_list, [])
        with colb2:
            st.markdown("**Rule row — add category-specific constraints**")
            rule_cat = st.selectbox("Category", _cats_list, index=0)
            rule_include = st.text_input("Must include (comma-separated keywords)", "")
            rule_exclude = st.text_input("Must exclude (comma-separated keywords)", "")
            add_rule = st.button("➕ Add rule to policy")

        if "policy_rules" not in st.session_state:
            st.session_state["policy_rules"] = []
        if add_rule:
            st.session_state["policy_rules"].append({
                "category": rule_cat,
                "must_include": [x.strip() for x in rule_include.split(",") if x.strip()],
                "must_exclude": [x.strip() for x in rule_exclude.split(",") if x.strip()],
            })

        if st.session_state["policy_rules"]:
            st.write(pd.DataFrame(st.session_state["policy_rules"]))

        built_policy = {
            "requires": req_sel,
            "forbids": forb_sel,
            "rules": st.session_state["policy_rules"],
        }
        st.code(json.dumps(built_policy, indent=2), language="json")

        cpol1, cpol2 = st.columns([2, 1])
        with cpol1:
            rules_text = st.text_area("Policy rules (JSON) — paste here (optional)", height=220, key="policy_text")
        with cpol2:
            up_json = st.file_uploader("…or upload JSON", type=["json"], key="policy_json")
            if up_json is not None:
                try:
                    rules_text = up_json.read().decode("utf-8")
                except Exception:
                    st.error("Could not read uploaded JSON.")

    if st.button("Evaluate Compliance", key="policy_eval"):
        if "clauses_df" not in st.session_state:
            st.error("No contract loaded. Upload and parse a contract first.")
        else:
            try:
                policy = (json.loads(rules_text) if rules_text.strip() else {}) or built_policy
                df = st.session_state["clauses_df"]
                cats = set(df["category"].tolist())

                rows = []
                for req in policy.get("requires", []):
                    rows.append({"rule": "requires", "category": req,
                                 "status": "PASS" if req in cats else "FAIL",
                                 "detail": "" if req in cats else "Missing required clause."})
                for f in policy.get("forbids", []):
                    rows.append({"rule": "forbids", "category": f,
                                 "status": "PASS" if f not in cats else "FAIL",
                                 "detail": "" if f not in cats else "Prohibited clause present."})
                for r in policy.get("rules", []):
                    cat = r.get("category", "")
                    sub = df[df["category"] == cat]
                    if sub.empty:
                        rows.append({"rule": "rules", "category": cat, "status": "FAIL",
                                     "detail": "No matching clause(s) to evaluate."})
                        continue
                    text = " ".join(sub["text"].astype(str).tolist()).lower()
                    must_inc = [w.lower() for w in r.get("must_include", [])]
                    must_exc = [w.lower() for w in r.get("must_exclude", [])]
                    inc_ok = all(w in text for w in must_inc)
                    exc_ok = all(w not in text for w in must_exc)
                    status = "PASS" if (inc_ok and exc_ok) else "FAIL"
                    detail = []
                    if not inc_ok and must_inc: detail.append("Missing required phrases.")
                    if not exc_ok and must_exc: detail.append("Contains excluded phrase(s).")
                    rows.append({"rule": "rules", "category": cat, "status": status,
                                 "detail": " ".join(detail) if detail else ""})

                res_df = pd.DataFrame(rows, columns=["rule", "category", "status", "detail"])
                st.subheader("Results")
                st.dataframe(res_df, use_container_width=True, height=320)
                st.caption(f"Total checks: {len(res_df)} • PASS: {(res_df['status']=='PASS').sum()} • FAIL: {(res_df['status']=='FAIL').sum()}")
            except Exception as e:
                st.error(f"Policy evaluation failed: {e}")

# === Compare Versions (Old vs New) ===========================================
# --- Plain chat completion for non-RAG use (Compare Versions tab) ---
def chat_completion_local(tok, model, system: str, user: str, max_new_tokens: int = 512) -> str:
    """
    Minimal, synchronous local chat helper.
    Uses chat templates when the tokenizer supports them.
    Deterministic first; if the answer looks empty, retries with light sampling.
    """
    # 1) Build prompt
    if supports_chat_template(tok):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        enc = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        enc = tok(
            f"{system}\n\nUSER:\n{user}\n\nASSISTANT:",
            return_tensors="pt", truncation=True, max_length=3800
        )

    # 2) Ensure attention mask and run generation (CPU by design in your loader)
    inputs = _ensure_attention_mask(tok, enc)

    def _decode(out_ids):
        text = tok.decode(out_ids[0], skip_special_tokens=True).strip()
        # Trim any scaffold if we used the simple prompt format
        return text.split("ASSISTANT:", 1)[-1].strip()

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            no_repeat_ngram_size=4,
            repetition_penalty=1.05,
        )
    answer = _decode(out)

    # 3) Gentle fallback with sampling if the deterministic pass looks empty/noisy
    if not answer or re.fullmatch(r"[!?.\s]{8,}", answer) or answer.lower().startswith("system "):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                num_beams=1,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                no_repeat_ngram_size=4,
                repetition_penalty=1.05,
            )
        answer = _decode(out)

    return answer.strip()

with tab3:
    st.subheader("🔍 Compare Contract Versions — Old vs New")
    _ensure_state({
        "cv_old_bytes": None, "cv_new_bytes": None,
        "cv_old_name": "", "cv_new_name": "",
        "cv_old_text": "", "cv_new_text": "",
        "cv_old_clauses": [], "cv_new_clauses": [],
        "cv_added_df": pd.DataFrame(), "cv_removed_df": pd.DataFrame(), "cv_modified_df": pd.DataFrame(),
        "cv_ready": False, "cv_report": ""
    })
    c_old, c_new = st.columns(2)
    with c_old:
        old_up = st.file_uploader("Upload OLD version (PDF/DOCX/TXT)", type=["pdf","docx","txt"], key="cv_old_up")
    with c_new:
        new_up = st.file_uploader("Upload NEW version (PDF/DOCX/TXT)", type=["pdf","docx","txt"], key="cv_new_up")
    if st.button("Parse, Align & Diff", key="cv_parse"):
        try:
            if old_up: st.session_state["cv_old_bytes"], st.session_state["cv_old_name"] = old_up.read(), old_up.name
            if new_up: st.session_state["cv_new_bytes"], st.session_state["cv_new_name"] = new_up.read(), new_up.name
            if not st.session_state["cv_old_bytes"] or not st.session_state["cv_new_bytes"]:
                st.error("Please upload both OLD and NEW versions before diffing.")
                st.session_state["cv_ready"] = False
            else:
                def _parse_any(name, raw):
                    ext = Path(name).suffix.lower()
                    kind, fn = SUPPORTED_TYPES.get(ext, (None, None))
                    if not fn: raise ValueError(f"Unsupported file type for {name}")
                    return fn(raw)
                old_text = _parse_any(st.session_state["cv_old_name"], st.session_state["cv_old_bytes"])
                new_text = _parse_any(st.session_state["cv_new_name"], st.session_state["cv_new_bytes"])
                st.session_state["cv_old_text"] = old_text
                st.session_state["cv_new_text"] = new_text
                old_clauses = segment_clauses(old_text)
                new_clauses = segment_clauses(new_text)
                st.session_state["cv_old_clauses"] = old_clauses
                st.session_state["cv_new_clauses"] = new_clauses
                with st.spinner("Aligning & computing differences…"):
                    embedder = st.session_state.get("embedder") or load_embedder("intfloat/e5-large-v2")
                    added_df, removed_df, modified_df = align_and_diff(old_clauses, new_clauses, embedder, sim_thresh=0.80)
                st.session_state["cv_added_df"]    = added_df
                st.session_state["cv_removed_df"]  = removed_df
                st.session_state["cv_modified_df"] = modified_df
                st.session_state["cv_ready"] = True
                st.success(f"Old: {len(old_clauses)} clauses • New: {len(new_clauses)} clauses")
        except Exception as e:
            st.session_state["cv_ready"] = False
            st.error(f"Diff failed: {e}")

    if st.session_state["cv_ready"]:
        a1, a2, a3 = st.columns(3)
        with a1: st.metric("Added",   int(len(st.session_state["cv_added_df"])))
        with a2: st.metric("Removed", int(len(st.session_state["cv_removed_df"])))
        with a3: st.metric("Modified",int(len(st.session_state["cv_modified_df"])))

        st.subheader("Added (in NEW only)")
        st.dataframe(st.session_state["cv_added_df"], use_container_width=True, height=220)
        if len(st.session_state["cv_added_df"]):
            st.download_button("📄 Added (CSV)",
                               st.session_state["cv_added_df"].to_csv(index=False).encode("utf-8"),
                               "added.csv", "text/csv")

        st.subheader("Removed (present in OLD, missing in NEW)")
        st.dataframe(st.session_state["cv_removed_df"], use_container_width=True, height=260)
        if len(st.session_state["cv_removed_df"]):
            st.download_button("📄 Removed (CSV)",
                               st.session_state["cv_removed_df"].to_csv(index=False).encode("utf-8"),
                               "removed.csv", "text/csv")

        st.subheader("Modified (matched but changed)")
        st.dataframe(st.session_state["cv_modified_df"], use_container_width=True, height=280)
        if len(st.session_state["cv_modified_df"]):
            st.download_button("📄 Modified (CSV)",
                               st.session_state["cv_modified_df"].to_csv(index=False).encode("utf-8"),
                               "modified.csv", "text/csv")

        st.markdown("---")
        st.subheader("LLM Change Report")
        if st.button("Generate Summary Report", key="cv_report_btn"):
            try:
                bullets = _bullets_for_llm(
                    st.session_state["cv_added_df"],
                    st.session_state["cv_removed_df"],
                    st.session_state["cv_modified_df"],
                    max_each=12
                ) or "(No differences detected.)"

                system = (
                    "You are a meticulous contract analyst. Write a concise change report that explains what changed "
                    "between the OLD and NEW versions. Group by: Added, Removed, Modified. "
                    "Highlight materiality and potential risk. Keep to 8–15 bullet points; be specific."
                )
                user = f"Summarize the following clause-level diffs:\n\n{bullets}"

                if backend == "Groq (API)":
                    st.session_state["cv_report"] = groq_chat(system, user, groq_model, max_new_tokens=max_new)
                else:
                    if "chat_tok" not in st.session_state or "chat_model" not in st.session_state:
                        chat_tok, chat_model = load_chat_model(
                            local_base_model,
                            local_adapter_dir if use_adapter and (local_adapter_dir or "").strip() else None
                        )
                        st.session_state["chat_tok"] = chat_tok
                        st.session_state["chat_model"] = chat_model

                    if use_adapter and local_adapter_dir:
                        st.caption(f"LoRA adapter enabled: {local_adapter_dir}")

                    st.session_state["cv_report"] = chat_completion_local(
                        st.session_state["chat_tok"],
                        st.session_state["chat_model"],
                        system,
                        user,
                        max_new_tokens=max_new
                    )
            except Exception as e:
                st.session_state["cv_report"] = f"Report generation failed: {e}"

        if st.session_state["cv_report"]:
            st.success("Change report:")
            st.write(st.session_state["cv_report"])

            report_txt = st.session_state["cv_report"]
            st.download_button(
                "⬇️ Download Change Report (TXT)",
                report_txt.encode("utf-8"),
                file_name="change_report.txt",
                mime="text/plain",
                key="dl_report_txt",
            )
            report_md = f"# LLM Change Report\n\n{report_txt}"
            st.download_button(
                "⬇️ Download Change Report (Markdown)",
                report_md.encode("utf-8"),
                file_name="change_report.md",
                mime="text/markdown",
                key="dl_report_md",
            )
