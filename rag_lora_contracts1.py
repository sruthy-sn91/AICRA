#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG + LoRA over CUAD-style QA datasets

This script supports:
1) Preparing SFT data from CUAD-style JSON (SQuAD-like) files
2) Training a LoRA adapter for a causal LM
3) Building a FAISS index from .txt contracts
4) Launching a Gradio RAG + LoRA chatbot

Quickstart (examples):

# 0) Install deps
pip install -U "transformers>=4.42.0" "peft>=0.11.1" "datasets>=2.20.0" \
    "accelerate>=0.33.0" "bitsandbytes>=0.43.1" "sentence-transformers>=3.0.1" \
    "faiss-cpu>=1.8.0" "gradio>=4.43.0" "tqdm>=4.66.4"

# 1) Prepare SFT data (uses your three JSON files; non-empty answers only)
python rag_lora_contracts1.py make_sft \
  --cuad_json data2/CUADv1.json \
  --train_json data2/train_separate_questions.json \
  --test_json data2/test.json \
  --out_jsonl ./data/cuad_sft.jsonl

# 2) Train LoRA (QLoRA if bitsandbytes available; fallback to full-precision CPU)
python rag_lora_contracts1.py train \
  --train_file ./data/cuad_sft.jsonl \
  --base_model "Qwen/Qwen2-7B-Instruct" \
  --out_dir ./lora_out_qwen2_7b

# 3) Build index from .txt contracts
python rag_lora_contracts1.py ingest \
  --data_dir "C:/path/to/contracts_txt" \
  --index_dir ./index

# 4) Chat (RAG + LoRA)
python rag_lora_contracts1.py chat \
  --index_dir ./index \
  --base_model "Qwen/Qwen2-7B-Instruct" \
  --adapter_dir ./lora_out_qwen2_7b

Notes
- If GPU constrained, try a smaller base model like "microsoft/Phi-3-mini-4k-instruct" or "Qwen/Qwen2.5-1.5B-Instruct".
- For Macs with Apple Silicon, MPS is auto-detected for training/inference.
- SFT generator keeps questions with at least one answer. If multiple, uses the first answer text.
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.90")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.50")

import re
import io
import json
import glob
import time
import math
import argparse
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ----------------------
# Lazy imports (lighter startup)
# ----------------------
def lazy_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

def lazy_faiss():
    import faiss
    return faiss

def lazy_transformers():
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    return transformers, AutoTokenizer, AutoModelForCausalLM

def lazy_peft():
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
    return LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

def lazy_bitsandbytes():
    import bitsandbytes as bnb  # noqa
    return bnb

def lazy_gradio():
    import gradio as gr
    return gr

# ----------------------
# Utils
# ----------------------
def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def read_text_file(path: str) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            continue
    # last resort
    with open(path, "r", errors="ignore") as f:
        return f.read()

def iter_txt_files(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.txt"), recursive=True))

def chunk_text(text: str, chunk_chars: int = 1500, overlap: int = 200) -> List[str]:
    text = clean_text(text)
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

# ----------------------
# CUAD-style JSON → SFT JSONL
# ----------------------
PROMPT_TEMPLATE = """You are a contracts analyst. Answer the question based ONLY on the provided contract excerpt. Keep the answer concise (3-6 sentences) and quote exact phrases where useful.

### Question
{question}

### Context
{context}

### Answer
"""

def _extract_squad_like(path: str) -> List[Dict[str, Any]]:
    """
    Accepts standard SQuAD/CUAD-like JSON structures:
    {
      "data": [
        {
          "title": "...",
          "paragraphs": [
            {
              "context": "...",
              "qas": [
                {
                  "id": "...",
                  "question": "...",
                  "is_impossible": false,
                  "answers": [{"text": "...", "answer_start": 123}, ...]
                }, ...
              ]
            }, ...
          ]
        }, ...
      ]
    }
    Some variants may be a flat list or directly contain "data" level 'paragraphs'.
    Returns normalized list of QA dicts: {id, question, context, answers(list[str])}
    """
    raw = json.loads(read_text_file(path))

    def norm_entry(q: Dict[str, Any], context: str) -> Optional[Dict[str, Any]]:
        # Skip impossible or no-answers
        answers = q.get("answers") or []
        answers = [a.get("text", "").strip() for a in answers if a.get("text")]
        answers = [a for a in answers if a]
        if not answers:
            # if marked impossible or genuinely no answer text, skip from SFT
            return None
        return {
            "id": q.get("id", ""),
            "question": q.get("question", "").strip(),
            "context": clean_text(context),
            "answers": answers
        }

    out: List[Dict[str, Any]] = []

    # Case 1: canonical {"data":[...]}
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        for di in raw["data"]:
            paras = di.get("paragraphs", [])
            for p in paras:
                context = p.get("context", "")
                for q in p.get("qas", []):
                    r = norm_entry(q, context)
                    if r:
                        out.append(r)
        if out:
            return out

    # Case 2: directly a list of paragraph containers
    if isinstance(raw, list):
        for di in raw:
            paras = di.get("paragraphs", [])
            for p in paras:
                context = p.get("context", "")
                for q in p.get("qas", []):
                    r = norm_entry(q, context)
                    if r:
                        out.append(r)
        if out:
            return out

    # Case 3: single object with "paragraphs"
    if isinstance(raw, dict) and "paragraphs" in raw:
        for p in raw.get("paragraphs", []):
            context = p.get("context", "")
            for q in p.get("qas", []):
                r = norm_entry(q, context)
                if r:
                    out.append(r)
        if out:
            return out

    # Case 4: a pre-flattened list of qas (with context alongside)
    if isinstance(raw, list):
        for q in raw:
            context = q.get("context", "")
            r = _ = None
            if context and "question" in q and "answers" in q:
                answers = q.get("answers") or []
                answers = [a.get("text", "").strip() for a in answers if a.get("text")]
                answers = [a for a in answers if a]
                if answers:
                    out.append({
                        "id": q.get("id", ""),
                        "question": q.get("question", "").strip(),
                        "context": clean_text(context),
                        "answers": answers,
                    })
        if out:
            return out

    return out

def make_sft_from_files(files: List[str], out_jsonl: str, seed: int = 42, max_per_context: Optional[int] = None) -> int:
    """
    Build instruction-tuning JSONL with fields:
      instruction, input, output, meta
    from the union of given CUAD-like JSONs.
    """
    rng = random.Random(seed)
    all_qas: List[Dict[str, Any]] = []
    for fp in files:
        if not fp:
            continue
        if not os.path.exists(fp):
            print(f"[WARN] File not found: {fp}")
            continue
        qs = _extract_squad_like(fp)
        print(f"[{timestamp()}] Loaded {len(qs)} QA from {fp}")
        all_qas.extend(qs)

    # Optional: limit per unique (context) to avoid overfitting to very long contexts
    if max_per_context is not None:
        bucket: Dict[str, List[Dict[str, Any]]] = {}
        for r in all_qas:
            bucket.setdefault(r["context"], []).append(r)
        balanced: List[Dict[str, Any]] = []
        for ctx, qlist in bucket.items():
            rng.shuffle(qlist)
            balanced.extend(qlist[:max_per_context])
        all_qas = balanced

    rng.shuffle(all_qas)
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in all_qas:
            question = r["question"]
            context = r["context"]
            answer = r["answers"][0].strip()  # first answer
            prompt = PROMPT_TEMPLATE.format(question=question, context=context)
            rec = {
                "instruction": question,
                "input": context,
                "output": answer,
                "prompt": prompt,  # convenience (not required by trainer)
                "meta": {"id": r.get("id", ""), "source": "CUAD-like"}
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[{timestamp()}] Wrote {n} SFT examples to {out_jsonl}")
    return n

# ----------------------
# Simple JSONL Dataset for Causal LM SFT
# ----------------------
@dataclass
class SimpleJsonlDataset:
    path: str
    tokenizer: Any
    max_length: int = 768

    def __post_init__(self):
        self.records: List[str] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                prompt = j.get("prompt")
                if not prompt:
                    # fallback to template
                    question = j.get("instruction", "")
                    context = j.get("input", "")
                    prompt = PROMPT_TEMPLATE.format(question=question, context=context)
                out = j.get("output", "")
                text = (prompt.strip() + "\n" + out.strip()).strip()
                self.records.append(text)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        toks = self.tokenizer(
            self.records[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        sample = {k: v.squeeze(0) for k, v in toks.items()}
        sample["labels"] = sample["input_ids"].clone()
        return sample

# ----------------------
# LoRA training
# ----------------------
# --- REPLACE your existing train_lora with this version ---
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

def train_lora(
    train_file: str,
    base_model: str,
    out_dir: str,
    lr: float = 2e-4,
    epochs: int = 2,
    per_device_batch: int = 1,
    grad_accum: int = 16,
    max_length: int = 768,
    cpu: bool = False,
):
    transformers, AutoTokenizer, AutoModelForCausalLM = lazy_transformers()
    LoraConfig, get_peft_model, _, prepare_model_for_kbit_training = lazy_peft()

    # ---- Device flags
    is_cuda = (not cpu) and torch.cuda.is_available()
    is_mps = (not cpu) and torch.backends.mps.is_available()

    # dtype for MPS: keep model in float16, BUT do NOT enable Trainer fp16/bf16
    model_dtype = torch.float16 if is_mps else None

    # ---- Tokenizer
    print(f"[{timestamp()}] Loading tokenizer: {base_model}")
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- Base model (+ optional 4-bit only on CUDA)
    model_kwargs = {}
    used_4bit = False
    if is_cuda:
        try:
            from transformers import BitsAndBytesConfig
            lazy_bitsandbytes()
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["device_map"] = "auto"
            used_4bit = True
            print("[train] Using CUDA with 4-bit QLoRA.")
        except Exception as e:
            print("[train] bitsandbytes not available; using full precision on CUDA.", e)

    print(f"[{timestamp()}] Loading base model (4bit={used_4bit}, mps={is_mps})")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=None if used_4bit else model_dtype,  # float16 for MPS model weights
        **model_kwargs
    )

    # ---- Prep for k-bit training if quantized (CUDA only)
    if used_4bit:
        model = prepare_model_for_kbit_training(model)

    # ---- LoRA config (broader targets generally help Qwen/Mistral)
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.config.use_cache = False

    # Enable grad checkpointing where it’s supported
    if (is_cuda or is_mps) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    model.print_trainable_parameters()

    # ---- Dataset
    ds = SimpleJsonlDataset(train_file, tok, max_length=max_length)
    os.makedirs(out_dir, exist_ok=True)

    # Mixed-precision flags for Trainer:
    # - CUDA: use bf16 if available (Ampere+), else fp16
    # - MPS: both fp16 and bf16 MUST be False (Accelerate restriction)
    use_bf16 = False
    use_fp16 = False
    if is_cuda:
        # Try bf16 on modern GPUs; fall back to fp16
        try:
            major_cc, _ = torch.cuda.get_device_capability(0)
            use_bf16 = major_cc >= 8  # Ampere or newer
        except Exception:
            use_bf16 = False
        use_fp16 = not use_bf16
    # if is_mps: leave both False

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=20,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        # IMPORTANT: do NOT enable fp16/bf16 on MPS; Accelerate will crash.
        bf16=use_bf16,
        fp16=use_fp16,
    )

    collator = DataCollatorForLanguageModeling(tok, mlm=False, pad_to_multiple_of=8)

    print(f"[{timestamp()}] Starting LoRA training on {len(ds)} examples…")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator
    )
    trainer.train()

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[{timestamp()}] Saved LoRA adapter to: {out_dir}")


# ----------------------
# FAISS index + search
# ----------------------
def build_faiss_index(
    data_dir: str,
    index_dir: str,
    embed_model_name: str = "intfloat/e5-large-v2",
    chunk_chars: int = 1500,
    overlap: int = 200
):
    os.makedirs(index_dir, exist_ok=True)
    files = iter_txt_files(data_dir)
    assert files, f"No .txt files under {data_dir}"
    print(f"[{timestamp()}] Using {len(files)} files from {data_dir}")

    SentenceTransformer = lazy_sentence_transformer()
    embedder = SentenceTransformer(embed_model_name)

    all_chunks, metas = [], []
    for fp in files:
        raw = read_text_file(fp)
        for i, ch in enumerate(chunk_text(raw, chunk_chars=chunk_chars, overlap=overlap)):
            all_chunks.append(ch)
            metas.append({"path": fp, "chunk_id": i})

    print(f"[{timestamp()}] Embedding {len(all_chunks)} chunks…")
    embs = embedder.encode(all_chunks, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    embs = np.asarray(embs, dtype="float32")

    faiss = lazy_faiss()
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))

    with open(os.path.join(index_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for ch, md in zip(all_chunks, metas):
            rec = dict(md)
            rec["text"] = ch
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "embed_model": embed_model_name,
        "chunk_chars": chunk_chars,
        "overlap": overlap,
        "created_at": timestamp(),
        "num_chunks": len(all_chunks),
    }
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[{timestamp()}] FAISS index built at {index_dir} (chunks={len(all_chunks)})")

def load_index(index_dir: str):
    faiss = lazy_faiss()
    SentenceTransformer = lazy_sentence_transformer()

    with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    embedder = SentenceTransformer(meta["embed_model"])

    idx = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    chunks = []
    with open(os.path.join(index_dir, "chunks.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return idx, embedder, chunks, meta

def search(index_dir: str, query: str, k: int = 6) -> List[Dict[str, Any]]:
    idx, embedder, chunks, _ = load_index(index_dir)
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    scores, ids = idx.search(q_emb, k)
    hits = []
    for sc, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        hits.append({
            "score": float(sc),
            "text": chunks[i]["text"],
            "path": chunks[i]["path"],
            "chunk_id": chunks[i]["chunk_id"],
        })
    return hits

# ----------------------
# Inference (RAG + LoRA)
# ----------------------
SYSTEM_PROMPT = """You are a helpful contract analyst. Answer ONLY from the provided context. If the answer is not in the context, say you don't know. Where possible, quote exact phrases and cite sources with (File, Chunk #). Keep it precise and neutral."""

def build_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    citations = []
    ctx_blobs = []
    for i, h in enumerate(contexts):
        fn = os.path.basename(h["path"])
        citations.append(f"({fn}, chunk {h['chunk_id']})")
        ctx_blobs.append(f"[{i+1}] {h['text']}\nSOURCE: {fn}#chunk{h['chunk_id']}")
    joined = "\n\n".join(ctx_blobs)
    prompt = f"""{SYSTEM_PROMPT}

### Question
{query}

### Retrieved Context
{joined}

### Answer (with citations like {', '.join(citations)}):
"""
    return prompt

def generate_answer(
    base_model: str,
    adapter_dir: Optional[str],
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
    cpu: bool = False,
) -> str:
    transformers, AutoTokenizer, AutoModelForCausalLM = lazy_transformers()
    _, _, PeftModel, _ = lazy_peft()

    model_kwargs = {}
    if not cpu:
        try:
            from transformers import BitsAndBytesConfig
            lazy_bitsandbytes()
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["device_map"] = "auto"
        except Exception:
            pass

    tok = AutoTokenizer.from_pretrained(adapter_dir or base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, **model_kwargs)
    if adapter_dir and os.path.isdir(adapter_dir):
        model = PeftModel.from_pretrained(model, adapter_dir)

    inputs = tok(prompt, return_tensors="pt")
    if not cpu:
        try:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        except Exception:
            pass

    gen_cfg = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_cfg)
    text = tok.decode(out_ids[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

# ----------------------
# Gradio UI
# ----------------------
def launch_gradio(index_dir: str, base_model: str, adapter_dir: Optional[str], cpu: bool = False):
    gr = lazy_gradio()
    _, _, _, _ = load_index(index_dir)  # sanity check

    device_info = "cpu"
    if not cpu:
        if torch.backends.mps.is_available():
            device_info = "mps"
        elif torch.cuda.is_available():
            device_info = torch.cuda.get_device_name(0)

    with gr.Blocks(title="Contract RAG + LoRA Chatbot") as demo:
        gr.Markdown("# 📄 Contract RAG + LoRA Chatbot\nAsk questions about your contracts. The model answers **only** from retrieved context and cites sources.")

        with gr.Row():
            q = gr.Textbox(label="Your question", placeholder="e.g., What is the governing law?")
        with gr.Row():
            topk = gr.Slider(2, 12, value=6, step=1, label="Top-k passages")
            temp = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
        out = gr.Markdown()
        src = gr.Dataframe(headers=["Score","File","Chunk","Preview"], datatype=["number","str","number","str"], interactive=False, wrap=True)

        state = gr.State({"index_dir": index_dir, "base_model": base_model, "adapter_dir": adapter_dir, "cpu": cpu})

        def _answer(query, k, temperature, st):
            hits = search(st["index_dir"], query, k=int(k))
            prompt = build_prompt(query, hits)
            ans = generate_answer(
                base_model=st["base_model"],
                adapter_dir=st["adapter_dir"],
                prompt=prompt,
                temperature=float(temperature),
                cpu=bool(st["cpu"]),
            )
            rows = []
            for h in hits:
                rows.append([
                    round(h["score"],4),
                    os.path.basename(h["path"]),
                    h["chunk_id"],
                    (h["text"][:220] + "…") if len(h["text"])>220 else h["text"]
                ])
            return ans, rows

        q.submit(_answer, inputs=[q, topk, temp, state], outputs=[out, src])
        gr.Markdown(f"**Device**: {device_info} | **Base model**: `{base_model}` | **Adapter**: `{adapter_dir or 'None'}`")

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

# ----------------------
# CLI
# ----------------------
def main():
    parser = argparse.ArgumentParser(description="RAG + LoRA with CUAD-style datasets")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # SFT maker
    p_sft = sub.add_parser("make_sft", help="Create SFT JSONL from CUAD-style JSONs")
    p_sft.add_argument("--cuad_json", type=str, default=None, help="e.g., data2/CUADv1.json")
    p_sft.add_argument("--train_json", type=str, default=None, help="e.g., data2/train_separate_questions.json")
    p_sft.add_argument("--test_json", type=str, default=None, help="e.g., data2/test.json")
    p_sft.add_argument("--out_jsonl", type=str, required=True, help="Where to save SFT JSONL")
    p_sft.add_argument("--max_per_context", type=int, default=None, help="Optional cap per context")

    # Train
    p_tr = sub.add_parser("train", help="Train LoRA adapter on SFT JSONL")
    p_tr.add_argument("--train_file", type=str, required=True)
    p_tr.add_argument("--base_model", type=str, default="Qwen/Qwen2-7B-Instruct")
    p_tr.add_argument("--out_dir", type=str, required=True)
    p_tr.add_argument("--lr", type=float, default=2e-4)
    p_tr.add_argument("--epochs", type=int, default=2)
    p_tr.add_argument("--per_device_batch", type=int, default=1)
    p_tr.add_argument("--grad_accum", type=int, default=16)
    p_tr.add_argument("--max_length", type=int, default=768)
    p_tr.add_argument("--cpu", action="store_true")

    # Ingest contracts
    p_ing = sub.add_parser("ingest", help="Build FAISS index from .txt contracts")
    p_ing.add_argument("--data_dir", type=str, required=True)
    p_ing.add_argument("--index_dir", type=str, required=True)
    p_ing.add_argument("--embed_model", type=str, default="intfloat/e5-large-v2")
    p_ing.add_argument("--chunk_chars", type=int, default=1500)
    p_ing.add_argument("--overlap", type=int, default=200)

    # Chat
    p_chat = sub.add_parser("chat", help="Launch Gradio RAG+LoRA chat")
    p_chat.add_argument("--index_dir", type=str, required=True)
    p_chat.add_argument("--base_model", type=str, default="Qwen/Qwen2-7B-Instruct")
    p_chat.add_argument("--adapter_dir", type=str, default=None)
    p_chat.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    if args.cmd == "make_sft":
        files = [args.cuad_json, args.train_json, args.test_json]
        n = make_sft_from_files(files, args.out_jsonl, max_per_context=args.max_per_context)
        print(f"[{timestamp()}] Total SFT examples: {n}")

    elif args.cmd == "train":
        train_lora(
            train_file=args.train_file,
            base_model=args.base_model,
            out_dir=args.out_dir,
            lr=args.lr,
            epochs=args.epochs,
            per_device_batch=args.per_device_batch,
            grad_accum=args.grad_accum,
            max_length=args.max_length,
            cpu=args.cpu,
        )

    elif args.cmd == "ingest":
        build_faiss_index(
            data_dir=args.data_dir,
            index_dir=args.index_dir,
            embed_model_name=args.embed_model,
            chunk_chars=args.chunk_chars,
            overlap=args.overlap
        )

    elif args.cmd == "chat":
        launch_gradio(
            index_dir=args.index_dir,
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            cpu=args.cpu
        )

if __name__ == "__main__":
    main()
