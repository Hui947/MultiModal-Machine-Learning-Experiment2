#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import fitz  # PyMuPDF
from tqdm import tqdm

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


DEFAULT_PAPERS_DIR = Path("papers") # 整理后的论文路径
DEFAULT_IMAGES_DIR = Path("images") # 图像路径
DEFAULT_EMBED_DIR = Path("embeddings") # 向量库路径
PAPERS_DB_DIR = DEFAULT_EMBED_DIR / "papers_chroma" # 文本向量库
IMAGES_DB_DIR = DEFAULT_EMBED_DIR / "images_chroma" # 图像向量库

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

PDF_MAX_PAGES = 10 # PDF只读取前N页，设为None表示全读


def ensure_dirs():
    """
    确保目录都存在
    """
    DEFAULT_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PAPERS_DB_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DB_DIR.mkdir(parents=True, exist_ok=True)


def safe_topic_name(s: str) -> str:
    """
    防止topic名字中存在非法字符
    """
    s = s.strip()
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    return s[:64] if s else "Unknown"


def parse_topics(topics_str: str) -> List[str]:
    """
    将topic转换成列表
    """
    topics = [t.strip() for t in topics_str.split(",") if t.strip()]
    return topics


def extract_pdf_text(pdf_path: Path, max_pages: Optional[int] = PDF_MAX_PAGES) -> str:
    """
    提取PDF前N页文本
    """
    doc = fitz.open(str(pdf_path))
    texts = []
    n_pages = len(doc)
    limit = n_pages if max_pages is None else min(n_pages, max_pages)
    for i in range(limit):
        page = doc.load_page(i)
        texts.append(page.get_text("text"))
    doc.close()
    text = "\n".join(texts).strip()
    return text


def get_chroma_collection(persist_dir: Path, name: str, metadata: Optional[Dict] = None):
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    if metadata is None:
        return client.get_or_create_collection(name=name)
    return client.get_or_create_collection(name=name, metadata=metadata)


_text_model = None
_clip_model = None
_clip_processor = None

def get_text_model() -> SentenceTransformer:
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(TEXT_MODEL_NAME)
    return _text_model

def get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        _clip_model.eval()
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return _clip_model, _clip_processor


def paper_doc_id(pdf_path: Path) -> str:
    return str(pdf_path.resolve()) # 以绝对路径作为ID

def add_or_update_paper_index(pdf_path: Path, topics_str: str = "") -> Dict:
    """
    读取pdf、编码、存入ChromaDB
    """
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"Not a PDF: {pdf_path}")

    text = extract_pdf_text(pdf_path)
    if not text:
        raise RuntimeError(f"Failed to extract text from: {pdf_path}")

    model = get_text_model()
    emb = model.encode(text, normalize_embeddings=True).tolist()

    papers_col = get_chroma_collection(
        PAPERS_DB_DIR,
        "papers",
        metadata={"hnsw:space": "cosine"},
    )
    doc_id = paper_doc_id(pdf_path)

    metadata = {
        "path": str(pdf_path),
        "topics": topics_str,
        "pages_used": PDF_MAX_PAGES if PDF_MAX_PAGES is not None else -1,
    }

    try:
        papers_col.delete(ids=[doc_id])
    except Exception:
        pass

    papers_col.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[emb],
        metadatas=[metadata],
    )
    return metadata

def classify_paper_to_topic(pdf_path: Path, topics: List[str]) -> str:
    """
    对paper进行topic分类（通过求文本嵌入相似度的方式）
    """
    model = get_text_model()
    text = extract_pdf_text(pdf_path)
    paper_emb = model.encode(text, normalize_embeddings=True)

    topic_embs = model.encode(topics, normalize_embeddings=True)
    best_idx = int((topic_embs @ paper_emb).argmax())
    return topics[best_idx]

def move_paper_to_topic_folder(pdf_path: Path, topic: str) -> Path:
    """
    将pdf放到相应topic的文件夹中，如果存在同名，则在尾部追加“_1、_2、_3、……”
    """
    topic_dir = DEFAULT_PAPERS_DIR / safe_topic_name(topic)
    topic_dir.mkdir(parents=True, exist_ok=True)
    target = topic_dir / pdf_path.name
    
    if target.exists():
        stem = target.stem
        suffix = target.suffix
        i = 1
        while True:
            cand = topic_dir / f"{stem}_{i}{suffix}"
            if not cand.exists():
                target = cand
                break
            i += 1
    shutil.move(str(pdf_path), str(target))
    return target

def cmd_add_paper(args):
    """
    添加并分类某篇论文
    """
    ensure_dirs()
    src = Path(args.path)
    if not src.exists():
        raise FileNotFoundError(src)

    topics = parse_topics(args.topics) if args.topics else []
    
    if topics:
        best_topic = classify_paper_to_topic(src, topics)
        dst = move_paper_to_topic_folder(src, best_topic)
        topics_str = best_topic
        print(f"[OK] Classified topic: {best_topic}")
        print(f"[OK] Moved to: {dst}")
        meta = add_or_update_paper_index(dst, topics_str=topics_str)
    else:
        DEFAULT_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        dst = DEFAULT_PAPERS_DIR / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(str(src), str(dst))
        meta = add_or_update_paper_index(dst, topics_str="")
        print(f"[OK] Added & indexed: {dst}")

    print(f"[META] {meta}")

def cmd_organize_papers(args):
    """
    批量整理一个混乱的文件夹中的所有论文
    """
    ensure_dirs()
    topics = parse_topics(args.topics)
    if not topics:
        raise ValueError("organize_papers requires --topics like 'CV,NLP,RL'.")

    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(root_dir)

    pdfs = sorted([p for p in root_dir.rglob("*.pdf") if p.is_file()])
    if not pdfs:
        print("[INFO] No PDFs found.")
        return

    print(f"[INFO] Found {len(pdfs)} PDFs. Organizing into papers/<topic>/ ...")
    for pdf in tqdm(pdfs, desc="Organizing PDFs"):
        try:
            best_topic = classify_paper_to_topic(pdf, topics)
            dst = move_paper_to_topic_folder(pdf, best_topic)
            add_or_update_paper_index(dst, topics_str=best_topic)
        except Exception as e:
            print(f"[WARN] Skip {pdf.name}: {e}")

    print("[OK] Batch organize done.")

def cmd_search_paper(args):
    """
    对论文库做语义检索
    """
    ensure_dirs()
    papers_col = get_chroma_collection(
        PAPERS_DB_DIR,
        "papers",
        metadata={"hnsw:space": "cosine"},
    )

    model = get_text_model()
    q_emb = model.encode(args.query, normalize_embeddings=True).tolist()

    res = papers_col.query(
        query_embeddings=[q_emb],
        n_results=args.top_k,
        include=["metadatas", "distances"],
    )

    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    if not metas:
        print("[INFO] No results. Have you indexed any PDFs via add_paper / organize_papers?")
        return

    print(f"[QUERY] {args.query}")
    for i, (m, d) in enumerate(zip(metas, dists), start=1):
        score = 1.0 - float(d)
        print(f"{i:02d}. score≈{score:.4f}  path={m.get('path')}  topic={m.get('topics','')}")


# =========================
# 图像：索引 / 以文搜图
# =========================
def image_doc_id(img_path: Path) -> str:
    return str(img_path.resolve())

def embed_image(img_path: Path) -> List[float]:
    """
    对图像编码（用CLIP）
    """
    model, processor = get_clip()
    device = next(model.parameters()).device

    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        feat = model.get_image_features(**inputs)  # [1, D]
        feat = feat / (feat.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    return feat[0].detach().cpu().tolist()

def embed_text_for_clip(text: str) -> List[float]:
    """
    对文本编码（用CLIP）
    """
    model, processor = get_clip()
    device = next(model.parameters()).device

    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        feat = model.get_text_features(**inputs)  # [1, D]
        feat = feat / (feat.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    return feat[0].detach().cpu().tolist()

def index_images_if_needed(images_dir: Path):
    """
    每次 search_image 之前扫描 images_dir，把未入库的图片补充索引。
    """
    images_col = get_chroma_collection(
        IMAGES_DB_DIR,
        "images",
        metadata={"hnsw:space": "cosine"},
    )
    
    try:
        existing = set(images_col.get(include=[]).get("ids", []))
    except Exception:
        existing = set()

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    # imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    imgs = []
    for p in images_dir.rglob("*"):
        if any(part.startswith(".") for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in exts:
            imgs.append(p)

    new_imgs = [p for p in imgs if image_doc_id(p) not in existing]
    if not new_imgs:
        return 0

    for p in tqdm(new_imgs, desc="Indexing images"):
        try:
            emb = embed_image(p)
            doc_id = image_doc_id(p)
            images_col.add(
                ids=[doc_id],
                embeddings=[emb],
                metadatas=[{"path": str(p.resolve())}],
            )
        except Exception as e:
            print(f"[WARN] Failed indexing {p.name}: {e}")

    return len(new_imgs)

def cmd_search_image(args):
    """
    对图片库做语义检索
    """
    ensure_dirs()
    DEFAULT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    added = index_images_if_needed(DEFAULT_IMAGES_DIR)
    if added:
        print(f"[INFO] Indexed {added} new images.")

    images_col = get_chroma_collection(
        IMAGES_DB_DIR,
        "images",
        metadata={"hnsw:space": "cosine"},
    )
    q_emb = embed_text_for_clip(args.query)

    res = images_col.query(
        query_embeddings=[q_emb],
        n_results=args.top_k,
        include=["metadatas", "distances"],
    )

    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    if not metas:
        print("[INFO] No results. Put some images under ./images/ first.")
        return

    print(f"[QUERY] {args.query}")
    for i, (m, d) in enumerate(zip(metas, dists), start=1):
        score = 1.0 - float(d)
        print(f"{i:02d}. score≈{score:.4f}  path={m.get('path')}")


        
def build_parser():
    parser = argparse.ArgumentParser(
        description="Local Multimodal AI Agent (papers semantic search + auto organize + text-to-image search)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add_paper", help="Add a paper PDF, classify into topics, and index it.")
    p_add.add_argument("path", type=str, help="Path to a PDF file")
    p_add.add_argument("--topics", type=str, default="", help='Comma-separated topics, e.g. "CV,NLP,RL"')
    p_add.set_defaults(func=cmd_add_paper)

    p_org = sub.add_parser("organize_papers", help="Batch organize a messy folder of PDFs into papers/<topic>/ and index.")
    p_org.add_argument("--root-dir", type=str, required=True, help="Root dir to scan PDFs")
    p_org.add_argument("--topics", type=str, required=True, help='Comma-separated topics, e.g. "CV,NLP,RL"')
    p_org.set_defaults(func=cmd_organize_papers)

    p_search_p = sub.add_parser("search_paper", help="Semantic search papers by natural language query.")
    p_search_p.add_argument("query", type=str, help="Natural language query")
    p_search_p.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_search_p.set_defaults(func=cmd_search_paper)

    p_search_i = sub.add_parser("search_image", help="Text-to-image search over local images/")
    p_search_i.add_argument("query", type=str, help="Natural language description")
    p_search_i.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_search_i.set_defaults(func=cmd_search_image)

    return parser

def main():
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
