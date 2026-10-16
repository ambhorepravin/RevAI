import os
import re
import gc
from typing import List, Dict, Any
from urllib.parse import urlparse, parse_qs

# Disable Hugging Face symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.cluster import AgglomerativeClustering

# Limit CPU threads to prevent RAM spikes on Windows
torch.set_num_threads(2)

app = FastAPI(title="RevAI - Lecture2Revision Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. HYBRID ARCHITECTURE: TRANSFORMER + LSTM MODULE
# ---------------------------------------------------------
class LectureSequenceLSTM(nn.Module):
    """
    Processes sequential Transformer sentence embeddings from multiple lectures
    using a Bidirectional LSTM to score sentence salience and temporal flow.
    """
    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 128):
        super(LectureSequenceLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embeddings: torch.Tensor):
        lstm_out, _ = self.lstm(embeddings)
        scores = self.sigmoid(self.attention_weights(lstm_out)).squeeze(-1)
        return lstm_out.squeeze(0), scores.squeeze(0)


print("Loading lightweight Transformer & LSTM models (Low-RAM Mode)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

SUMMARIZER_MODEL = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
seq2seq_model.eval()

lstm_scorer = LectureSequenceLSTM(embedding_dim=384, hidden_dim=128)
lstm_scorer.eval()

gc.collect()
print("Models loaded successfully! Server starting at http://127.0.0.1:8000")


# ---------------------------------------------------------
# 2. DATA SCHEMAS & HELPER FUNCTIONS
# ---------------------------------------------------------
class LectureRequest(BaseModel):
    topic: str
    video_inputs: List[str]


def generate_summary(text: str) -> str:
    """Generates an abstractive summary using the lightweight Transformer."""
    prompt = f"Summarize the following lecture notes clearly: {text[:1500]}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = seq2seq_model.generate(
            inputs.input_ids,
            max_length=100,
            min_length=25,
            num_beams=2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/").split("?")[0]
    if parsed.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith(("/embed/", "/v/", "/shorts/", "/live/")):
            return parsed.path.split("/")[2]
    return ""


def fetch_transcript_or_text(item: str, idx: int) -> str:
    item = item.strip()
    video_id = extract_video_id(item)
    if video_id:
        try:
            ytt_api = YouTubeTranscriptApi()
            try:
                # Try standard and regional English transcripts first
                fetched = ytt_api.fetch(video_id, languages=["en", "en-US", "en-GB", "en-IN"])
            except Exception:
                # Fallback: pick any available transcript and translate to English if possible
                transcript_list = ytt_api.list(video_id)
                first_transcript = next(iter(transcript_list))
                if first_transcript.is_translatable and not first_transcript.language_code.startswith("en"):
                    fetched = first_transcript.translate("en").fetch()
                else:
                    fetched = first_transcript.fetch()

            # Support both v1.2+ FetchedTranscript objects and legacy dict lists
            if hasattr(fetched, "to_raw_data"):
                raw_data = fetched.to_raw_data()
                text = " ".join([entry["text"] for entry in raw_data])
            else:
                text = " ".join([
                    entry.text if hasattr(entry, "text") else entry["text"]
                    for entry in fetched
                ])

            return re.sub(r"\s+", " ", text).strip()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not fetch YouTube transcript for video #{idx + 1} (ID: {video_id}). Error: {str(e)}"
            )
    else:
        return re.sub(r"\s+", " ", item)


def split_into_sentences(text: str) -> List[str]:
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 5]
    if not sentences and text.strip():
        words = text.split()
        sentences = [" ".join(words[i:i + 18]) + "." for i in range(0, len(words), 18)]
    return sentences


def combine_and_analyze_lectures(topic: str, raw_texts: List[str]) -> Dict[str, Any]:
    all_sentences = []
    video_sources = []

    for v_idx, text in enumerate(raw_texts):
        sents = split_into_sentences(text)
        for s in sents:
            all_sentences.append(s)
            video_sources.append(f"Video {v_idx + 1}")

    if not all_sentences:
        raise HTTPException(status_code=400, detail="No valid content found in the provided inputs.")

    # Step 1: Transformer Embeddings
    embeddings_np = embedder.encode(all_sentences, convert_to_numpy=True)
    embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32).unsqueeze(0)

    # Step 2: BiLSTM Sequential Importance & Context Encoding
    with torch.no_grad():
        _, lstm_scores = lstm_scorer(embeddings_tensor)
        lstm_scores_np = lstm_scores.numpy()
        if lstm_scores_np.ndim == 0:
            lstm_scores_np = np.array([float(lstm_scores_np)])

    # Boost score by cosine similarity with the target topic
    topic_emb = embedder.encode([topic], convert_to_numpy=True)[0]
    topic_sim = np.dot(embeddings_np, topic_emb) / (
        np.linalg.norm(embeddings_np, axis=1) * np.linalg.norm(topic_emb) + 1e-8
    )
    combined_scores = 0.5 * lstm_scores_np + 0.5 * topic_sim

    # Step 3: Multi-Video Semantic Clustering
    n_clusters = min(max(2, len(all_sentences) // 4), 5)
    if len(all_sentences) >= n_clusters:
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clustering.fit_predict(embeddings_np)
    else:
        cluster_labels = np.zeros(len(all_sentences), dtype=int)
        n_clusters = 1

    notes_sections = []
    flashcards = []

    for c_id in range(n_clusters):
        indices = np.where(cluster_labels == c_id)[0]
        if len(indices) == 0:
            continue

        ranked_idx = sorted(indices, key=lambda idx: combined_scores[idx], reverse=True)
        top_idx = sorted(ranked_idx[:6])

        cluster_text = " ".join([all_sentences[i] for i in top_idx])
        contributing_videos = sorted(list(set([video_sources[i] for i in indices])))

        if len(cluster_text.split()) > 25:
            summary_text = generate_summary(cluster_text)
        else:
            summary_text = cluster_text

        key_points = [all_sentences[i] for i in ranked_idx[:3]]

        notes_sections.append({
            "module_title": f"{topic} — Core Concept {c_id + 1}",
            "sources": contributing_videos,
            "summary": summary_text,
            "key_points": key_points
        })

        for rank_pos, s_idx in enumerate(ranked_idx[:2]):
            sent = all_sentences[s_idx]
            question = generate_question_from_sentence(sent, topic, c_id + 1, rank_pos + 1)
            flashcards.append({
                "id": len(flashcards) + 1,
                "source": video_sources[s_idx],
                "difficulty": "High Yield" if rank_pos == 0 else "Conceptual",
                "question": question,
                "answer": sent
            })

    gc.collect()
    return {
        "topic": topic,
        "video_count": len(raw_texts),
        "total_sentences_analyzed": len(all_sentences),
        "notes": notes_sections,
        "flashcards": flashcards
    }


def generate_question_from_sentence(sentence: str, topic: str, module_num: int, item_num: int) -> str:
    s_clean = sentence.strip()
    match = re.match(r"^([A-Z][a-zA-Z0-9\s\-]{2,35}?)\s+(is|are|refers to|means|represents)\s+(.+)", s_clean)
    if match:
        subject = match.group(1).strip()
        return f"In the context of {topic}, what {match.group(2)} {subject}?"

    if " because " in s_clean.lower():
        parts = re.split(r"because", s_clean, flags=re.IGNORECASE)
        return f"Why does the following occur in {topic}: '{parts[0].strip()}'?"

    words = s_clean.split()
    preview = " ".join(words[:6]) + "..."
    return f"Explain the key principle regarding '{preview}' covered in {topic} (Concept {module_num}.{item_num})."


@app.post("/api/process")
async def process_lectures(request: LectureRequest):
    if not request.video_inputs:
        raise HTTPException(status_code=400, detail="Please provide at least one video URL or transcript.")

    raw_texts = []
    for idx, item in enumerate(request.video_inputs):
        if item.strip():
            raw_texts.append(fetch_transcript_or_text(item, idx))

    if not raw_texts:
        raise HTTPException(status_code=400, detail="All video inputs were empty.")

    return combine_and_analyze_lectures(request.topic, raw_texts)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    