# 💡 Cortica

**Cortica** is a lightweight, modular memory engine for intelligent systems.  
Inspired by the human cortex, it enables semantic storage, decay-aware retrieval, and memory traversal — all in a portable package that integrates easily into any Python or AI/LLM project.

---

## 🔌 Plug & Play Usage

```python
from cortica.cortex import Cortex

cortex = Cortex()

cortex.remember("Mitochondrial inflammation is a key biomarker")
cortex.remember("Neuroimmune disorders often present with fatigue")
cortex.remember("Autonomic dysfunction can be tracked via HRV")

results = cortex.query("What symptoms are associated with long-COVID?")
for r in results:
    print(f"💡 {r['concept']} (score: {r['score']:.3f})")
```

---

## 🧠 Core Features

| Feature               | Description |
|-----------------------|-------------|
| **Memory Storage**    | Store concept nodes with embeddings + metadata |
| **Semantic Retrieval**| Cosine similarity-based querying, with optional memory decay |
| **Traversal**         | Walk memory chains to simulate contextual thought |
| **Memory Decay**      | Built-in half-life model — memories fade unless reinforced |
| **RAG/LLM Ready**     | Plug in OpenAI, LangChain, or custom embedders easily |

---

## ✨ Why Cortica?

- **No heavy dependencies**
- Works offline with dummy embeddings
- Drop-in for LangChain, FastAPI, notebooks, or custom stacks
- Designed for **semantic cognition**, not just search

---

## 🧠 How Cortica Compares to ChromaDB

While ChromaDB is a powerful tool for storing and retrieving vectorized documents in large-scale LLM pipelines, Cortica serves a different purpose: it acts as a **lightweight conceptual memory system**.

| Feature / Capability          | **ChromaDB**                             | **Cortica**                                |
|------------------------------|-------------------------------------------|---------------------------------------------|
| **Primary Role**             | Persistent vector store (RAG backend)     | Semantic memory system (conceptual graph)   |
| **Memory structure**         | Flat — document fragments or chunks       | Graph — semantically-linked concepts        |
| **Recall logic**             | Similarity search                         | Similarity + relationship traversal         |
| **Decay / Forgetting**       | ❌ None                                    | ✅ Supports temporal decay & memory strength |
| **Traversal**                | ❌ No path-based reasoning                | ✅ Drift-based conceptual paths              |
| **Storage model**            | Disk-based, DB-backed                     | In-memory or swappable storage              |
| **Ideal Use Case**           | Large-scale RAG over static corpora       | Local-first reasoning, agents, thought drift|

> 🧠 **Cortica is to Chroma what working memory is to a bookshelf.**  
Chroma stores your data — Cortica lets your system *form opinions* on it over time.

---


## 🧩 Architecture

```
Cortex (main interface)
 ├── MemoryGraph       → Stores and retrieves memories
 ├── DefaultEmbedder   → Dummy or OpenAI embeddings
 └── MemoryDecay       → Tracks memory age + retention strength
```

---

## ⚙️ Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/yourusername/Cortica.git
cd Cortica
pip install -e .
```

> Cortica requires Python 3.10+

---

## 🚀 Usage

Cortica **requires an embedding backend** — it does not include a default model.  
You must pass in an object with a method: `.embed(text: str) -> List[float]`.

### 🧠 Example: Using OpenAI Embeddings

```python
from cortica.cortex import Cortex
from cortica.embed import DefaultEmbedder

import os
embedder = DefaultEmbedder(mode="openai", api_key=os.getenv("OPENAI_API_KEY"))
cortex = Cortex(embedder=embedder)

# Store concepts
cortex.remember("Price above SMA20 showed strong KL divergence")
cortex.remember("MACD histogram crossed above zero at day 13")

# Query memory
results = cortex.query("What indicators showed strong divergence?")
for r in results:
    print(f"💡 {r['concept']} (score: {r['score']:.3f})")
```

---

### 💡 Tip: Bring Your Own Embedder

You can pass in **any object** with a `.embed(text: str)` method.

```python
class MyEmbedder:
    def embed(self, text):
        return [0.1, 0.5, 0.3]  # your own vector logic

cortex = Cortex(embedder=MyEmbedder())
```

---

Cortica is portable, model-agnostic, and designed for clean integration into any RAG or vector-based reasoning pipeline.

---

## 🧪 Coming Soon

- Disk-based memory persistence
- LangChain + Streamlit examples
- Memory clustering and concept hierarchies
- Graph-based relationship visualization

---

## 📄 License

MIT License  
Cortica is free to use, fork, and integrate.

---

Made with brainwaves.  

