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

## 🧩 Architecture

```
Cortex (main interface)
 ├── MemoryGraph       → Stores and retrieves memories
 ├── DefaultEmbedder   → Dummy or OpenAI embeddings
 └── MemoryDecay       → Tracks memory age + retention strength
```

---

## ⚙️ Installation

### 📦 1. Clone the repository

```bash
git clone https://github.com/yourusername/Cortica.git
cd Cortica
```

### 🧱 2. Install dependencies

#### Option A: Standard Python

```bash
pip install -e .
```

#### Option B: Using Pipenv (recommended for development)

```bash
pipenv install -e .
pipenv shell
```

> Note: `-e .` installs Cortica in editable mode, so changes to source files reflect immediately in your environment.

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

