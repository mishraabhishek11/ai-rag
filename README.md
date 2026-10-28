# ai-rag

A command-line question-answering assistant that uses Retrieval-Augmented Generation (RAG). It is the follow-up to [ai-assistant](https://github.com/mishraabhishek11/ai-assistant): instead of relying only on the model's own knowledge, it finds the most relevant local document for each question and asks the model to answer from that document only.

## How it works

1. At startup, every `.txt` file in `knowledge/` is loaded and turned into an embedding.
2. For each question you type, the question is embedded as well.
3. Cosine similarity picks the single best-matching document.
4. That document and the question are sent to the chat model, which is told to answer using only the supplied context.

## Requirements

- Python 3
- Packages: `openai`, `python-dotenv`, `numpy`
- An OpenAI-compatible API that supports both chat completions and embeddings

## Setup

1. Create and activate a virtual environment, then install the packages:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install openai python-dotenv numpy
   ```

2. Create a `.env` file in the project root:

   ```env
   BASE_URL=<your API base URL>
   API_KEY=<your API key>
   MODEL=<chat model name>
   EMBEDDING_MODEL=<embedding model name>
   ```

3. Add your knowledge as `.txt` files in `knowledge/`. The folder currently contains `machine_learning.txt`, `mysql.txt`, `python.txt` and `weather.txt`.

## Usage

```bash
python rag.py
```

Type a question at the `You :` prompt. Type `quit` to exit.

## Project structure

| File | Purpose |
|---|---|
| `rag.py` | Entry point: embeds the documents, runs the chat loop, builds the prompt |
| `embedding_helper.py` | `create_embedding` (API call) and `check_similarity` (cosine similarity) |
| `retriever.py` | `load_documents` reads `knowledge/*.txt`; `retrieve` is a simple substring matcher not used by `rag.py` |
| `knowledge/` | Source documents the assistant answers from |

## Known limitations

- Only the single best document is used; there is no chunking, so long documents are sent whole.
- Document embeddings are recomputed on every start and are not cached.
- The best match is used even when its similarity is low, so unrelated questions still get a (possibly poor) context.
- API errors are not caught.
- `knowledge/` is read relative to the current directory, so run the script from the project root.
- No `requirements.txt` or tests yet.

## Planned work

- Chunk documents and retrieve the top-k chunks
- Cache embeddings on disk
- Add a similarity threshold with an "I don't know" fallback
- Error handling with retries
- `requirements.txt` and tests
