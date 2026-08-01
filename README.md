# 📄 JJRAG — Document Q&A with a switchable model backend

Upload your own documents and ask questions grounded in their contents. JJRAG
retrieves the relevant passages and generates an answer — and lets you choose,
from a single dropdown, **how** that answer is produced:

| Mode | Backend | Use it for |
| --- | --- | --- |
| 🔒 **Private (Local)** | Open-source models via **Ollama** + local embeddings | Maximum privacy — nothing leaves your machine, no API key needed |
| 🎯 **Accuracy (Cloud)** | The current best hosted model (**Claude Opus 4.8** by default; OpenAI optional) | Best answer quality — you supply your **own** API key |
| 🚀 **Self-hosted (Your GPU)** | **Kimi** (or any model) served with vLLM on **[AMD Developer Cloud](#-running-kimi-on-amd-developer-cloud)** — OpenAI-compatible | Open-weights power at frontier scale, on GPUs *you* control |

The third option is an OpenAI-compatible endpoint, so you can point JJRAG at any
model you host yourself. The walkthrough below uses **Kimi on AMD Instinct GPUs
in AMD Developer Cloud**.

Your API key is entered in the UI, held in session memory only, and is **never
written to disk, never logged, and never sent anywhere except the provider you
picked**. Document text is embedded **locally in both modes**, so your files are
never uploaded just to be indexed.

## Demo

![JJRAG demo](demo.gif)

## Features

- 📥 **Manually add documents** — drag in any number of PDF, DOCX, or TXT files.
- 🔀 **Switchable model** — pick Private vs. Accuracy from a dropdown; in
  Accuracy mode choose Anthropic (Claude) or OpenAI (GPT).
- 🔑 **Private API-key slot** — password-masked, session-only, never persisted.
- 🔒 **Local embeddings** (`all-MiniLM-L6-v2`) + **FAISS** vector search.
- 💬 **Grounded answers** with the exact source excerpts shown alongside.
- 🌊 **Streamed responses** for both local and cloud backends.

## Architecture

```
uploaded docs ─▶ load (PDF/DOCX/TXT) ─▶ chunk ─▶ LOCAL embeddings ─▶ FAISS
                                                                        │
                        question ─▶ similarity search (top-k) ──────────┘
                                            │
                    ┌───────────────────────┴────────────────────────┐
              🔒 Private                                  🎯 Accuracy / 🚀 Self-hosted
        Ollama (local LLM)                    Claude · OpenAI GPT · Kimi on AMD Dev Cloud
        no network, no key                          your key / your endpoint, in-memory only
```

Accuracy and Self-hosted share one OpenAI-compatible code path — the only
difference is whether `base_url` points at OpenAI or at *your* vLLM server.

The generation step is the piece added on top of the original retrieval-only
prototype (`JJrag_final.py`), which returned raw chunks without ever calling a
model. The chunk size was also fixed (40 → 1000 chars) so retrieval returns
coherent passages.

## Quick start

### 1. Install

```bash
git clone https://github.com/<your-user>/JJRAG.git
cd JJRAG
pip install -r requirements.txt
```

### 2. (Private mode) install a local model

```bash
# https://ollama.com — then pull any model you like:
ollama pull llama3          # or: ollama pull deepseek-r1:8b
```

### 3. Run

```bash
streamlit run rag_app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).

## Using the app

1. In the sidebar, choose **Private** or **Accuracy** from the dropdown.
   - **Accuracy** → pick a provider and paste your API key into the 🔑 slot.
     - Anthropic (Claude), OpenAI (GPT), or **Kimi on AMD Developer Cloud** — the
       last one asks for an **endpoint base URL** (your self-hosted vLLM server).
   - **Private** → pick a locally installed Ollama model.
2. Drag documents into the uploader and click **Process documents**.
3. Type a question and click **Ask**. The answer streams in, with the source
   excerpts shown in the **Sources used** expander.

## 🚀 Running Kimi on AMD Developer Cloud

[AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud.html) gives
you on-demand **AMD Instinct™ MI300X** GPUs (192 GB HBM3 each) with ROCm, PyTorch,
and vLLM preinstalled — and new accounts get **free credits** to start. That's
enough horsepower to self-host a frontier open-weights model like **Kimi**
(Moonshot AI's Mixture-of-Experts model) and expose it as an **OpenAI-compatible
endpoint** that JJRAG talks to directly.

> **Which "Kimi"?** These steps work for any Kimi release — swap the Hugging Face
> model id for the tag you want (e.g. `moonshotai/Kimi-K2-Instruct`, or the newer
> **Kimi K3** id once you have access). Kimi is a very large MoE model; plan for a
> multi-GPU node (an 8× MI300X instance comfortably serves the full-precision
> weights, and smaller/quantized variants fit on fewer).

### 1. Get access and credits

1. Go to **[AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud.html)**
   and sign in with (or create) your AMD account.
2. Apply for access / claim your starter **credits** when prompted.
3. Add an SSH public key to your account so you can log into instances
   (`ssh-keygen -t ed25519` locally if you don't have one).

### 2. Launch an AMD Instinct GPU instance

1. In the console, **create a new GPU droplet/instance** and pick an
   **MI300X** shape (choose an 8× GPU node for the full Kimi weights).
2. Select a **ROCm + PyTorch** base image (vLLM ships in AMD's ROCm images).
3. Launch it, then copy the instance's **public IP** and SSH in:

   ```bash
   ssh root@<your-amd-droplet-ip>
   rocm-smi          # confirm the GPUs are visible
   ```

### 3. Serve Kimi with a vLLM OpenAI-compatible server

AMD publishes ROCm-optimized vLLM images. Start the server on port **8000** —
`vllm serve` exposes the standard `/v1/chat/completions` OpenAI API:

```bash
# Pull AMD's ROCm vLLM image and serve Kimi (adjust the tag to the latest ROCm build)
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --network=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=<your-hugging-face-token> \
  rocm/vllm:latest \
  vllm serve moonshotai/Kimi-K2-Instruct \
    --served-model-name moonshotai/Kimi-K2-Instruct \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --api-key sk-jjrag-demo \
    --port 8000
```

- `--tensor-parallel-size 8` shards the model across 8 MI300X GPUs — set it to the
  GPU count of your node.
- `--api-key sk-jjrag-demo` is a token **you choose**; you'll paste the same value
  into JJRAG. Omit it for no auth (use an SSH tunnel instead — see below).
- The first run downloads the weights from Hugging Face (large — this takes a
  while); subsequent runs reuse the cached copy.

Confirm it's up:

```bash
curl http://localhost:8000/v1/models   # lists moonshotai/Kimi-K2-Instruct
```

### 4. Reach the endpoint from your laptop

**Recommended (secure): SSH tunnel** — keeps the port off the public internet:

```bash
# On your laptop, forward local :8000 to the droplet's :8000
ssh -N -L 8000:localhost:8000 root@<your-amd-droplet-ip>
```

Then JJRAG's base URL is `http://localhost:8000/v1`.

**Alternative: open the port** in the AMD Developer Cloud firewall/security group
for your IP, and use `http://<your-amd-droplet-ip>:8000/v1` directly.

### 5. Point JJRAG at Kimi

Run the app (`streamlit run rag_app.py`) and in the sidebar:

1. Mode → **🎯 Accuracy (Cloud API)**.
2. Provider → **Kimi on AMD Developer Cloud (OpenAI-compatible)**.
3. **Endpoint base URL** → `http://localhost:8000/v1` (tunnel) or your droplet URL.
4. **🔑 API key** → the `--api-key` value you launched vLLM with (e.g.
   `sk-jjrag-demo`; any placeholder works if you started the server without one).
5. **Model** → `moonshotai/Kimi-K2-Instruct` (must match `--served-model-name`).

Process your documents and ask away — retrieval and embeddings still run locally;
only the final generation call goes to *your* Kimi server on AMD hardware.

> **Tip:** prefer not to type the URL each time? Put
> `OPENAI_BASE_URL=http://localhost:8000/v1` in your `.env` and it pre-fills the box.

## Privacy & security

- 🔑 API keys live only in `st.session_state` for the browser session — not on
  disk, not in logs.
- 🔒 `.gitignore` excludes `.env`, `key.env`, `*.key`, logs, and caches so
  secrets can never be committed.
- 🖥️ Private mode makes **zero** outbound network calls — verify with a network
  monitor if you like.
- 📁 Embeddings run locally in both modes; only Accuracy mode's final
  generation call reaches a provider.

## Configuration

Copy `.env.example` to `.env` to override defaults (e.g. `OLLAMA_HOST`).
Environment-variable keys are optional — the in-app key box is the primary path.

## Original design notes

The initial design write-up and the original prototype (`JJrag.py`) are kept in
[`DESIGN_NOTES.md`](DESIGN_NOTES.md) and `JJrag.py`.
