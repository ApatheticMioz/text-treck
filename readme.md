
# 🇵🇰 E-Sahulat: Sovereign Multilingual Agentic RAG

An edge-deployable, zero-trust Agentic Retrieval-Augmented Generation (RAG) system engineered for citizen services at the **Directorate General of Immigration & Passports (DGI&P)**[cite: 2]. It natively supports **Urdu (Nastaliq script)**, **Pakistani Roman Urdu**, **Code-Mixed queries**, and **English** with deterministic safety guardrails[cite: 2].

---

## 🚀 Key Highlights

* **100% Benchmark Accuracy:** Passes 36/36 test cases across all 12 official hackathon evaluation categories[cite: 2].
* **Hybrid Retrieval (W-RRF):** Weighted Reciprocal Rank Fusion combining content-filtered BM25Okapi ($w=0.22$) with dense multilingual embeddings (`BAAI/bge-m3`, $w=0.78$)[cite: 2].
* **Zero Hallucination Guardrails:** Semantic prototype routing and information-theoretic entropy filters intercept prompt injections, ambiguous queries, and personal tracking requests deterministically[cite: 2].
* **Sovereign Edge Deployment:** Runs locally inside **3.27 GB GPU VRAM** using `Qwen/Qwen2.5-3B-Instruct` in 4-bit NF4 quantization on an Nvidia T4[cite: 2].

---

## 🛠️ System Architecture

```text
Citizen Query (Urdu / Roman Urdu / English / Code-Mixed)
                           │
                           ▼
  [Stage 1: Linguistic & Phonetic Normalizer]
  • Unicode NFKC canonicalization + Aerab diacritic stripping
  • Vernacular Roman Urdu phonetic lemmatization ("pasport" ➔ "passport", "hy" ➔ "hai")
                           │
                           ▼
  [Stage 2: Hybrid W-RRF Retriever]
  • BM25Okapi (lexical matching) + BAAI/bge-m3 (dense cross-lingual cosine search)
                           │
                           ▼
  [Stage 3: Agentic Semantic Prototype Router & Entropy Filter]
  • Gibbs entropy check for keyboard mash / non-linguistic noise
  • Contrastive prototype routing across 6 guardrail clusters
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    [Guardrail Abstention]     [Grounded Synthesis]
    Deterministic safety       Qwen2.5-3B-Instruct (NF4 4-bit)
    response + official link   Script-locked few-shot ICL + verbatim citation

```

---

## 📊 Benchmark & Evaluation Results

Tested against the official 12-category hackathon benchmark suite ($N=36$):

| # | Validation Category | $N$ | Passed | Pass Rate | Primary Evaluation Focus |
| --- | --- | --- | --- | --- | --- |
| 1 | **Urdu Script Query (Nastaliq)** | 3 | 3 | **100.0%** | Exact Intent Recall + RTL Grounded Output

 |
| 2 | **Roman Urdu Query** | 3 | 3 | **100.0%** | Phonetic Robustness + Native Roman Urdu Synthesis

 |
| 3 | **English Query** | 3 | 3 | **100.0%** | Regulatory Policy Grounding

 |
| 4 | **Code-Mixed Query** | 3 | 3 | **100.0%** | Cross-Lingual Token Ingestion

 |
| 5 | **Spelling Variation / Typos** | 3 | 3 | **100.0%** | Phonetic Lemmatization & Fuzzy Match

 |
| 6 | **Paraphrased Query** | 3 | 3 | **100.0%** | Dense Semantic Invariance

 |
| 7 | **Unsupported Personal Question** | 3 | 3 | **100.0%** | Abstention; Direct to Official Tracking

 |
| 8 | **Out of Domain (OOD)** | 3 | 3 | **100.0%** | Domain Boundary Enforcement

 |
| 9 | **Ambiguous Question** | 3 | 3 | **100.0%** | Dynamic Clarification Prompting

 |
| 10 | **Contradictory / Outdated Evidence** | 3 | 3 | **100.0%** | Re-anchors to Canonical Live Authority

 |
| 11 | **Citation Integrity** | 3 | 3 | **100.0%** | 100% Verifiable Official DGI&P Source URLs

 |
| 12 | **No Semantic Reference (Gibberish)** | 3 | 3 | **100.0%** | Entropy & Mash Interception

 |
| **TOTAL** | **Full Benchmark Suite** | **36** | **36** | **100.0%** | **Mean Latency: 2,541.7 ms | P95: 7,181.5 ms**<br> |

### Retrieval Ablation Comparison

| Retriever Architecture | Top-1 Recall (Answerable) | Full 36-Case Accuracy | Typo / Phonetic Handling |
| --- | --- | --- | --- |
| **Lexical BM25 Only** (Content-Filtered) | 18 / 21 (85.7%) | N/A (No OOD Router) | Fails on slang / spelling drifts |
| **Dense Only** (`BAAI/bge-m3`) | 20 / 21 (95.2%) | N/A (OOD False Positives) | Robust across languages; drifts on numbers |
| **E-Sahulat Hybrid W-RRF + Router** | **21 / 21 (100.0%)** | **36 / 36 (100.0%)** | **Zero Failures** |

---

## 📂 Repository Artifacts

* `Untitled3.ipynb`: Complete runnable pipeline (normalizer, retrieval, router, 4-bit LLM, benchmark harness, and Gradio UI).


* `Multilingual E-Sahulat AI_Hackathon_Dataset.txt`: 150 documented records across 30 official DGI&P intents with source URLs and evidence text.


* `validation_pack_36_cases.csv`: Full evaluation trace for all 36 test cases with execution latency and router telemetry.


* `retrieval_comparison_table.csv`: Quantitative retrieval ablation matrix.


* `validation_pack_deliverable.zip`: Bundled evaluation package.


* `E_Sahulat_Technical_Report.pdf`: 5-page technical report with full methodology and error analysis.


* `E-Sahulat_ Sovereign Multilingual RAG.pptx`: Official presentation slide deck.



---

## ⚡ Quickstart & Installation

### 1. Requirements

Ensure a GPU environment is available (e.g., Google Colab with standard 16GB Nvidia T4 GPU).

```bash
pip install torch transformers sentence-transformers rank-bm25 bitsandbytes accelerate gradio numpy

```

### 2. Run the Notebook

Open `Untitled3.ipynb` in Google Colab or Jupyter and execute all cells sequentially. The script will:

1. Load and index the 30 DGI&P intents from `Multilingual E-Sahulat AI_Hackathon_Dataset.txt`.


2. Initialize `BAAI/bge-m3` and load `Qwen/Qwen2.5-3B-Instruct` in NF4 4-bit quantization.


3. Run the complete 36-case benchmark evaluation.


4. Launch the local and public Gradio interface.



---

## 🌐 Live Demonstration

* **Live Demo Interface:** [https://4b08bd017577503d8f.gradio.live](https://4b08bd017577503d8f.gradio.live)
* **Official Data Provenance:** Directorate General of Immigration & Passports, Pakistan (`https://dgip.gov.pk`)



```

