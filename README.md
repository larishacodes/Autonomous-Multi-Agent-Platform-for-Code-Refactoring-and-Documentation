# Autonomous Multi-Agent Platform for Java Code Refactoring and Documentation

> Fine-tuned CodeT5+ 770M with LoRA adapters for automated Java code smell detection, refactoring, and Javadoc generation. Includes comparative evaluation against zero-shot Ollama models.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Modules](#modules)
- [Models](#models)
- [Ollama Testing](#ollama-testing)
- [Evaluation Metrics](#evaluation-metrics)
- [Known Limitations](#known-limitations)
- [Results Summary](#results-summary)

---

## What This Project Does

This platform takes a Java source file as input and runs it through a 6-block pipeline:

```
Java File
    ↓
[Block 1] Upload & Config
    ↓
[Block 2] Parse — extracts LOC, params, conditionals, nesting depth
    ↓
[Block 3] Smell Detection + Prompt Generation — DACOS thresholds
    ↓
[Block 4] Agent Orchestration
    ↓
[Block 5a] Refactor Agent (CodeT5+ 770M + LoRA, trained on RCCT Java)
[Block 5b] Doc Agent     (CodeT5+ 770M + LoRA, trained on CodeSearchNet Java)
    ↓
[Block 6a] Refactor Evaluator — AST validity, CodeBLEU, semantic, style
[Block 6b] Doc Evaluator     — coverage, completeness, @param/@return checks
    ↓
outputs/run_TIMESTAMP/
```

---

## Project Structure

```
Major_Project_Java/
│
├── main.py                          ← Entry point
├── pipeline.py                      ← Agent orchestration (Block 4)
├── config.json                      ← Settings (max_output, num_beams, thresholds)
│
├── SimpleTest.java                  ← Demo input (3 smells, fits model window)
├── OrderProcessor.java              ← Complex input (4 smells, smell detection demo)
│
├── parser/
│   └── java_parser.py               ← Block 2: javalang parser + regex fallback
│
├── prompt_engine/
│   ├── prompting_engine.py          ← Block 3: generates refactor + doc prompts
│   ├── smell_detector.py            ← Detects Long Method, Long Param, Complex Conditional, Multifaceted Abstraction
│   ├── templates.py                 ← 5-line prompt templates for 512-token window
│   ├── dacos_knowledge.py           ← DACOS default thresholds (LM=30, LP=5, CC=5, MA=2)
│   ├── dacos_integration.py         ← Loads real thresholds from DACOS SQL files
│   └── dacos_evaluator.py           ← DACOS-based smell scoring
│
├── agents/
│   ├── refactor_agent.py            ← Block 5a: CodeT5+ + LoRA refactoring
│   └── doc_agent.py                 ← Block 5b: CodeT5+ + LoRA documentation
│
├── evaluator/
│   ├── refactor_evaluator.py        ← Block 6a: AST, style, CodeBLEU, semantic, improvement
│   └── doc_evaluator.py             ← Block 6b: coverage, completeness, length
│
├── models/
│   ├── refactor_agent_final/        ← LoRA adapter (trained on RCCT Java, BLEU-4=0.6488)
│   └── doc_agent_final/             ← LoRA adapter (trained on CodeSearchNet Java 50k)
│
├── datasets/
│   └── dacos/                       ← DACOS SQL files for threshold loading
│
├── training/
│   └── train_doc_agent_java.py      ← Kaggle training script (doc agent)
│
├── ollama_testing/                  ← Separate Ollama comparison module
│   ├── run_ollama_test.py           ← Entry point
│   ├── ollama_client.py             ← Ollama REST API wrapper
│   ├── ollama_refactor.py           ← Refactoring via Ollama
│   ├── ollama_doc.py                ← Documentation via Ollama
│   ├── ollama_evaluator.py          ← Evaluation metrics (same as main pipeline)
│   ├── ollama_compare.py            ← Side-by-side comparison report
│   └── README.md                    ← Ollama module documentation
│
└── outputs/
    └── run_TIMESTAMP/               ← One folder per pipeline run
        ├── refactored_code.java
        ├── documentation.md
        ├── smell_report.txt
        ├── refactoring_plan.txt
        ├── EVALUATION_refactor.json
        ├── EVALUATION_refactor.txt
        ├── EVALUATION_doc.json
        ├── EVALUATION_doc.txt
        ├── PROMPT_refactor_agent.txt
        ├── PROMPT_doc_agent.txt
        ├── parsed_analysis.json
        └── summary.json
```

---

## Setup

### Requirements

- Python 3.10+
- Windows 10/11 (tested) or Linux
- 8 GB RAM minimum (16 GB recommended)
- GPU optional — CPU inference supported (slower)

### Installation

```bash
cd C:\Users\Administrator\Documents\Major_Project_Java

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install transformers==4.44.0 peft==0.10.0 accelerate==0.29.0
pip install torch datasets safetensors
pip install rouge-score sacrebleu bert-score
pip install javalang matplotlib seaborn plotly
pip install requests huggingface_hub
```

---

## How to Run

### Main Pipeline

```bash
# Activate venv first
venv\Scripts\activate

# Run on demo file (both refactoring and documentation)
python main.py --file SimpleTest.java --mode both

# Refactoring only
python main.py --file SimpleTest.java --mode refactor

# Documentation only
python main.py --file SimpleTest.java --mode doc

# Run on your own Java file
python main.py --file YourClass.java --mode both
```

### Output

Results are saved to `outputs/run_TIMESTAMP/`. Key files:

| File | Contents |
|------|----------|
| `refactored_code.java` | Refactored Java output from CodeT5+ |
| `documentation.md` | Generated Javadoc |
| `smell_report.txt` | Detected smells with reasons |
| `EVALUATION_refactor.txt` | Human-readable refactor scores |
| `EVALUATION_doc.txt` | Human-readable doc scores |
| `summary.json` | All scores in one JSON |

---

## Modules

### Block 2 — Parser (`parser/java_parser.py`)

Uses `javalang` (Java 8) with regex fallback. Extracts:

- Lines of code (LOC)
- Parameter count
- Conditional count
- Loop count
- Nesting depth
- Responsibility count

Verified: 32/32 checks pass on `OrderProcessor.java`

### Block 3 — Prompting Engine (`prompt_engine/`)

Detects smells using DACOS thresholds and generates prompts:

| Smell | Default Threshold |
|-------|-----------------|
| Long Method | 30 lines |
| Long Parameter List | 5 params |
| Complex Conditional | 5 conditionals |
| Multifaceted Abstraction | 2 responsibilities |

Prompt fits within 512 tokens (method-level only, not full file).

### Block 5a — Refactor Agent (`agents/refactor_agent.py`)

- Base model: `Salesforce/codet5p-770m`
- Fine-tuned on RCCT Java dataset
- BLEU-4 = 0.6488 on training set
- `max_output_length=256`, `num_beams=2` (CPU-friendly)
- Post-processing fixes `#comments`, unclosed braces, indentation

### Block 5b — Doc Agent (`agents/doc_agent.py`)

- Base model: `Salesforce/codet5p-770m`
- Fine-tuned on CodeSearchNet Java (50k examples, 3 epochs)
- LoRA: r=8, alpha=32, dropout=0.05
- ROUGE-L = 0.2012 on test set (real generation metric)

### Block 6a — Refactor Evaluator (`evaluator/refactor_evaluator.py`)

| Metric | Weight | Description |
|--------|--------|-------------|
| Style | 20% | 6 Java style checks |
| CodeBLEU | 25% | Code-aware BLEU score |
| Semantic | 35% | Method preservation rate |
| Improvement | 20% | LOC reduction score |

### Block 6b — Doc Evaluator (`evaluator/doc_evaluator.py`)

| Metric | Weight | Description |
|--------|--------|-------------|
| Coverage | 35% | Methods documented / total |
| Completeness | 35% | Has overview, params, return, Javadoc style |
| Length | 30% | Word count in expected range |

---

## Models

### Refactor Agent

| Property | Value |
|----------|-------|
| Base model | Salesforce/codet5p-770m |
| Training dataset | RCCT Java |
| Training platform | Google Colab → Kaggle |
| BLEU-4 | 0.6488 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Location | `models/refactor_agent_final/` |

### Doc Agent

| Property | Value |
|----------|-------|
| Base model | Salesforce/codet5p-770m |
| Training dataset | CodeSearchNet Java (50k / 412k) |
| Training platform | Kaggle P100 |
| ROUGE-L | 0.2012 (test set, greedy decoding) |
| ROUGE-1 | 0.2625 |
| BLEU | 4.59 |
| BERTScore | 0.7644 |
| LoRA rank | 8 |
| LoRA alpha | 32 |
| Epochs | 3 (early stopping at epoch 1) |
| Location | `models/doc_agent_final/` |

---

## Ollama Testing

A separate module tests Ollama models on the same prompts as the main pipeline — no changes to the main project.

### Setup

```bash
# Ollama already installed? Start it:
ollama serve

# Pull models
ollama pull codellama:7b
ollama pull deepseek-coder:6.7b
ollama pull gemma3:12b
```

### Run

```bash
# Step 1 — run main pipeline to generate prompts
python main.py --file SimpleTest.java --mode both

# Step 2 — test Ollama models on same prompts
cd ollama_testing
python run_ollama_test.py --pipeline_run ..\outputs\run_TIMESTAMP

# Step 3 — generate comparison report
python ollama_compare.py \
    --pipeline_run ..\outputs\run_TIMESTAMP \
    --ollama_run   ollama_outputs
```

### Ollama Output Structure

```
ollama_outputs/
├── codellama_7b/
│   ├── refactored_code.java
│   ├── documentation.md
│   ├── eval_refactor.json
│   ├── eval_doc.json
│   ├── result_refactor.json
│   └── result_doc.json
├── deepseek-coder_6_7b/
│   └── ...
├── gemma3_12b/
│   └── ...
└── ollama_summary_TIMESTAMP.json
```

---

## Evaluation Metrics

### Refactoring (main pipeline)

| Metric | CodeT5+ Result |
|--------|---------------|
| AST Valid | YES |
| Style score | 0.833 |
| CodeBLEU | 0.383 |
| Semantic | 1.000 |
| Improvement | 0.892 |
| Confidence | 0.791 (GOOD) |

### Documentation (main pipeline)

| Metric | CodeT5+ Result |
|--------|---------------|
| Coverage | 1.000 |
| Completeness | 0.800 |
| Has @return | NO |
| Confidence | 0.930* |

*Note: 0.930 is a structural score. The model generates one sentence with no `@param` or `@return` tags.

### Ollama Comparison (codellama:7b, zero-shot)

| Metric | CodeT5+ | CodeLlama 7B |
|--------|---------|--------------|
| Refactor confidence | 0.791 | 0.830 |
| Logic correctness | ~0.40 | 1.000 |
| Doc confidence | 0.930* | 1.000 |
| @param coverage | 0.000 | 1.000 |
| Has @return | NO | YES |

---

## Known Limitations

| Limitation | Details |
|-----------|---------|
| Java 8 only | `javalang` parser supports Java 8 syntax. Java 9+ uses regex fallback |
| Model window | Effective on methods up to ~45 lines / 6 params (512-token window) |
| Refactor logic | Model may produce structurally valid but logically incorrect code |
| Doc quality | Fine-tuned model generates brief descriptions without `@param`/`@return` |
| Doc training | 50k of 412k examples used due to P100 GPU time constraints |
| Refactor training | Epoch 3 trained on Kaggle after epochs 1–2 on Colab (checkpoint transfer) |
| Style score | 0.833 — model outputs bare methods without class wrapper (known behaviour) |

---

## Results Summary

### Input: SimpleTest.java — PaymentProcessor.applyDiscount

- 40 lines, 1 method, 33 LOC, 6 params, 8 conditionals
- Smells: Long Method, Long Parameter List, Complex Conditional (all MEDIUM)

### Pipeline Run

```
Refactoring:
  Confidence  : 0.791  GOOD
  AST Valid   : YES
  LOC change  : 40 → 33
  CodeBLEU    : 0.383

Documentation:
  Confidence  : 0.930  EXCELLENT (structural)
  Coverage    : 1.000
  Completeness: 0.800
```

### Key Finding

Zero-shot Ollama models (codellama:7b, 7B params) outperform fine-tuned CodeT5+ (770M params) on both refactoring logic correctness and documentation completeness. This highlights the trade-off between model size and fine-tuning on domain-specific data at limited compute budgets.

---

## Paper Citation

If using this project for research:

```
Autonomous Multi-Agent Platform for Java Code Smell Detection,
Refactoring, and Documentation using Fine-tuned CodeT5+ 770M with LoRA Adapters.
Dataset: RCCT Java (refactoring), CodeSearchNet Java (documentation), DACOS (thresholds).
```
