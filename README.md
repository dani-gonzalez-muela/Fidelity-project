# Financial Compliance Text Classifier

**FINRA Rule 2210 violation detection using FinBERT** — built as a mini version of [Saifr](https://saifr.ai)'s core compliance product.

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. Run the full pipeline (scraping + training + evaluation)
jupyter notebook compliance_classifier_full.ipynb    # run all cells (~20 min first time)

# 2. Launch the demo
streamlit run app.py
```

> **Smart caching:** The notebook skips scraping if data exists and skips training if the model is already saved. Set `FORCE_RESCRAPE = True` or `FORCE_RETRAIN = True` at the top of the notebook to re-run any step.

---

## Project Structure

```
├── compliance_classifier_full.ipynb    # Full pipeline: scraping → augmentation → baselines → FinBERT → SHAP
├── app.py                              # Streamlit demo (2 pages: Overview + Interactive Checker)
├── requirements.txt                    # Dependencies
├── .env                                # (optional) API key for LLM rewrites
├── compliance_model_best/              # Trained model (auto-generated)
├── compliance_dataset.csv              # Scraped data (auto-generated)
├── compliance_dataset_multilabel.csv   # Augmented dataset (auto-generated)
└── benchmark_results.json              # Saved training times (auto-generated)
```

---

## Streamlit App

### Page 1: Overview & Results
- Problem context (FINRA Rule 2210, regulatory risk)
- Violation types explained with compliant/non-compliant examples
- Model benchmark comparison (TF-IDF vs fastText vs FinBERT)
- Technical approach and skills demonstrated

### Page 2: Interactive Checker
- Select from 12 pre-built test cases or type your own
- Real-time classification with confidence scores
- SHAP token attribution (which words triggered the flag)
- LLM-powered compliant rewrite suggestions

---

## Model Benchmark

| Model | F1 | Training Time | Size |
|-------|-----|--------------|------|
| TF-IDF + LogReg | ~0.90 | <1s | ~2 MB |
| fastText | ~0.92 | ~3s | ~8 MB |
| **FinBERT** | **~0.99** | **~20 min** | **~440 MB** |

---

## LLM Rewrite Setup (Optional)

The app suggests compliant rewrites for flagged text. It works out of the box with rule-based rewrites, but for higher quality, connect an LLM:

**Option 1:** Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=sk-ant-...
```
or
```
OPENAI_API_KEY=sk-...
```

**Option 2:** Set as environment variable before running:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

Install the corresponding SDK:
```bash
pip install anthropic   # for Claude
pip install openai      # for GPT-4o-mini
```

---

## Technical Approach

**Data:** Real FINRA enforcement actions (Rule 2210 violations) + compliance-reviewed fund marketing text + pattern-based augmentation (~500 examples across 5 classes)

**Model:** ProsusAI/FinBERT fine-tuned with 5-class classification head (compliant, promissory, exaggerated, unbalanced, misleading)

**Baselines:** TF-IDF + Logistic Regression, fastText — benchmarked against FinBERT for F1, latency, and model size

**Explainability:** SHAP token-level attribution for model interpretability

**GenAI:** LLM-powered compliant rewrite suggestions (Anthropic Claude or OpenAI GPT-4o-mini)
