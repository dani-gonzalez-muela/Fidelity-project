"""
Financial Compliance Checker — Streamlit Demo
=============================================
Page 1: Overview — concise intro, violation types (tabs), model benchmark (dataframe)
Page 2: Interactive Checker — classify, rewrite, then SHAP

Usage: streamlit run app.py
"""

import streamlit as st
#import torch
import numpy as np
import pandas as pd
import os
import json
import re

st.set_page_config(
    page_title="Financial Compliance Checker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load .env if present ──
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

# ── Minimal CSS — just font size bump, no color overrides ──
st.markdown("""
<style>
    .stMarkdown p, .stMarkdown li { font-size: 17px !important; line-height: 1.65 !important; }
    h1 { font-size: 2.6rem !important; }
    h2 { font-size: 1.9rem !important; }
    h3 { font-size: 1.4rem !important; }
    .stTextArea textarea { font-size: 17px !important; }
    .stSelectbox label, .stTextArea label { font-size: 17px !important; font-weight: 600 !important; }
    
    /* SHAP container */
    .shap-container {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 22px;
        margin: 12px 0;
        font-size: 19px;
        line-height: 2.4;
    }
    .shap-token {
        padding: 3px 6px;
        border-radius: 5px;
        margin: 1px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# Constants
# ══════════════════════════════════════════
LABELS = ["compliant", "promissory", "exaggerated", "unbalanced", "misleading"]

LABEL_ICONS = {
    "compliant": "✅", "promissory": "🔴", "exaggerated": "🟠",
    "unbalanced": "🟡", "misleading": "🟣",
}

LABEL_DESCRIPTIONS = {
    "promissory": {
        "short": "Guarantees or promises of returns",
        "rule": "FINRA Rule 2210(d)(1)(B)",
        "detail": "Language that guarantees, promises, or implies certain investment returns or safety of principal.",
        "triggers": "guarantee, promise, will deliver, risk-free, never lose, consistent returns",
        "example_bad": "Our fund guarantees 12% returns every year regardless of market conditions.",
        "example_good": "Our fund seeks to provide competitive risk-adjusted returns. Past performance is not a guarantee of future results.",
    },
    "exaggerated": {
        "short": "Superlatives or unsubstantiated claims",
        "rule": "FINRA Rule 2210(d)(1)(A)",
        "detail": "Unsubstantiated superlatives or performance claims without supporting evidence.",
        "triggers": "best, #1, unmatched, always outperforms, no other fund comes close",
        "example_bad": "This is the best performing fund in the industry. No other fund comes close.",
        "example_good": "This fund has delivered strong historical performance relative to its benchmark.",
    },
    "unbalanced": {
        "short": "Returns without risk disclosure",
        "rule": "FINRA Rule 2210(d)(1)(A)",
        "detail": "Highlighting positive performance without mentioning that investors could lose money.",
        "triggers": "returned X%, don't miss out, act now, invest today (without risk disclaimer)",
        "example_bad": "Our fund returned 15% last year. Don't miss out — invest today!",
        "example_good": "Our fund returned 15% last year. Past performance does not guarantee future results. All investing involves risk.",
    },
    "misleading": {
        "short": "False or deceptive statements",
        "rule": "FINRA Rule 2210(d)(1)(A) + SEC Rule 206(4)-1",
        "detail": "Claims that are factually false or make claims about protections that don't exist.",
        "triggers": "FDIC insured (for funds), backed by government, 99% accuracy, risk-free",
        "example_bad": "Our AI predicts the market with 99% accuracy. Your money is completely safe.",
        "example_good": "Our quantitative models use data-driven analysis to inform decisions. All investing is subject to risk.",
    },
}

EXAMPLE_TEXTS = {
    "🔴 Promissory — Guaranteed returns": "Our fund guarantees 12% returns every year regardless of market conditions.",
    "🔴 Promissory — Saifr's published example": "Investing in XYZ fund provides consistent returns through market ups and downs.",
    "🟠 Exaggerated — Superlative claims": "This is the best performing fund in the industry. No other fund comes close.",
    "🟠 Exaggerated — Talent claims": "Our fund managers are the most talented in the world.",
    "🟡 Unbalanced — Returns without risk": "Our fund returned 15% last year. Don't miss out — invest today!",
    "🟡 Unbalanced — FOMO pressure": "Join thousands of investors who earned double-digit returns with us.",
    "🟣 Misleading — False safety": "Our AI predicts the market with 99% accuracy. Your money is completely safe.",
    "🟣 Misleading — False insurance": "This fund is FDIC insured and risk-free.",
    "✅ Compliant — Index tracking": "The fund seeks to track the performance of the S&P 500 Index.",
    "✅ Compliant — Proper disclaimer": "Past performance is not a guarantee of future results. You could lose money.",
    "⚪ Edge — Returns + disclaimer": "Our fund returned 15% last year. Past performance does not guarantee future results.",
    "⚪ Edge — Hedged marketing": "We believe our disciplined approach may provide attractive risk-adjusted returns over time.",
}


# ══════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════
@st.cache_resource
def load_model():
    from transformers import pipeline
    return pipeline("text-classification", model="./compliance_model_best",
                    tokenizer="./compliance_model_best", device=-1, top_k=None)


@st.cache_resource
def load_explainer():
    import shap
    return shap.Explainer(load_model())


# ══════════════════════════════════════════
# LLM Rewrite
# ══════════════════════════════════════════
REWRITE_SYSTEM_PROMPT = """You are a financial compliance expert specializing in FINRA Rule 2210 
(Communications with the Public) and SEC Marketing Rule 206(4)-1.

Given a non-compliant financial marketing sentence and its violation type, provide:
1. A brief explanation of WHY it's non-compliant (1 sentence)
2. A compliant rewrite that preserves the marketing intent but uses proper hedge language

Rules for compliant rewrites:
- Replace guarantees/promises with "seeks to" or "is designed to"
- Replace superlatives with qualified language ("among the", "we believe")
- Add risk disclaimers when presenting performance data
- Use "may", "could", "potential" instead of certainties
- Never promise specific returns or safety

Format your response as:
WHY: [explanation]
REWRITE: [compliant version]"""


def get_llm_rewrite(text, violation_type):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=300, system=REWRITE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Violation type: {violation_type}\n\nNon-compliant text: \"{text}\"\n\nProvide a compliant rewrite."}],
            )
            return response.content[0].text, "anthropic"
        except Exception:
            pass

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Violation type: {violation_type}\n\nNon-compliant text: \"{text}\"\n\nProvide a compliant rewrite."},
                ],
                max_tokens=300, temperature=0.3,
            )
            return response.choices[0].message.content, "openai"
        except Exception:
            pass

    return _rule_based_rewrite(text, violation_type), "rule-based"


def _rule_based_rewrite(text, violation_type):
    rewrite = text
    if violation_type == "promissory":
        for p, r in [
            (r"\bguarantee[sd]?\b", "seeks to provide"), (r"\bwill deliver\b", "seeks to deliver"),
            (r"\bwill provide\b", "seeks to provide"), (r"\bwill never lose\b", "is designed to help manage"),
            (r"\bensures?\b", "seeks to provide"), (r"\bpromises?\b", "aims"),
            (r"\bconsistent returns\b", "competitive risk-adjusted returns"),
            (r"\bnever lost money\b", "historically managed downside risk"),
            (r"\bsafe\b", "diversified"), (r"\block[s]? in\b", "targets"),
            (r"\beliminate[sd]? all (?:investment )?risk\b", "seeks to manage risk"),
            (r"\brisk.free\b", "lower-risk"), (r"\bzero risk\b", "managed risk"),
        ]:
            rewrite = re.sub(p, r, rewrite, flags=re.IGNORECASE)
        rewrite += " Past performance is not a guarantee of future results."
    elif violation_type == "exaggerated":
        for p, r in [
            (r"\bthe best\b", "among the leading"), (r"\b(?:best-in-class|best in class)\b", "highly competitive"),
            (r"\btop-performing\b", "strong-performing"), (r"\b#1\b", "a top-ranked"),
            (r"\bunmatched\b", "competitive"), (r"\bunbeatable\b", "competitive"),
            (r"\bindustry-leading\b", "competitive"), (r"\bsuperior\b", "strong"),
            (r"\balways come out ahead\b", "have historically participated in market growth"),
            (r"\bevery single year\b", "in many years"), (r"\bever created\b", "available"),
            (r"\bperfect track record\b", "strong historical performance"),
            (r"\bthe most talented\b", "a highly experienced"),
            (r"\bNo other .+ comes close\b", "We believe our results are competitive"),
        ]:
            rewrite = re.sub(p, r, rewrite, flags=re.IGNORECASE)
    elif violation_type == "unbalanced":
        for p in [r"Don't miss out[!.]?", r"Act now[^.]*\.?", r"Get in now[^.]*\.?",
                   r"Start investing today!?", r"why aren't you investing yet\??",
                   r"You could be next\.?", r"invest today!?"]:
            rewrite = re.sub(p, "", rewrite, flags=re.IGNORECASE)
        rewrite = rewrite.strip()
        if not re.search(r"past performance", rewrite, re.IGNORECASE):
            rewrite += " Past performance does not guarantee future results. All investing involves risk, including possible loss of principal."
    elif violation_type == "misleading":
        for p, r in [
            (r"\bFDIC insured\b", "not FDIC insured"), (r"\bcompletely risk.free\b", "subject to investment risk"),
            (r"\bbacked by the U\.?S\.? government\b", "not guaranteed by any government agency"),
            (r"\b99% accuracy\b", "analytical tools"), (r"\bnever made a wrong\b", "uses data-driven"),
            (r"\bas safe as a savings account\b", "designed to manage risk, though not equivalent to a savings account,"),
            (r"\bendorsed by the SEC\b", "registered with the SEC"),
            (r"\beliminates human error\b", "seeks to reduce behavioral bias"),
            (r"\bcompletely safe\b", "subject to investment risk"),
        ]:
            rewrite = re.sub(p, r, rewrite, flags=re.IGNORECASE)
    why_map = {
        "promissory": "WHY: Uses guarantee/promise language implying certain returns, violating FINRA Rule 2210(d)(1)(B).",
        "exaggerated": "WHY: Makes unsubstantiated superlative claims without basis, violating FINRA Rule 2210(d)(1)(A).",
        "unbalanced": "WHY: Presents performance data without adequate risk disclosure, violating FINRA Rule 2210(d)(1)(A).",
        "misleading": "WHY: Contains false or materially misleading statements, violating FINRA Rule 2210(d)(1)(A) and SEC Rule 206(4)-1.",
    }
    return f"{why_map.get(violation_type, '')}\n\nREWRITE: {rewrite}"


# ══════════════════════════════════════════
# SHAP rendering
# ══════════════════════════════════════════
def render_shap_section(text, explainer, pred_label):
    """Render SHAP tokens + top trigger words."""
    shap_values = explainer([text])
    pred_idx = LABELS.index(pred_label)
    sv = shap_values[0, :, pred_idx]
    tokens, values = sv.data, sv.values
    max_abs = max(abs(values.min()), abs(values.max()), 1e-6)

    # Token HTML
    parts = []
    for token, val in zip(tokens, values):
        norm = val / max_abs
        if norm > 0.15:
            r, g, b = 220, 50, 50
            opacity = min(abs(norm) * 0.85, 0.85)
        elif norm < -0.15:
            r, g, b = 34, 139, 34
            opacity = min(abs(norm) * 0.7, 0.7)
        else:
            r, g, b = 0, 0, 0
            opacity = 0
        bg = f"background-color:rgba({r},{g},{b},{opacity});" if opacity > 0 else ""
        parts.append(f'<span class="shap-token" style="{bg}">{token}</span>')

    st.markdown(f'<div class="shap-container">{" ".join(parts)}</div>', unsafe_allow_html=True)
    st.caption("🔴 Red = pushes toward violation · 🟢 Green = pushes toward compliant")

    # Top triggers
    token_scores = list(zip(sv.data, sv.values))
    top_triggers = sorted(token_scores, key=lambda x: -x[1])[:5]
    top_safe = sorted(token_scores, key=lambda x: x[1])[:3]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🔴 Top violation triggers:**")
        for token, score in top_triggers:
            if score > 0:
                bar_len = int(min(score / max(t[1] for t in top_triggers), 1.0) * 15)
                st.markdown(f"`{token}` {'█' * bar_len} +{score:.3f}")
    with c2:
        st.markdown("**🟢 Top compliance signals:**")
        for token, score in top_safe:
            if score < 0:
                bar_len = int(min(abs(score) / max(abs(t[1]) for t in top_safe), 1.0) * 15)
                st.markdown(f"`{token}` {'█' * bar_len} {score:.3f}")


# ══════════════════════════════════════════
# PAGE 1: Overview
# ══════════════════════════════════════════
def page_overview():
    st.title("🛡️ Financial Compliance Text Classifier")

    st.markdown("""
    An NLP system that detects **FINRA Rule 2210** violations in financial marketing text — 
    classifying *what type* of violation, explaining *which words* triggered the flag, 
    and suggesting *compliant alternatives*. Built as a mini version of 
    [Saifr](https://saifr.ai) (a Fidelity company).
    """)

    st.markdown("---")

    # ── Violation Types (v1 style — tabs with bad/good examples) ──
    st.header("🔍 Violation Types Explained")

    tabs = st.tabs([f"{LABEL_ICONS[l]} {l.title()}" for l in LABEL_DESCRIPTIONS])
    for tab, (label, info) in zip(tabs, LABEL_DESCRIPTIONS.items()):
        with tab:
            st.markdown(f"**{info['short']}** · {info['rule']}")
            st.markdown(info["detail"])
            st.markdown(f"**Common triggers:** `{info['triggers']}`")
            c1, c2 = st.columns(2)
            with c1:
                st.error(f"🚫 **Non-compliant:**\n\n*\"{info['example_bad']}\"*")
            with c2:
                st.success(f"✅ **Compliant:**\n\n*\"{info['example_good']}\"*")

    st.markdown("---")

    # ── Model Benchmark (v1 style — dataframe + quote) ──
    st.header("📊 Model Benchmark")
    st.markdown("Three approaches compared on the same dataset — accuracy vs. speed vs. size.")

    bench_data = {
        "Model": ["TF-IDF + Logistic Regression", "fastText", "FinBERT (fine-tuned)"],
        "Type": ["Classical ML", "Shallow Neural", "Transformer"],
        "F1 Score": ["~0.90", "~0.92", "~0.99"],
        "Training Time": ["< 1 second", "~3 seconds", "~20 minutes"],
        "Model Size": ["~2 MB", "~8 MB", "~440 MB"],
        "Strengths": [
            "Fast, interpretable, great for keyword-based violations",
            "Handles large vocabularies, good for production at scale",
            "Captures semantic nuance between compliant and non-compliant phrasing",
        ],
    }
    st.dataframe(pd.DataFrame(bench_data), use_container_width=True, hide_index=True)

    st.markdown("""
> **Key insight:** Baselines catch obvious violations via keyword matching, but FinBERT captures 
> *semantic nuance* — distinguishing **"consistent returns"** (promissory) from 
> **"seeks to provide consistent income"** (compliant). That's exactly what compliance officers need.
    """)

    st.markdown("---")
    st.info("👉 Head to the **Interactive Checker** in the sidebar to try it live!")


# ══════════════════════════════════════════
# PAGE 2: Interactive Checker
# ══════════════════════════════════════════
def page_checker():
    st.title("🔍 Interactive Compliance Checker")

    if not os.path.exists("./compliance_model_best"):
        st.error("⚠️ Model not found. Run the training notebook first to generate `compliance_model_best/`.")
        st.stop()

    clf = load_model()

    # ── Input ──
    input_method = st.radio("Input method:", ["Select an example", "Type your own"], horizontal=True)

    if input_method == "Select an example":
        selected = st.selectbox("Choose a test case:", list(EXAMPLE_TEXTS.keys()))
        text_input = EXAMPLE_TEXTS[selected]
        st.code(text_input, language=None)
    else:
        text_input = st.text_area(
            "📝 Paste financial marketing text:",
            height=120,
            placeholder="e.g., 'Our fund guarantees consistent returns through any market condition.'",
        )

    analyze_btn = st.button("🔍  Analyze Compliance", type="primary", use_container_width=True)

    # ── Analysis ──
    if analyze_btn and text_input.strip():
        text = text_input.strip()

        with st.spinner("Classifying..."):
            result = sorted(clf(text)[0], key=lambda x: -x["score"])
            top = result[0]

        st.markdown("---")

        # ── 1. Verdict ──
        if top["label"] == "compliant":
            st.success(f"✅ **COMPLIANT** — {top['score']:.1%} confidence")
            st.markdown("This text uses appropriate compliance language with proper hedging and disclosures.")
        else:
            info = LABEL_DESCRIPTIONS[top["label"]]
            st.error(f"🚨 **{top['label'].upper()} VIOLATION** — {top['score']:.1%} confidence")
            st.markdown(f"**{info['short']}** · Violates {info['rule']}")

        # Score bars
        with st.expander("📊 Full classification scores", expanded=False):
            for r in result:
                icon = LABEL_ICONS.get(r["label"], "")
                st.markdown(f"{icon} **{r['label']}** — {r['score']:.1%}")
                st.progress(r["score"])

        # ── 2. Rewrite (fast — shown first) ──
        if top["label"] != "compliant":
            st.markdown("---")
            st.markdown("### ✏️ Compliant Rewrite")

            with st.spinner("Generating compliant alternative..."):
                rewrite, source = get_llm_rewrite(text, top["label"])

            if "REWRITE:" in rewrite:
                parts = rewrite.split("REWRITE:")
                why_part = parts[0].replace("WHY:", "").strip()
                rewrite_part = parts[1].strip()

                st.markdown(f"**Why flagged:** {why_part}")

                c1, c2 = st.columns(2)
                with c1:
                    st.error(f"**🚫 Original**\n\n{text}")
                with c2:
                    st.success(f"**✅ Compliant version**\n\n{rewrite_part}")
            else:
                st.success(rewrite)

            source_labels = {
                "anthropic": "✨ Rewrite by Claude (Anthropic API)",
                "openai": "✨ Rewrite by GPT-4o-mini (OpenAI API)",
                "rule-based": "📝 Rule-based rewrite",
            }
            st.caption(source_labels.get(source, ""))

            # ── 3. SHAP (slower — shown after rewrite) ──
            st.markdown("---")
            st.markdown("### 🔬 Why Was This Flagged?")
            st.markdown("SHAP token attribution shows which words pushed the model toward a violation.")

            try:
                with st.spinner("Computing SHAP explanations..."):
                    render_shap_section(text, load_explainer(), top["label"])
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}")

    elif analyze_btn:
        st.warning("Please enter some text to analyze.")

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("---")
        st.markdown("#### 🔑 LLM Rewrite Status")
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        if has_anthropic:
            st.success("✅ Anthropic API connected")
        elif has_openai:
            st.success("✅ OpenAI API connected")
        else:
            st.warning("No API key — rule-based rewrites")
            with st.expander("Setup"):
                st.markdown("Create `.env` next to `app.py`:\n```\nANTHROPIC_API_KEY=sk-ant-...\n```\nThen `pip install anthropic` and restart.")


# ══════════════════════════════════════════
# Navigation
# ══════════════════════════════════════════
PAGES = {
    "📋 Overview": page_overview,
    "🔍 Interactive Checker": page_checker,
}

with st.sidebar:
    st.markdown("## 🛡️ Compliance Checker")
    page = st.radio("Navigate:", list(PAGES.keys()), label_visibility="collapsed")

PAGES[page]()
