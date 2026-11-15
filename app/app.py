"""Network Intrusion Detector - Streamlit demo.
Run from the project folder:  streamlit run app/app.py
"""
import base64
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Inter", "DejaVu Sans"]

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.preprocessing import signed_log1p

st.set_page_config(page_title="Network Intrusion Detector", page_icon="🛡️", layout="wide")

THRESHOLD = 0.50   # the same alert line used for all reported results

# very faint repeating network pattern used in the page background
PATTERN = (
    "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='280' height='280'>"
    "<g stroke='%2338BDF8' stroke-opacity='.055' stroke-width='1' fill='none'>"
    "<path d='M40 50 L130 110 L220 60'/><path d='M130 110 L90 200'/><path d='M130 110 L200 220 L250 150 L220 60'/>"
    "</g><g fill='%2338BDF8' fill-opacity='.14'>"
    "<circle cx='40' cy='50' r='2.5'/><circle cx='130' cy='110' r='3'/><circle cx='220' cy='60' r='2.5'/>"
    "<circle cx='90' cy='200' r='2.5'/><circle cx='200' cy='220' r='2.5'/><circle cx='250' cy='150' r='2.5'/>"
    "</g></svg>\")"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {--ui-font:"Inter","Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,sans-serif;}
.stApp, .stApp button, .stApp input, .stApp textarea, .stApp [data-baseweb] {font-family:var(--ui-font) !important;}
#MainMenu, footer, .stAppDeployButton {display:none !important;}
.stApp {background:
    radial-gradient(900px 500px at 86% 6%, rgba(56,189,248,.12), transparent 60%),
    radial-gradient(800px 620px at 6% 34%, rgba(99,102,241,.13), transparent 60%),
    radial-gradient(1300px 560px at 50% -8%, #17335F 0%, transparent 62%),
    __PATTERN__,
    linear-gradient(rgba(148,163,184,.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148,163,184,.025) 1px, transparent 1px),
    #0B1220;
  background-size:auto, auto, auto, 280px 280px, 56px 56px, 56px 56px, auto;
  background-attachment:fixed;}
[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:1240px; padding-top:1.6rem; padding-bottom:3rem;}
.hero {position:relative; overflow:hidden; text-align:center; color:#fff; padding:50px 34px 52px;
       border-radius:22px; margin-bottom:22px; border:1px solid rgba(255,255,255,.14);
       background:radial-gradient(circle at 50% -30%, rgba(255,255,255,.20), transparent 55%),
                  linear-gradient(120deg,#1E3A8A 0%,#2563EB 55%,#0EA5E9 100%);
       box-shadow:0 14px 38px rgba(37,99,235,.30);}
.hero > * {position:relative; z-index:1;}
.hero .hero-net {position:absolute; top:0; height:100%; width:34%; opacity:.28; z-index:0;}
.hero .hn-right {right:-2%;}
.hero .hn-left {left:-2%; transform:scaleX(-1);}
.hero h1 {color:#fff; font-size:2.75rem; font-weight:700; letter-spacing:-.025em; line-height:1.15;
          margin:0 0 12px 0; padding:0; text-shadow:0 2px 14px rgba(15,23,42,.28);}
.hero p {color:#DBEAFE; font-size:1.1rem; font-weight:400; letter-spacing:.005em; margin:0;}
.st-key-page_nav [data-testid="stButtonGroup"] {justify-content:center;}
.st-key-page_nav button {min-width:200px; justify-content:center;}
.field-label {font-size:.78rem; letter-spacing:.09em; text-transform:uppercase; color:#94A3B8; font-weight:700; margin:6px 0 8px 0;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:18px;}
[data-baseweb="select"] > div {min-height:54px; border-radius:12px; font-size:1.05rem;}
[data-testid="stButtonGroup"] {gap:16px !important; flex-wrap:wrap;}
[data-testid="stButtonGroup"] button {min-height:54px; padding:0 28px; font-size:1.05rem; border-radius:12px !important;
                                      margin:0 8px 0 0 !important;}
[data-testid="stFileUploader"] section {padding:14px 18px; border-radius:12px;}
.verdict {display:flex; justify-content:space-between; align-items:center; gap:24px; padding:28px 36px;
          border-radius:18px; border:1px solid; margin:20px 0 10px 0;}
.verdict.attack {border-color:rgba(239,68,68,.35); border-left:8px solid #EF4444;
                 background:linear-gradient(90deg,rgba(239,68,68,.18),#111A2E 72%);}
.verdict.benign {border-color:rgba(34,197,94,.30); border-left:8px solid #22C55E;
                 background:linear-gradient(90deg,rgba(34,197,94,.15),#111A2E 72%);}
.v-label {font-size:.76rem; letter-spacing:.09em; font-weight:600; color:#94A3B8; text-transform:uppercase;}
.v-main {font-size:2.15rem; font-weight:800; line-height:1.15; margin-top:3px;}
.attack .v-main {color:#F87171;}
.benign .v-main {color:#4ADE80;}
.v-sub {color:#CBD5E1; font-size:1rem; margin-top:5px;}
.v-prob {font-size:2.15rem; font-weight:800; color:#F8FAFC; margin-top:3px;}
.bar {width:280px; height:10px; background:#22304F; border-radius:99px; overflow:hidden; margin-top:6px;}
.bar > div {height:100%; border-radius:99px;}
.attack .bar > div {background:#EF4444;}
.benign .bar > div {background:#22C55E;}
.chips {text-align:center; margin-bottom:8px;}
.chip {display:inline-block; padding:6px 16px; border-radius:999px; font-size:.88rem; font-weight:600; margin:0 6px 6px 6px;}
.chip.ok {background:rgba(34,197,94,.18); color:#86EFAC;}
.chip.bad {background:rgba(239,68,68,.18); color:#FCA5A5;}
.chip.info {background:rgba(56,189,248,.16); color:#7DD3FC;}
.tiles {display:grid; grid-template-columns:repeat(var(--n,4),minmax(0,1fr)); gap:14px; margin:8px 0 18px 0;}
@media (max-width:900px) {.tiles {grid-template-columns:repeat(2,minmax(0,1fr));}}
.tile {background:rgba(17,26,46,.92); border:1px solid #22304F; border-radius:14px; padding:18px 20px; min-height:112px;}
.t-label {font-size:.72rem; letter-spacing:.05em; text-transform:uppercase; color:#94A3B8; font-weight:600;
          white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.t-value {font-size:1.75rem; font-weight:800; color:#F8FAFC; margin-top:6px; white-space:nowrap;
          overflow:hidden; text-overflow:ellipsis;}
.t-note {font-size:.8rem; color:#64748B; margin-top:3px; min-height:1.1em;}
.section {font-size:1.3rem; font-weight:700; color:#F1F5F9; margin:30px 0 12px 0;}
.hint {color:#94A3B8; font-size:.9rem; margin:-4px 0 12px 0;}
.explain {display:grid; grid-template-columns:1.5fr 1fr; gap:16px; align-items:stretch;}
.two {display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:stretch;}
@media (max-width:900px) {.explain, .two {grid-template-columns:1fr;}}
.chart-card {background:rgba(17,26,46,.92); border:1px solid #22304F; border-radius:16px; padding:18px 22px;}
.chart-title {font-weight:700; color:#F1F5F9; font-size:1.05rem; margin-bottom:8px;}
.chart-card img {width:100%; height:auto; display:block;}
.empty {color:#94A3B8; padding:60px 10px; text-align:center;}
.reasons {display:flex; flex-direction:column; gap:10px;}
.reason {flex:1; display:flex; justify-content:space-between; align-items:center; gap:12px; padding:12px 18px;
         background:rgba(17,26,46,.92); border:1px solid #22304F; border-left:4px solid; border-radius:12px;}
.reason.up {border-left-color:#EF4444;}
.reason.down {border-left-color:#22C55E;}
.r-name {font-weight:600; color:#F1F5F9; font-size:.97rem;}
.r-desc {color:#94A3B8; font-size:.8rem; margin-top:2px;}
.r-tag {text-align:center; font-size:.72rem; font-weight:700; padding:4px 10px; border-radius:999px; white-space:nowrap;}
.r-tag.up {background:rgba(239,68,68,.18); color:#FCA5A5;}
.r-tag.down {background:rgba(34,197,94,.18); color:#86EFAC;}
.note {background:rgba(245,158,11,.10); border:1px solid rgba(245,158,11,.35); color:#FCD34D; border-radius:14px;
       padding:16px 22px; font-size:.93rem; margin-top:16px;}
.note b {color:#FDE68A; font-size:1rem;}
.note ul {margin:8px 0 0 18px; padding:0;}
.note li {margin:5px 0;}
</style>
""".replace("__PATTERN__", PATTERN)
st.markdown(CSS, unsafe_allow_html=True)

DESC = {
    "Destination Port": "port the traffic is sent to",
    "Bwd Packet Length Std": "variation in reply-packet sizes",
    "Bwd Packet Length Mean": "average reply-packet size",
    "Init_Win_bytes_forward": "TCP window size, sender side",
    "Init_Win_bytes_backward": "TCP window size, receiver side",
    "Average Packet Size": "average packet size",
    "Fwd IAT Min": "shortest gap between sent packets",
    "Flow IAT Min": "shortest gap between any two packets",
    "Fwd Header Length": "header bytes sent",
    "Bwd Header Length": "header bytes in replies",
    "Flow Bytes/s": "data rate of the flow",
    "Bwd Packets/s": "reply packets per second",
    "Flow Duration": "how long the flow lasted",
    "Total Length of Bwd Packets": "total reply data bytes",
    "Total Length of Fwd Packets": "total data bytes sent",
    "Idle Min": "shortest idle period",
    "Bwd IAT Total": "total time between reply packets",
    "ACK Flag Count": "number of ACK flags",
    "Fwd Packet Length Max": "largest packet sent",
    "Packet Length Std": "variation in packet sizes",
    "act_data_pkt_fwd": "sent packets carrying data",
    "Down/Up Ratio": "download-to-upload ratio",
}


def html(s):
    """Render HTML (lines are joined so markdown does not mangle it)."""
    st.markdown("".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)


def fmt(v):
    if pd.isna(v):
        return "missing"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"


@st.cache_resource
def load_assets():
    bin_booster = xgb.Booster()
    bin_booster.load_model(str(ROOT / "models" / "xgb_binary.json"))
    multi_booster = xgb.Booster()
    multi_booster.load_model(str(ROOT / "models" / "xgb_multiclass.json"))
    with open(ROOT / "models" / "model_features.json") as f:
        features = json.load(f)
    with open(ROOT / "models" / "label_mapping.json") as f:
        mapping = json.load(f)
    class_names = [c for c, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
    explainer = shap.TreeExplainer(bin_booster)
    return bin_booster, multi_booster, features, class_names, explainer


bin_booster, multi_booster, FEATURES, CLASS_NAMES, explainer = load_assets()


@st.cache_data
def load_demo(filename):
    return pd.read_csv(ROOT / "app" / filename)


def to_log(df):
    """Raw clue values -> signed-log values (same transform as in training)."""
    x = df[FEATURES].to_numpy(dtype="float64")
    return signed_log1p(x).astype("float32")


def predict(df, threshold):
    """Binary model decides ATTACK/BENIGN; the 10-class model names the attack type."""
    x = xgb.DMatrix(to_log(df))
    p_attack = bin_booster.predict(x)
    p_multi = multi_booster.predict(x)
    attack_part = p_multi[:, 1:]
    attack_norm = attack_part / (attack_part.sum(axis=1, keepdims=True) + 1e-12)
    attack_type = np.array(CLASS_NAMES[1:])[attack_norm.argmax(axis=1)]
    is_attack = p_attack >= threshold
    return pd.DataFrame({
        "verdict": np.where(is_attack, "ATTACK", "BENIGN"),
        "attack_probability": p_attack.round(4),
        "likely_attack_type": np.where(is_attack, attack_type, "-"),
        "type_confidence": np.where(is_attack, attack_norm.max(axis=1).round(3), np.nan),
    })


def explain(row_df, top_n=6):
    sv = explainer.shap_values(to_log(row_df))[0]
    table = pd.DataFrame({"clue": FEATURES,
                          "raw value": row_df[FEATURES].iloc[0].values,
                          "push": sv})
    order = table["push"].abs().sort_values(ascending=False).index
    return table.loc[order].head(top_n).reset_index(drop=True)


@st.cache_data(show_spinner="Analysing the file...")
def global_importance(feat_df):
    """Average size of each clue's push over (up to 1,000) flows of the uploaded file."""
    sample = feat_df.sample(min(len(feat_df), 1000), random_state=0)
    sv = explainer.shap_values(to_log(sample))
    return pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False).head(8)


def _png_b64(fig, bg):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def chart_b64(table):
    """Large, sharp bar chart (dark style) as a base64 PNG for the card."""
    bg = "#111A2E"
    t = table.iloc[::-1]
    labels = [f"{c}  =  {fmt(v)}" for c, v in zip(t["clue"], t["raw value"])]
    fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor=bg)
    ax.set_facecolor(bg)
    colors = np.where(t["push"] > 0, "#EF4444", "#22C55E").tolist()
    ax.barh(labels, t["push"], color=colors, height=0.62)
    ax.axvline(0, color="#64748B", linewidth=1.2)
    lo, hi = min(float(t["push"].min()), 0.0), max(float(t["push"].max()), 0.0)
    span = (hi - lo) or 1.0
    ax.set_xlim(lo - 0.18 * span, hi + 0.18 * span)
    for y, v in enumerate(t["push"]):
        ax.text(v + (0.015 if v >= 0 else -0.015) * span, y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=11, fontweight="bold", color="#E2E8F0")
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.xaxis.grid(True, color="#1E2B47")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelsize=11.5, labelcolor="#E2E8F0")
    ax.tick_params(axis="x", labelsize=10, labelcolor="#94A3B8")
    ax.set_xlabel("How strongly each clue pushed the verdict", fontsize=11, color="#94A3B8", labelpad=8)
    ax.legend(handles=[Patch(color="#22C55E", label="Points to NORMAL"),
                       Patch(color="#EF4444", label="Points to ATTACK")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False,
              fontsize=11, labelcolor="#CBD5E1")
    fig.tight_layout()
    return _png_b64(fig, bg)


def simple_bars_b64(series, color, xlabel, int_labels=False):
    """Dark horizontal bar chart for the overall-analysis cards (largest bar on top)."""
    bg = "#111A2E"
    s = series.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=bg)
    ax.set_facecolor(bg)
    ax.barh(s.index.tolist(), s.values, color=color, height=0.62)
    top = float(s.max()) if len(s) else 1.0
    ax.set_xlim(0, top * 1.14)
    for y, v in enumerate(s.values):
        ax.text(v + top * 0.012, y, f"{int(v)}" if int_labels else f"{v:.2f}", va="center", ha="left",
                fontsize=10.5, fontweight="bold", color="#E2E8F0")
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.xaxis.grid(True, color="#1E2B47")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelsize=10.5, labelcolor="#E2E8F0")
    ax.tick_params(axis="x", labelsize=9.5, labelcolor="#94A3B8")
    ax.set_xlabel(xlabel, fontsize=10.5, color="#94A3B8", labelpad=8)
    fig.tight_layout()
    return _png_b64(fig, bg)


def f1_bars_b64(pc):
    """F1 score per class; classes below 0.90 are drawn in amber."""
    bg = "#111A2E"
    s = pc.set_index("class")["f1"].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=bg)
    ax.set_facecolor(bg)
    ax.barh(s.index.tolist(), s.values, color=["#F59E0B" if v < 0.9 else "#38BDF8" for v in s.values], height=0.62)
    ax.set_xlim(0, 1.13)
    for y, v in enumerate(s.values):
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", ha="left", fontsize=10.5, fontweight="bold", color="#E2E8F0")
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.xaxis.grid(True, color="#1E2B47")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelsize=10.5, labelcolor="#E2E8F0")
    ax.tick_params(axis="x", labelsize=9.5, labelcolor="#94A3B8")
    ax.set_xlabel("F1 score (1.0 = perfect, amber = below 0.90)", fontsize=10.5, color="#94A3B8", labelpad=8)
    fig.tight_layout()
    return _png_b64(fig, bg)


def show_explanation(row_df):
    table = explain(row_df)
    rows = ""
    for _, r in table.iterrows():
        up = r["push"] > 0
        name = r["clue"]
        title = DESC.get(name, name)
        title = title[:1].upper() + title[1:]
        cls = "up" if up else "down"
        rows += (f'<div class="reason {cls}"><div><div class="r-name">{title}</div>'
                 f'<div class="r-desc">{name} = {fmt(r["raw value"])}</div></div>'
                 f'<div class="r-tag {cls}">{"Points to ATTACK" if up else "Points to NORMAL"}</div></div>')
    img = chart_b64(table)
    html('<div class="section">Why this verdict?</div>')
    html(f'<div class="explain">'
         f'<div class="chart-card"><div class="chart-title">What drove this decision</div>'
         f'<img src="data:image/png;base64,{img}"/></div>'
         f'<div class="reasons">{rows}</div></div>')


def verdict_panel(res, true_label=None):
    attack = res["verdict"] == "ATTACK"
    pct = res["attack_probability"] * 100
    if attack:
        sub = (f'Likely type: <b>{res["likely_attack_type"]}</b> '
               f'(type confidence {res["type_confidence"] * 100:.0f}%)')
    else:
        sub = "No attack pattern found in this flow."
    html(f'<div class="verdict {"attack" if attack else "benign"}">'
         f'<div><div class="v-label">Verdict</div>'
         f'<div class="v-main">{"ATTACK DETECTED" if attack else "BENIGN TRAFFIC"}</div>'
         f'<div class="v-sub">{sub}</div></div>'
         f'<div><div class="v-label">Attack probability</div><div class="v-prob">{pct:.1f}%</div>'
         f'<div class="bar"><div style="width:{min(max(pct, 2), 100):.1f}%"></div></div></div></div>')
    if true_label is not None:
        true_attack = true_label != "BENIGN"
        if attack == true_attack:
            if attack and res["likely_attack_type"] != true_label:
                msg, cls = "Attack found, but the type is wrong", "bad"
            else:
                msg, cls = "Correct prediction", "ok"
        elif attack:
            msg, cls = "False alarm (the flow was actually normal)", "bad"
        else:
            msg, cls = "Missed attack", "bad"
        html(f'<div class="chips"><span class="chip info">Actual label: {true_label}</span>'
             f'<span class="chip {cls}">{msg}</span></div>')


def tiles(items, cols=None):
    cells = "".join(f'<div class="tile"><div class="t-label">{l}</div>'
                    f'<div class="t-value">{v}</div><div class="t-note">{n or "&nbsp;"}</div></div>'
                    for l, v, n in items)
    html(f'<div class="tiles" style="--n:{cols or len(items)}">{cells}</div>')


def color_verdict(v):
    if v == "ATTACK":
        return "background-color:rgba(239,68,68,.22);color:#FCA5A5;font-weight:600"
    return "background-color:rgba(34,197,94,.18);color:#86EFAC;font-weight:600"


def f1_style(v):
    a = 0.12 + 0.55 * max(0.0, min(1.0, (v - 0.5) / 0.5))
    return f"background-color:rgba(56,189,248,{a:.2f});color:#F8FAFC;font-weight:600"


def net_svg(cls):
    """Faint network graphic for the banner corners."""
    return ("<svg class='hero-net " + cls + "' viewBox='0 0 300 200' xmlns='http://www.w3.org/2000/svg'>"
            "<g stroke='white' stroke-opacity='.30' stroke-width='1.1' fill='none'>"
            "<path d='M20 150 L90 90 L170 120 L240 50 L285 100'/>"
            "<path d='M90 90 L110 30 L190 20 L240 50'/>"
            "<path d='M170 120 L200 180 L270 160 L285 100'/></g>"
            "<g fill='white' fill-opacity='.55'>"
            "<circle cx='20' cy='150' r='3.5'/><circle cx='90' cy='90' r='4.5'/><circle cx='170' cy='120' r='3.5'/>"
            "<circle cx='240' cy='50' r='4.5'/><circle cx='285' cy='100' r='3'/><circle cx='110' cy='30' r='3'/>"
            "<circle cx='190' cy='20' r='2.5'/><circle cx='200' cy='180' r='3'/><circle cx='270' cy='160' r='2.5'/>"
            "</g></svg>")


# ------------------------------------------------------------------ page
html(f'<div class="hero">{net_svg("hn-left")}{net_svg("hn-right")}'
     '<h1>Network Intrusion Detector</h1>'
     '<p>Detect network attacks, identify their type, and see exactly why the model decided</p></div>')

page = st.segmented_control("Page", ["Detector", "Model performance"], default="Detector",
                            key="page_nav", label_visibility="collapsed") or "Detector"

if "tbl_ver" not in st.session_state:
    st.session_state.tbl_ver = 0

# ------------------------------------------------------------ detector page
if page == "Detector":
    with st.container(border=True):
        c1, c2 = st.columns([1, 1.5], gap="large")
        with c1:
            html('<div class="field-label">Input source</div>')
            source = st.segmented_control("Input source", ["Sample flows", "Upload a file"],
                                          default="Sample flows", label_visibility="collapsed") or "Sample flows"
        with c2:
            if source == "Sample flows":
                demo = load_demo("demo_flows.csv")
                html('<div class="field-label">Sample flow</div>')
                i = st.selectbox("Sample flow", list(range(len(demo))),
                                 format_func=lambda k: f"Sample flow {k + 1}", label_visibility="collapsed")
            else:
                html('<div class="field-label">CSV file</div>')
                up = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
        if source == "Upload a file":
            with st.expander("Show the 48 required columns"):
                st.write(FEATURES)

    if source == "Sample flows":
        row = demo.iloc[[i]]
        res = predict(row, THRESHOLD).iloc[0]
        verdict_panel(res, demo.loc[i, "true_label"])
        show_explanation(row)
    elif up is not None:
        df = pd.read_csv(up)
        df.columns = df.columns.str.strip()
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()
        df = df.replace([np.inf, -np.inf], np.nan)
        res = predict(df, THRESHOLD)
        n = len(res)
        attacks = res[res["verdict"] == "ATTACK"]
        counts = attacks["likely_attack_type"].value_counts()
        has_label = "true_label" in df.columns

        tiles([("Flows checked", f"{n:,}", "rows in your file"),
               ("Flagged as attack", f"{len(attacks):,}", f"{len(attacks) / n * 100:.1f}% of flows"),
               ("Normal flows", f"{n - len(attacks):,}", f"{(n - len(attacks)) / n * 100:.1f}% of flows"),
               ("Top attack type", counts.index[0] if len(counts) else "-", "among flagged flows")])

        # ---- flow-by-flow table
        html('<div class="section">Flow-by-flow results</div>')
        view = st.segmented_control("Show", ["All flows", "Attacks only", "Normal only"],
                                    default="All flows", label_visibility="collapsed") or "All flows"
        disp = pd.DataFrame({
            "Flow #": np.arange(1, n + 1),
            "Verdict": res["verdict"].values,
            "Attack probability": res["attack_probability"].values * 100,
            "Likely attack type": res["likely_attack_type"].values,
            "Type confidence": ["-" if pd.isna(v) else f"{v * 100:.0f}%" for v in res["type_confidence"]],
        })
        if has_label:
            disp["Actual label"] = df["true_label"].values
        if view == "Attacks only":
            disp = disp[disp["Verdict"] == "ATTACK"]
        elif view == "Normal only":
            disp = disp[disp["Verdict"] == "BENIGN"]
        disp = disp.head(1000).reset_index(drop=True)
        hint = "Click a row to see why the model decided that way."
        if n > 1000:
            hint += f" Showing at most 1,000 flows here; the download button gives all {n:,}."
        html(f'<div class="hint">{hint}</div>')

        sel_rows = []
        if len(disp):
            tkey = f"results_{st.session_state.tbl_ver}_{view}_{getattr(up, 'name', 'f')}_{n}"
            event = st.dataframe(
                disp.style.map(color_verdict, subset=["Verdict"]),
                hide_index=True, height=420, key=tkey,
                on_select="rerun", selection_mode="single-row",
                column_config={
                    "Flow #": st.column_config.NumberColumn("Flow #", format="%d"),
                    "Attack probability": st.column_config.ProgressColumn(
                        "Attack probability", min_value=0, max_value=100, format="%.1f%%"),
                })
            sel_rows = list(event.selection.rows)
        else:
            st.info("No flows match this filter.")
        st.download_button("Download all results (CSV)", res.to_csv(index=False), "predictions.csv")

        if sel_rows and sel_rows[0] < len(disp):
            # ---- one flow explained
            k = int(disp.iloc[sel_rows[0]]["Flow #"])
            html(f'<div class="section">Flow {k} explained</div>')
            if st.button("← Back to overall analysis"):
                st.session_state.tbl_ver += 1
                st.rerun()
            verdict_panel(res.iloc[k - 1], df["true_label"].iloc[k - 1] if has_label else None)
            show_explanation(df.iloc[[k - 1]])
        else:
            # ---- overall analysis of the whole file
            html('<div class="section">Overall analysis</div>')
            p = res["attack_probability"].to_numpy()
            confident = int(((p <= 0.10) | (p >= 0.90)).sum())
            oitems = [("Confident verdicts", f"{confident:,}", f"{confident / n * 100:.1f}% of flows"),
                      ("Borderline flows", f"{n - confident:,}", "probability 10% to 90%")]
            if has_label:
                actual_attack = (df["true_label"].astype(str).str.strip() != "BENIGN").to_numpy()
                pred_attack = (res["verdict"] == "ATTACK").to_numpy()
                oitems += [
                    ("Attacks caught", f"{int((actual_attack & pred_attack).sum()):,}",
                     f"of {int(actual_attack.sum()):,} real attacks"),
                    ("Attacks missed", f"{int((actual_attack & ~pred_attack).sum()):,}", "real attacks not flagged"),
                    ("False alarms", f"{int((~actual_attack & pred_attack).sum()):,}", "normal flows flagged")]
            tiles(oitems, cols=5)

            if len(counts):
                left = f'<img src="data:image/png;base64,{simple_bars_b64(counts, "#38BDF8", "Number of flows", True)}"/>'
            else:
                left = '<div class="empty">No attacks were flagged in this file.</div>'
            imp = global_importance(df[FEATURES])
            right = f'<img src="data:image/png;base64,{simple_bars_b64(imp, "#818CF8", "Average push on the verdict (bigger = mattered more)")}"/>'
            html(f'<div class="two">'
                 f'<div class="chart-card"><div class="chart-title">Flagged flows by attack type</div>{left}</div>'
                 f'<div class="chart-card"><div class="chart-title">Clues that mattered most in this file</div>{right}</div>'
                 f'</div>')

# --------------------------------------------------------- performance page
else:
    tables = ROOT / "results" / "tables"
    bpath = tables / "07_test_binary_metrics.csv"
    mpath = tables / "07_test_multiclass_summary.csv"
    cpath = tables / "07_test_multiclass_per_class.csv"
    html('<div class="section">Normal vs attack</div>')
    html('<div class="hint">Measured on 499,596 test flows the model never saw during training.</div>')
    if bpath.exists():
        b = pd.read_csv(bpath).iloc[0]
        tiles([("Attacks caught", f"{b['recall'] * 100:.2f}%", f"recall · {int(b['missed_attacks'])} attacks missed"),
               ("Precision", f"{b['precision'] * 100:.2f}%", "flagged flows that were real"),
               ("False-alarm rate", f"{b['false_alarm_rate'] * 100:.3f}%", f"{int(b['false_alarms'])} normal flows flagged"),
               ("F1 score", f"{b['f1']:.4f}", "balance of recall and precision"),
               ("ROC-AUC", f"{b['roc_auc']:.5f}", "ranking quality")])
    html('<div class="section">Which attack type</div>')
    html('<div class="hint">Normal traffic plus 9 attack types, each measured separately.</div>')
    if mpath.exists():
        m = pd.read_csv(mpath).iloc[0]
        tiles([("Accuracy", f"{m['accuracy'] * 100:.2f}%", "high because most traffic is normal"),
               ("Macro F1", f"{m['macro_f1']:.4f}", "all classes count equally")], cols=5)
    if cpath.exists():
        raw = pd.read_csv(cpath)
        pc = raw.rename(columns={"class": "Class", "precision": "Precision", "recall": "Recall",
                                 "f1": "F1", "test rows": "Test flows"})
        cl, cr = st.columns([1.1, 1], gap="medium")
        with cl:
            st.dataframe(pc.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "Test flows": "{:,}"})
                         .map(f1_style, subset=["F1"]),
                         hide_index=True, height=(len(pc) + 1) * 35 + 3)
        with cr:
            html(f'<div class="chart-card"><div class="chart-title">F1 score by class</div>'
                 f'<img src="data:image/png;base64,{f1_bars_b64(raw)}"/></div>')
    html('<div class="note"><b>Keep in mind</b><ul>'
         '<li>Heartbleed (2 test flows) and Infiltration (7 test flows) are too small to judge reliably.</li>'
         '<li>Bot is the hardest attack type to recognise (F1 about 0.84).</li>'
         '<li>All results come from one lab dataset, and the model relies strongly on destination port and '
         'TCP window size, which may partly reflect the lab setup.</li></ul></div>')