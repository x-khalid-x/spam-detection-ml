import streamlit as st
import pickle
import plotly.graph_objects as go
import plotly.express as px
from preprocess import clean_text

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #F7F8FC;
}

/* Hide default header */
header {visibility: hidden;}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1A1F36 0%, #2D3561 100%);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🛡️";
    font-size: 180px;
    position: absolute;
    right: 40px;
    top: -20px;
    opacity: 0.08;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1.05rem;
    opacity: 0.7;
    margin: 0;
}

/* Stat cards */
.stat-card {
    background: white;
    border-radius: 16px;
    padding: 24px 28px;
    border: 1px solid #E8EAF0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    text-align: center;
}
.stat-number {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1A1F36;
    line-height: 1;
    margin-bottom: 6px;
}
.stat-label {
    font-size: 0.82rem;
    color: #8B90A0;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Input area */
.input-card {
    background: white;
    border-radius: 16px;
    padding: 28px;
    border: 1px solid #E8EAF0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    margin-bottom: 24px;
}

/* Result cards */
.result-spam {
    background: linear-gradient(135deg, #FFF1F0, #FFE4E1);
    border: 1.5px solid #FFB3AD;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 10px;
}
.result-ham {
    background: linear-gradient(135deg, #F0FFF4, #DCFCE7);
    border: 1.5px solid #86EFAC;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 10px;
}
.result-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.result-msg {
    font-size: 0.95rem;
    color: #1A1F36;
    font-family: 'DM Mono', monospace;
}

/* History card */
.history-item-spam {
    background: white;
    border-left: 4px solid #FF4444;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin-bottom: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
.history-item-ham {
    background: white;
    border-left: 4px solid #22C55E;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin-bottom: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
.history-text {
    font-size: 0.9rem;
    color: #1A1F36;
    margin-bottom: 4px;
}
.history-meta {
    font-size: 0.78rem;
    color: #8B90A0;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1A1F36, #2D3561) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 36px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}

/* Section titles */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1A1F36;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Textarea */
.stTextArea textarea {
    border-radius: 10px !important;
    border: 1.5px solid #E8EAF0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    background: #FAFBFF !important;
    color: #1A1F36 !important;
}
.stTextArea textarea:focus {
    border-color: #2D3561 !important;
    box-shadow: 0 0 0 3px rgba(45,53,97,0.08) !important;
}

/* Divider */
.divider {
    height: 1px;
    background: #E8EAF0;
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model ─────────────────────────────────────────────────────────────
with open("models/spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ─── Session State ──────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛡️ Spam Detector</h1>
    <p>Analyse instantanée de vos messages — propulsé par Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ─── Stats Row ──────────────────────────────────────────────────────────────
total = len(st.session_state.history)
spam_count = sum(1 for h in st.session_state.history if h["label"] == "SPAM")
ham_count = total - spam_count
avg_prob = (sum(h["prob"] for h in st.session_state.history) / total * 100) if total > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{total}</div>
        <div class="stat-label">Messages analysés</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number" style="color:#FF4444">{spam_count}</div>
        <div class="stat-label">Spams détectés</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number" style="color:#22C55E">{ham_count}</div>
        <div class="stat-label">Messages légitimes</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{avg_prob:.0f}%</div>
        <div class="stat-label">Prob. spam moyenne</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ─── Main Layout ────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 1], gap="large")

with left_col:
    st.markdown('<div class="section-title">✍️ Analyser un message</div>', unsafe_allow_html=True)
    text = st.text_area("Messages (un par ligne)", height=180, placeholder="Ex: Congratulations! You've won a free prize...\nHey, are we meeting tomorrow?")

    if st.button("🔍 Analyser"):
        if text.strip() == "":
            st.warning("Veuillez entrer au moins un message.")
        else:
            messages = [m.strip() for m in text.split("\n") if m.strip()]
            cleaned = [clean_text(m) for m in messages]
            vect = vectorizer.transform(cleaned)

            from scipy.special import expit
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(vect)
            else:
                probs = expit(model.decision_function(vect))
                probs = [[1 - p, p] for p in probs]

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Résultats</div>', unsafe_allow_html=True)

            for i, (msg, prob) in enumerate(zip(messages, probs)):
                spam_prob = prob[1]
                label = "SPAM" if spam_prob > 0.5 else "NON-SPAM"
                color_class = "result-spam" if label == "SPAM" else "result-ham"
                icon = "🚨" if label == "SPAM" else "✅"
                label_color = "#FF4444" if label == "SPAM" else "#22C55E"

                st.markdown(f"""
                <div class="{color_class}">
                    <div class="result-label" style="color:{label_color}">{icon} {label} — {spam_prob*100:.1f}% de probabilité spam</div>
                    <div class="result-msg">{msg[:120]}{'...' if len(msg)>120 else ''}</div>
                </div>""", unsafe_allow_html=True)

                st.session_state.history.insert(0, {
                    "msg": msg, "label": label,
                    "prob": spam_prob
                })

with right_col:
    st.markdown('<div class="section-title">📈 Statistiques</div>', unsafe_allow_html=True)

    if total > 0:
        # Donut chart
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Spam", "Légitime"],
            values=[spam_count, ham_count],
            hole=0.65,
            marker=dict(colors=["#FF4444", "#22C55E"]),
            textinfo="percent",
            hovertemplate="%{label}: %{value} messages<extra></extra>",
        )])
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=10, b=10, l=10, r=10),
            height=240,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"<b>{total}</b><br>total", x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # Bar chart of last 8 probabilities
        if len(st.session_state.history) >= 2:
            last = st.session_state.history[:8][::-1]
            labels_short = [h["msg"][:18] + "…" if len(h["msg"]) > 18 else h["msg"] for h in last]
            probs_vals = [round(h["prob"] * 100, 1) for h in last]
            colors = ["#FF4444" if h["label"] == "SPAM" else "#22C55E" for h in last]

            fig_bar = go.Figure(go.Bar(
                x=labels_short,
                y=probs_vals,
                marker_color=colors,
                hovertemplate="%{x}<br>Prob spam: %{y}%<extra></extra>",
            ))
            fig_bar.update_layout(
                yaxis=dict(title="Prob. spam (%)", range=[0, 100], gridcolor="#F0F0F0"),
                xaxis=dict(tickangle=-25),
                margin=dict(t=10, b=60, l=10, r=10),
                height=220,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.markdown('<div class="section-title" style="margin-top:8px">🕐 Derniers messages</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#8B90A0;">
            <div style="font-size:3rem">📭</div>
            <div style="margin-top:12px; font-size:0.95rem">Analysez un message pour voir les statistiques</div>
        </div>""", unsafe_allow_html=True)

# ─── History ────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    col_title, col_clear = st.columns([4, 1])
    with col_title:
        st.markdown('<div class="section-title">🗂️ Historique complet</div>', unsafe_allow_html=True)
    with col_clear:
        if st.button("🗑️ Effacer"):
            st.session_state.history = []
            st.rerun()

    for h in st.session_state.history:
        css_class = "history-item-spam" if h["label"] == "SPAM" else "history-item-ham"
        icon = "🚨" if h["label"] == "SPAM" else "✅"
        color = "#FF4444" if h["label"] == "SPAM" else "#22C55E"
        st.markdown(f"""
        <div class="{css_class}">
            <div class="history-text">{h['msg'][:100]}{'...' if len(h['msg'])>100 else ''}</div>
            <div class="history-meta">{icon} <span style="color:{color};font-weight:600">{h['label']}</span> — probabilité spam : {h['prob']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

