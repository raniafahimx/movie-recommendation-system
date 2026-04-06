"""
╔══════════════════════════════════════════════════════════════╗
║         CineAI — Movie Recommendation System                 ║
║         Streamlit Web Application                            ║
║         Run: streamlit run movie_recommender_app.py          ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="CineAI · Intelligent Movie Recommendations",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# MASTER CSS — CineAI Brand System
# Palette: Purple (#452c63, #1d1160, #33006F, #D8BFD8) + Teal (#004c4c, #006666, #b2d8d8)
# Fonts: Syne (display/logo), DM Sans (body)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── Root palette ── */
:root {
  --purple-deep:    #1d1160;
  --purple-mid:     #33006F;
  --purple-dark:    #452c63;
  --purple-soft:    #D8BFD8;
  --teal-deep:      #004c4c;
  --teal-mid:       #006666;
  --teal-light:     #b2d8d8;
  --teal-faint:     #e0f5f5;
  --white:          #FFFFFF;
  --off-white:      #f8f7fc;
  --text-1:         #1d1160;
  --text-2:         #452c63;
  --text-3:         #7a6a9a;
  --border-light:   rgba(69,44,99,0.15);
  --border-mid:     rgba(69,44,99,0.25);
  --shadow-purple:  rgba(51,0,111,0.14);
  --shadow-teal:    rgba(0,102,102,0.12);
}

/* ── Global resets ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', system-ui, sans-serif !important;
  color: var(--text-1) !important;
}

/* ── Main background — rich purple-to-teal gradient ── */
.stApp {
  background:
    radial-gradient(ellipse at 0% 0%, rgba(69,44,99,0.28) 0%, transparent 55%),
    radial-gradient(ellipse at 100% 0%, rgba(51,0,111,0.22) 0%, transparent 50%),
    radial-gradient(ellipse at 50% 100%, rgba(0,102,102,0.20) 0%, transparent 55%),
    radial-gradient(ellipse at 100% 100%, rgba(0,76,76,0.18) 0%, transparent 45%),
    linear-gradient(160deg, #f5f2fb 0%, #f8f7fc 35%, #f0f5f5 70%, #f2f0f8 100%) !important;
  min-height: 100vh;
}

/* Remove grid overlay */
.stApp::before, .stApp::after { display: none !important; }

/* Main content */
.main .block-container {
  position: relative;
  z-index: 1;
  padding-top: 0 !important;
  max-width: 1280px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1d1160 0%, #2a0858 45%, #1a3333 100%) !important;
  backdrop-filter: blur(20px);
  border-right: 1px solid rgba(216,191,216,0.12) !important;
  box-shadow: 4px 0 40px rgba(29,17,96,0.3) !important;
}
[data-testid="stSidebar"] * {
  color: rgba(216,191,216,0.85) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] b,
[data-testid="stSidebar"] strong {
  color: var(--white) !important;
}
[data-testid="stSidebar"] hr {
  border-color: rgba(216,191,216,0.12) !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div > div {
  background: var(--teal-mid) !important;
}
/* Remove red slider thumb */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
  background: var(--teal-light) !important;
  border-color: var(--teal-mid) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(216,191,216,0.18) !important;
  border-radius: 10px !important;
  color: rgba(216,191,216,0.85) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label {
  color: rgba(216,191,216,0.55) !important;
  font-size: 10px !important;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
}

/* ── Override ALL slider thumbs (remove red) ── */
[data-baseweb="slider"] [role="slider"] {
  background: var(--teal-mid) !important;
  border-color: var(--teal-mid) !important;
  box-shadow: 0 0 0 4px rgba(0,102,102,0.2) !important;
}
.stSlider > div > div > div > div {
  background: var(--teal-mid) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--purple-deep) 0%, var(--purple-mid) 100%) !important;
  color: white !important;
  border: 1px solid rgba(216,191,216,0.2) !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  letter-spacing: 0.2px;
  padding: 10px 24px !important;
  box-shadow: 0 4px 20px rgba(29,17,96,0.25), inset 0 1px 0 rgba(255,255,255,0.10) !important;
  transition: all 0.28s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
  width: 100%;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 32px rgba(51,0,111,0.35), 0 0 0 1px rgba(0,102,102,0.3) !important;
  border-color: rgba(0,102,102,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Labels ── */
.stSelectbox label, .stSlider label, .stTextInput label,
.stNumberInput label, .stMultiSelect label {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  color: var(--text-3) !important;
  text-transform: uppercase;
  letter-spacing: 0.7px;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
  background: white !important;
  border: 1px solid var(--border-light) !important;
  border-radius: 10px !important;
  color: var(--text-1) !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.stSelectbox > div > div:focus-within {
  border-color: var(--teal-mid) !important;
  box-shadow: 0 0 0 3px rgba(0,102,102,0.10) !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input {
  background: white !important;
  border: 1px solid var(--border-light) !important;
  border-radius: 10px !important;
  color: var(--text-1) !important;
  font-family: 'DM Sans', sans-serif !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.stTextInput > div > div > input:focus {
  border-color: var(--teal-mid) !important;
  box-shadow: 0 0 0 3px rgba(0,102,102,0.10) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: rgba(255,255,255,0.78) !important;
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-light) !important;
  border-top: 2px solid var(--teal-mid) !important;
  border-radius: 14px !important;
  padding: 20px 22px !important;
  box-shadow: 0 4px 24px var(--shadow-purple) !important;
  transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.25s ease !important;
}
[data-testid="metric-container"]:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 36px var(--shadow-purple) !important;
}
[data-testid="metric-container"] label {
  font-size: 10px !important;
  color: var(--text-3) !important;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 26px !important;
  background: linear-gradient(135deg, var(--purple-deep), var(--teal-mid));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.72) !important;
  backdrop-filter: blur(14px);
  border: 1px solid var(--border-light) !important;
  border-radius: 14px !important;
  padding: 5px 6px !important;
  gap: 3px !important;
  box-shadow: 0 4px 20px var(--shadow-purple) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 12.5px !important;
  color: var(--text-2) !important;
  padding: 8px 16px !important;
  border: none !important;
  background: transparent !important;
  transition: all 0.2s ease !important;
  letter-spacing: 0.1px;
}
.stTabs [data-baseweb="tab"]:hover {
  background: rgba(69,44,99,0.06) !important;
  color: var(--purple-deep) !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--purple-deep) 0%, var(--purple-mid) 100%) !important;
  color: white !important;
  box-shadow: 0 4px 14px rgba(29,17,96,0.3) !important;
}

/* ── Dataframe — professional table styling ── */
.stDataFrame {
  border: 1px solid var(--border-light) !important;
  border-radius: 14px !important;
  overflow: hidden;
  box-shadow: 0 4px 24px var(--shadow-purple) !important;
}
.stDataFrame thead tr th {
  background: linear-gradient(135deg, var(--purple-deep), var(--purple-mid)) !important;
  color: white !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  padding: 12px 16px !important;
  border: none !important;
}
.stDataFrame tbody tr td {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13px !important;
  color: var(--text-1) !important;
  padding: 11px 16px !important;
  border-bottom: 1px solid rgba(69,44,99,0.06) !important;
}
.stDataFrame tbody tr:nth-child(even) td {
  background: rgba(69,44,99,0.025) !important;
}
.stDataFrame tbody tr:hover td {
  background: rgba(0,102,102,0.05) !important;
}

/* ── Divider ── */
hr { border-color: var(--border-light) !important; opacity: 0.5; }

/* ── Expander ── */
.streamlit-expanderHeader {
  background: rgba(255,255,255,0.65) !important;
  border: 1px solid var(--border-light) !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  color: var(--text-1) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--teal-mid) !important; }

/* ── Alerts ── */
.stAlert { border-radius: 12px !important; }

/* ── Progress bar — remove red ── */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--purple-deep), var(--teal-mid)) !important;
}
/* Streamlit internal progress */
[data-baseweb="progress-bar"] > div {
  background: linear-gradient(90deg, var(--purple-deep), var(--teal-mid)) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(216,191,216,0.15); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(var(--purple-dark), var(--teal-mid));
  border-radius: 3px;
}

/* ══════════════════════════════════════════════════
   SECTION STYLES
══════════════════════════════════════════════════ */
.content-section {
  padding: 36px 0;
  animation: fadeIn 0.5s ease both;
}

.section-heading {
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: 22px;
  color: var(--purple-deep);
  letter-spacing: -0.4px;
  margin: 0 0 6px 0;
}
.section-subheading {
  font-family: 'DM Sans', sans-serif;
  font-size: 13px;
  color: var(--text-3);
  margin: 0 0 20px 0;
}

.tab-description-card {
  background: linear-gradient(135deg, rgba(29,17,96,0.04) 0%, rgba(0,102,102,0.04) 100%);
  border: 1px solid rgba(69,44,99,0.12);
  border-left: 3px solid var(--teal-mid);
  border-radius: 12px;
  padding: 18px 22px;
  margin-bottom: 24px;
}
.tab-description-card b {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: 15px !important;
  color: var(--purple-deep) !important;
}
.tab-description-card span {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13px !important;
  color: var(--text-2) !important;
  line-height: 1.65 !important;
}

/* ── Animations ── */
@keyframes fadeSlideDown {
  from { opacity: 0; transform: translateY(-16px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}

.stTabs [data-baseweb="tab-panel"] > div > div:nth-child(1) { animation: fadeSlideUp 0.4s ease both; }
.stTabs [data-baseweb="tab-panel"] > div > div:nth-child(2) { animation: fadeSlideUp 0.4s ease 0.08s both; }
.stTabs [data-baseweb="tab-panel"] > div > div:nth-child(3) { animation: fadeSlideUp 0.4s ease 0.15s both; }
.stTabs [data-baseweb="tab-panel"] > div > div:nth-child(4) { animation: fadeSlideUp 0.4s ease 0.22s both; }

.stMarkdown h3 {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  color: var(--purple-deep) !important;
  letter-spacing: -0.2px !important;
}
.stMarkdown h4 {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: 16px !important;
  color: var(--text-1) !important;
}

/* ── Footer ── */
.cineai-footer {
  text-align: center;
  padding: 20px 0 32px;
  font-family: 'DM Sans', sans-serif;
  font-size: 12px;
  color: var(--text-3);
  letter-spacing: 0.3px;
  border-top: 1px solid var(--border-light);
  margin-top: 16px;
}
.cineai-footer span {
  color: var(--teal-mid);
  font-weight: 500;
}

/* ── Responsive ── */
@media (max-width: 900px) {
  .main .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HERO SECTION
# Full-width, with in-hero nav that scrolls to tabs
# ══════════════════════════════════════════════════════════════

_HERO_SHIM_CSS = """
<style>
/* Pull the hero iframe to full viewport width with zero margin */
[data-testid="stCustomComponentV1"] {
    width: 100vw !important;
    margin-left: calc(-1 * (100vw - 100%) / 2) !important;
    margin-top: -80px !important;
    display: block;
}
iframe[title="components.html"] {
    display: block;
    margin-top: 0 !important;
}
/* Remove Streamlit top padding that creates gap */
.main .block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
section[data-testid="stAppViewContainer"] > div:first-child {
    padding-top: 0 !important;
}
/* Header bar */
header[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: none !important;
}
</style>
"""

def render_hero():
    st.markdown(_HERO_SHIM_CSS, unsafe_allow_html=True)

    hero_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html, body {
    width: 100%; height: 100%;
    font-family: 'DM Sans', system-ui, sans-serif;
    overflow-x: hidden;
    background: #0e0824;
  }

  .hero {
    position: relative;
    width: 100%;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: #0e0824;
  }

  /* Video */
  #hero-video {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    opacity: 0; z-index: 0;
  }

  /* Overlays */
  .hero-overlay {
    position: absolute; inset: 0; z-index: 1;
    background: linear-gradient(
      to bottom,
      rgba(14,8,36,0.62) 0%,
      rgba(14,8,36,0.30) 40%,
      rgba(14,8,36,0.50) 72%,
      rgba(14,8,36,0.92) 100%
    );
  }
  /* Purple tint overlay */
  .hero-color-overlay {
    position: absolute; inset: 0; z-index: 1;
    background: linear-gradient(
      135deg,
      rgba(69,44,99,0.35) 0%,
      rgba(51,0,111,0.22) 50%,
      rgba(0,76,76,0.18) 100%
    );
    mix-blend-mode: multiply;
  }
  .hero-glow {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -52%);
    width: min(900px, 90vw); height: 520px;
    background: rgba(29,17,96,0.65);
    filter: blur(90px);
    border-radius: 50%;
    z-index: 2; pointer-events: none;
  }

  /* Content layer */
  .hero-content {
    position: relative; z-index: 3;
    display: flex; flex-direction: column;
    min-height: 100vh;
  }

  /* ── Navbar ── */
  nav.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 24px 52px; width: 100%;
  }

  /* Logo — wordmark style, no emoji */
  .nav-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800; font-size: 24px;
    letter-spacing: -0.5px;
    display: flex; align-items: center; gap: 3px;
  }
  .logo-cine {
    color: #ffffff;
  }
  .logo-ai {
    color: #b2d8d8;
  }
  .logo-dot {
    width: 6px; height: 6px;
    background: #006666;
    border-radius: 50%;
    margin-left: 1px;
    margin-bottom: 14px;
    flex-shrink: 0;
    display: inline-block;
    vertical-align: baseline;
  }

  .nav-links {
    display: flex; align-items: center; gap: 4px;
  }
  .nav-link {
    font-family: 'DM Sans', sans-serif;
    font-weight: 400; font-size: 14px;
    color: rgba(255,255,255,0.78);
    padding: 8px 16px; border-radius: 8px;
    background: transparent; border: none;
    cursor: pointer;
    transition: background 0.18s ease, color 0.18s ease;
  }
  .nav-link:hover { background: rgba(255,255,255,0.08); color: #fff; }

  .nav-cta {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500; font-size: 13px;
    color: #1d1160; background: #D8BFD8;
    padding: 9px 22px; border-radius: 50px; border: none;
    cursor: pointer; letter-spacing: 0.1px;
    box-shadow: 0 4px 16px rgba(69,44,99,0.35);
    transition: all 0.22s cubic-bezier(0.34,1.56,0.64,1);
  }
  .nav-cta:hover {
    background: #b2d8d8;
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(0,76,76,0.3);
  }

  .nav-divider {
    height: 1px; margin: 0 52px;
    background: linear-gradient(90deg, transparent, rgba(216,191,216,0.20), transparent);
  }

  /* ── Hero body ── */
  .hero-body {
    flex: 1; display: flex;
    align-items: center; justify-content: center;
    padding: 40px 24px;
  }
  .hero-inner { text-align: center; max-width: 880px; width: 100%; }

  /* Eyebrow */
  .eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 11px; font-weight: 500;
    letter-spacing: 1.8px; text-transform: uppercase;
    color: rgba(178,216,216,0.85);
    background: rgba(178,216,216,0.08);
    border: 1px solid rgba(178,216,216,0.22);
    padding: 6px 16px; border-radius: 50px;
    margin-bottom: 28px;
    animation: fadeSlideDown 0.7s cubic-bezier(0.22,1,0.36,1) both;
  }
  .eyebrow-dot {
    width: 6px; height: 6px;
    background: #006666; border-radius: 50%;
    animation: pulseDot 2s ease-in-out infinite;
  }

  /* Headline */
  h1.headline {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(56px, 9.5vw, 116px);
    line-height: 1.0; letter-spacing: -0.03em;
    color: #fff; margin-bottom: 4px;
    animation: fadeSlideUp 0.75s cubic-bezier(0.22,1,0.36,1) 0.15s both;
  }
  .headline-accent {
    background: linear-gradient(120deg, #D8BFD8 0%, #fff 38%, #b2d8d8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  /* Subtitle */
  p.subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: clamp(15px, 2vw, 18px); font-weight: 300;
    color: rgba(216,191,216,0.72);
    line-height: 1.72; max-width: 520px;
    margin: 18px auto 0;
    animation: fadeSlideUp 0.75s cubic-bezier(0.22,1,0.36,1) 0.28s both;
  }

  /* CTA row */
  .cta-row {
    display: flex; align-items: center; justify-content: center;
    gap: 14px; margin-top: 36px; flex-wrap: wrap;
    animation: fadeSlideUp 0.75s cubic-bezier(0.22,1,0.36,1) 0.40s both;
  }
  .btn-primary {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500; font-size: 14px;
    color: #1d1160; background: #D8BFD8;
    padding: 14px 32px; border-radius: 12px; border: none;
    cursor: pointer; letter-spacing: 0.1px;
    box-shadow: 0 4px 20px rgba(69,44,99,0.35), inset 0 1px 0 rgba(255,255,255,0.5);
    transition: all 0.28s cubic-bezier(0.34,1.56,0.64,1);
    text-decoration: none; display: inline-block;
  }
  .btn-primary:hover {
    background: #b2d8d8;
    color: #004c4c;
    transform: translateY(-2px);
    box-shadow: 0 10px 32px rgba(0,76,76,0.3);
  }
  .btn-secondary {
    font-family: 'DM Sans', sans-serif;
    font-weight: 400; font-size: 14px;
    color: rgba(255,255,255,0.82);
    background: rgba(255,255,255,0.06);
    padding: 14px 28px; border-radius: 12px;
    border: 1px solid rgba(216,191,216,0.18);
    cursor: pointer; letter-spacing: 0.1px;
    display: inline-flex; align-items: center; gap: 8px;
    transition: all 0.2s ease;
    text-decoration: none;
  }
  .btn-secondary:hover {
    background: rgba(255,255,255,0.10);
    border-color: rgba(178,216,216,0.35);
    color: #fff;
  }

  /* Stats */
  .stats {
    display: flex; align-items: center; justify-content: center;
    gap: 48px; margin-top: 52px;
    animation: fadeIn 1s cubic-bezier(0.22,1,0.36,1) 0.60s both;
  }
  .stat { text-align: center; }
  .stat-value {
    font-family: 'Syne', sans-serif;
    font-weight: 700; font-size: 30px;
    color: #fff; letter-spacing: -0.5px; line-height: 1;
  }
  .stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px; color: rgba(178,216,216,0.58);
    letter-spacing: 0.8px; text-transform: uppercase; margin-top: 4px;
  }
  .stat-divider { width: 1px; height: 36px; background: rgba(216,191,216,0.18); }

  /* ── Marquee ── */
  .marquee-section {
    padding: 28px 52px 38px;
    animation: fadeIn 1s ease 0.8s both;
  }
  .marquee-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px; color: rgba(178,216,216,0.38);
    letter-spacing: 1px; text-transform: uppercase;
    text-align: center; margin-bottom: 16px;
  }
  .marquee-wrapper {
    overflow: hidden;
    -webkit-mask-image: linear-gradient(90deg, transparent 0%, black 14%, black 86%, transparent 100%);
    mask-image: linear-gradient(90deg, transparent 0%, black 14%, black 86%, transparent 100%);
  }
  .marquee-track {
    display: flex; gap: 40px;
    animation: marqueeScroll 28s linear infinite;
    width: max-content;
  }
  .marquee-item {
    display: flex; align-items: center; gap: 10px;
    color: rgba(255,255,255,0.48);
    font-family: 'DM Sans', sans-serif;
    font-size: 14px; font-weight: 500; letter-spacing: 0.2px;
    white-space: nowrap;
    transition: color 0.18s ease;
  }
  .marquee-item:hover { color: rgba(255,255,255,0.85); }
  .m-icon {
    width: 28px; height: 28px;
    background: rgba(178,216,216,0.08);
    border: 1px solid rgba(178,216,216,0.15);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700;
    font-family: 'Syne', sans-serif;
    color: #b2d8d8;
    flex-shrink: 0;
  }

  /* ── Keyframes ── */
  @keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-14px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
  @keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.45; transform: scale(0.68); }
  }
  @keyframes marqueeScroll {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
  }

  /* ── Responsive ── */
  @media (max-width: 768px) {
    nav.navbar { padding: 18px 22px; }
    .nav-divider { margin: 0 22px; }
    .marquee-section { padding: 22px 22px 30px; }
    .stats { gap: 22px; flex-wrap: wrap; }
  }
  @media (max-width: 600px) {
    .nav-links { display: none; }
    .stat-divider { display: none; }
    h1.headline { font-size: clamp(40px, 12vw, 64px); }
  }
</style>
</head>
<body>
<section class="hero">
  <video id="hero-video"
    src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260328_065045_c44942da-53c6-4804-b734-f9e07fc22e08.mp4"
    muted playsinline preload="auto" aria-hidden="true"></video>

  <div class="hero-overlay"></div>
  <div class="hero-color-overlay"></div>
  <div class="hero-glow"></div>

  <div class="hero-content">
    <nav class="navbar">
      <div class="nav-logo">
        <span class="logo-cine">Cine</span><span class="logo-ai">AI</span><span class="logo-dot"></span>
      </div>
      <div class="nav-links">
        <button class="nav-link" onclick="scrollToSection('overview')">Overview</button>
        <button class="nav-link" onclick="scrollToSection('models')">Models</button>
        <button class="nav-link" onclick="scrollToSection('evaluation')">Evaluation</button>
        <button class="nav-link" onclick="scrollToSection('dataset')">Dataset</button>
      </div>
      <button class="nav-cta" onclick="scrollToSection('overview')">Get Started</button>
    </nav>
    <div class="nav-divider"></div>

    <div class="hero-body">
      <div class="hero-inner">
        <div class="eyebrow">
          <span class="eyebrow-dot"></span>
          MovieLens 100K &middot; ML Recommendation Engine
        </div>

        <h1 class="headline">
          Discover<br>
          <span class="headline-accent">Cinema</span>
        </h1>

        <p class="subtitle">
          Powered by User-CF, Item-CF &amp; SVD Matrix Factorization &mdash;
          intelligent film recommendations built on real viewing patterns.
        </p>

        <div class="cta-row">
          <button class="btn-primary" onclick="scrollToSection('overview')">Explore Recommendations</button>
          <button class="btn-secondary" onclick="scrollToSection('models')">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8" fill="currentColor" stroke="none"/></svg>
            How It Works
          </button>
        </div>

        <div class="stats">
          <div class="stat">
            <div class="stat-value">9,742</div>
            <div class="stat-label">Movies</div>
          </div>
          <div class="stat-divider"></div>
          <div class="stat">
            <div class="stat-value">100K+</div>
            <div class="stat-label">Ratings</div>
          </div>
          <div class="stat-divider"></div>
          <div class="stat">
            <div class="stat-value">610</div>
            <div class="stat-label">Users</div>
          </div>
          <div class="stat-divider"></div>
          <div class="stat">
            <div class="stat-value">3</div>
            <div class="stat-label">ML Models</div>
          </div>
        </div>
      </div>
    </div>

    <div class="marquee-section">
      <div class="marquee-label">Powered by industry-grade algorithms</div>
      <div class="marquee-wrapper">
        <div class="marquee-track">
          <div class="marquee-item"><div class="m-icon">U</div>User-CF</div>
          <div class="marquee-item"><div class="m-icon">I</div>Item-CF</div>
          <div class="marquee-item"><div class="m-icon">S</div>SVD Factorization</div>
          <div class="marquee-item"><div class="m-icon">P</div>Precision@K</div>
          <div class="marquee-item"><div class="m-icon">C</div>Cosine Similarity</div>
          <div class="marquee-item"><div class="m-icon">M</div>MovieLens</div>
          <div class="marquee-item"><div class="m-icon">L</div>Latent Factors</div>
          <div class="marquee-item"><div class="m-icon">R</div>Recall@K</div>
          <div class="marquee-item"><div class="m-icon">U</div>User-CF</div>
          <div class="marquee-item"><div class="m-icon">I</div>Item-CF</div>
          <div class="marquee-item"><div class="m-icon">S</div>SVD Factorization</div>
          <div class="marquee-item"><div class="m-icon">P</div>Precision@K</div>
          <div class="marquee-item"><div class="m-icon">C</div>Cosine Similarity</div>
          <div class="marquee-item"><div class="m-icon">M</div>MovieLens</div>
          <div class="marquee-item"><div class="m-icon">L</div>Latent Factors</div>
          <div class="marquee-item"><div class="m-icon">R</div>Recall@K</div>
        </div>
      </div>
    </div>
  </div>
</section>

<script>
// Scroll navigation — sends message to parent Streamlit window
function scrollToSection(section) {
  // Map section names to tab indices
  var tabMap = {
    'overview':   0,
    'models':     1,
    'evaluation': 5,
    'dataset':    4
  };

  // Try to communicate with parent frame to click the right tab
  try {
    window.parent.postMessage({ type: 'cineai_nav', section: section, tabIndex: tabMap[section] || 0 }, '*');
  } catch(e) {}

  // Also scroll the parent window down past the hero
  try {
    window.parent.scrollTo({ top: window.innerHeight, behavior: 'smooth' });
  } catch(e) {}
}

// Video fade logic
(function() {
  var v = document.getElementById('hero-video');
  if (!v) return;
  var FADE = 500, raf = null;

  function fadeIn() {
    var t0 = null;
    function step(ts) {
      if (!t0) t0 = ts;
      var p = Math.min((ts - t0) / FADE, 1);
      v.style.opacity = p;
      if (p < 1) raf = requestAnimationFrame(step);
    }
    raf = requestAnimationFrame(step);
  }

  function fadeOut(cb) {
    var op0 = parseFloat(v.style.opacity) || 1, t0 = null;
    function step(ts) {
      if (!t0) t0 = ts;
      var p = Math.min((ts - t0) / FADE, 1);
      v.style.opacity = op0 * (1 - p);
      if (p < 1) { raf = requestAnimationFrame(step); }
      else if (cb) cb();
    }
    raf = requestAnimationFrame(step);
  }

  v.addEventListener('canplay', function() {
    v.play().then(fadeIn).catch(function(){});
  }, { once: true });

  v.addEventListener('timeupdate', function() {
    if (v.duration && v.currentTime >= v.duration - 0.6 && !v._fo) {
      v._fo = true;
      fadeOut(function() {
        v.pause(); v.style.opacity = 0; v._fo = false;
        setTimeout(function() {
          v.currentTime = 0;
          v.play().then(fadeIn).catch(function(){});
        }, 120);
      });
    }
  });

  v.load();
})();
</script>
</body>
</html>"""

    components.html(hero_html, height=820, scrolling=False)


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
DATA_PATH = "ml-latest-small/"

@st.cache_data(show_spinner=False)
def load_data():
    movies  = pd.read_csv(DATA_PATH + "movies.csv")
    ratings = pd.read_csv(DATA_PATH + "ratings.csv")
    tags    = pd.read_csv(DATA_PATH + "tags.csv")
    links   = pd.read_csv(DATA_PATH + "links.csv")
    return movies, ratings, tags, links

@st.cache_data(show_spinner=False)
def build_matrix(ratings):
    return ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)


# ══════════════════════════════════════════════════════════════
# ML MODELS
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def compute_user_similarity(_matrix):
    R = _matrix.values.astype(np.float32)
    means = np.true_divide(R.sum(1), (R != 0).sum(1).clip(min=1))
    C = R.copy()
    for i in range(len(means)):
        mask = C[i] != 0
        C[i, mask] -= means[i]
    sim = cosine_similarity(C)
    np.fill_diagonal(sim, 0)
    return sim, means

@st.cache_data(show_spinner=False)
def compute_item_similarity(_matrix):
    sim = cosine_similarity(_matrix.T.values.astype(np.float32))
    np.fill_diagonal(sim, 0)
    return sim

@st.cache_data(show_spinner=False)
def compute_svd(_matrix, k=50):
    R = _matrix.values.astype(np.float32)
    means = np.true_divide(R.sum(1), (R != 0).sum(1).clip(min=1))
    C = R.copy()
    for i in range(len(means)):
        mask = C[i] != 0
        C[i, mask] -= means[i]
    k_act = min(k, min(C.shape) - 1)
    U, s, Vt = svds(csr_matrix(C), k=k_act)
    order = np.argsort(s)[::-1]
    return U[:, order], s[order], Vt[order, :], means


def ucf_recommend(user_id, matrix, user_sim, user_ids, movie_ids, movies_df,
                  n_neighbors=20, n_recs=10):
    u_idx = user_ids.index(user_id)
    sims  = user_sim[u_idx]
    top_k = np.argsort(sims)[::-1][:n_neighbors]
    actual = matrix.iloc[u_idx].values
    seen   = actual != 0
    pred   = np.zeros(len(movie_ids))
    ssum   = np.zeros(len(movie_ids))
    for nb in top_k:
        s = sims[nb]
        if s <= 0: continue
        nb_r = matrix.iloc[nb].values
        m = (~seen) & (nb_r != 0)
        pred[m]  += s * nb_r[m]
        ssum[m]  += abs(s)
    with np.errstate(divide="ignore", invalid="ignore"):
        pred = np.where(ssum > 0, pred / ssum, 0)
    pred[seen] = -1
    top = np.argsort(pred)[::-1][:n_recs]
    rows = []
    mi = movies_df.set_index("movieId")
    for idx in top:
        if pred[idx] <= 0: break
        mid = movie_ids[idx]
        rows.append({
            "Title": mi.loc[mid, "title"] if mid in mi.index else f"Movie {mid}",
            "Genres": mi.loc[mid, "genres"] if mid in mi.index else "",
            "Predicted Rating": round(pred[idx], 3),
        })
    return pd.DataFrame(rows)


def icf_recommend(user_id, matrix, item_sim, movie_ids, movies_df, n_recs=10):
    user_r   = matrix.loc[user_id].values
    seen     = user_r != 0
    unseen   = np.where(~seen)[0]
    scores   = np.zeros(len(movie_ids))
    rated    = np.where(seen)[0]
    for m in unseen:
        sv  = item_sim[m, rated]
        den = sv.sum()
        if den > 0:
            scores[m] = np.dot(sv, user_r[rated]) / den
    top = np.argsort(scores)[::-1][:n_recs]
    mi  = movies_df.set_index("movieId")
    rows = []
    for idx in top:
        if scores[idx] <= 0: break
        mid = movie_ids[idx]
        rows.append({
            "Title": mi.loc[mid, "title"] if mid in mi.index else f"Movie {mid}",
            "Genres": mi.loc[mid, "genres"] if mid in mi.index else "",
            "Predicted Rating": round(scores[idx], 3),
        })
    return pd.DataFrame(rows)


def icf_similar_movies(movie_id, item_sim, movie_ids, movies_df, n=10):
    if movie_id not in movie_ids: return pd.DataFrame()
    m_idx = movie_ids.index(movie_id)
    top   = np.argsort(item_sim[m_idx])[::-1][:n]
    mi    = movies_df.set_index("movieId")
    rows  = []
    for idx in top:
        mid = movie_ids[idx]
        rows.append({
            "Title": mi.loc[mid, "title"] if mid in mi.index else f"Movie {mid}",
            "Genres": mi.loc[mid, "genres"] if mid in mi.index else "",
            "Similarity": round(item_sim[m_idx][idx], 4),
        })
    return pd.DataFrame(rows)


def svd_recommend(user_id, U, s, Vt, means, user_ids, movie_ids, matrix, movies_df,
                  n_recs=10):
    u_idx = user_ids.index(user_id)
    pred  = (U[u_idx] @ np.diag(s) @ Vt) + means[u_idx]
    seen  = matrix.iloc[u_idx].values != 0
    pred  = pred.copy()
    pred[seen] = -1
    top   = np.argsort(pred)[::-1][:n_recs]
    mi    = movies_df.set_index("movieId")
    rows  = []
    for idx in top:
        if pred[idx] <= 0: break
        mid = movie_ids[idx]
        rows.append({
            "Title": mi.loc[mid, "title"] if mid in mi.index else f"Movie {mid}",
            "Genres": mi.loc[mid, "genres"] if mid in mi.index else "",
            "Predicted Rating": round(float(pred[idx]), 3),
        })
    return pd.DataFrame(rows)


def precision_at_k(rec_ids, relevant, k):
    return len(set(rec_ids[:k]) & set(relevant)) / k if rec_ids else 0.0

def recall_at_k(rec_ids, relevant, k):
    return len(set(rec_ids[:k]) & set(relevant)) / len(relevant) if relevant else 0.0


# ══════════════════════════════════════════════════════════════
# CHART HELPERS — Purple/Teal palette
# ══════════════════════════════════════════════════════════════
BRAND_SEQ = ["#f0ebf8","#D8BFD8","#9b7fbd","#6a3d9a","#452c63","#33006F","#1d1160","#004c4c","#006666","#b2d8d8"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#452c63", size=12),
    margin=dict(l=0, r=0, t=30, b=0),
)

PURPLE_SCALE = [[0,"#f0ebf8"], [0.35,"#D8BFD8"], [0.65,"#6a3d9a"], [1,"#1d1160"]]
TEAL_ACCENT_SCALE = [[0,"#e0f5f5"], [0.4,"#b2d8d8"], [0.7,"#006666"], [1,"#004c4c"]]
DIVERGE_SCALE = [[0,"#1d1160"], [0.5,"#D8BFD8"], [1,"#004c4c"]]


def rating_bar_chart(df):
    colors = ["#D8BFD8" if v < 4 else "#33006F" for v in df["rating"]]
    fig = go.Figure(go.Bar(
        x=df["rating"].astype(str) + "★",
        y=df["count"],
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="Rating: %{x}<br>Count: %{y:,}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(69,44,99,0.07)", zeroline=False)
    return fig


def singular_value_chart(s):
    total = (s**2).sum()
    cum   = np.cumsum(s**2) / total * 100
    k     = np.arange(1, len(s)+1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=k, y=s, name="Singular Value",
                         marker_color="#33006F", marker_line_width=0, yaxis="y"))
    fig.add_trace(go.Scatter(x=k, y=cum, name="Cumulative Variance %",
                             line=dict(color="#006666", width=2.5), yaxis="y2"))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        yaxis=dict(title="Singular Value", gridcolor="rgba(69,44,99,0.07)"),
        yaxis2=dict(title="Cumul. Variance %", overlaying="y", side="right",
                    range=[0, 100], gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
        barmode="overlay",
    )
    return fig


def recommendation_chart(df, score_col="Predicted Rating"):
    fig = px.bar(df.head(10), x=score_col, y="Title", orientation="h",
                 color=score_col,
                 color_continuous_scale=PURPLE_SCALE,
                 text=score_col)
    fig.update_layout(**PLOTLY_LAYOUT,
                      yaxis=dict(autorange="reversed", showgrid=False),
                      xaxis=dict(gridcolor="rgba(69,44,99,0.07)", zeroline=False),
                      showlegend=False, coloraxis_showscale=False)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                      marker_line_width=0)
    return fig


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    # Hero — full width, no gap at top
    render_hero()

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # ── Load data ──────────────────────────────────────────────
    with st.spinner("Loading MovieLens dataset..."):
        try:
            movies, ratings, tags, links = load_data()
        except FileNotFoundError as e:
            st.error(f"Dataset files not found at `{DATA_PATH}`\n\n{e}")
            st.stop()

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding: 24px 16px 0;">
          <div style="font-family:'Syne',sans-serif;font-weight:700;
                      font-size:20px;color:#fff;letter-spacing:-0.4px;margin-bottom:4px;">
            Configuration
          </div>
          <div style="font-size:12px;color:rgba(216,191,216,0.50);margin-bottom:24px;letter-spacing:0.3px;">
            Tune the ML models
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin: 0 16px 20px;
                    background:rgba(178,216,216,0.07);
                    border:1px solid rgba(178,216,216,0.14);
                    border-left:3px solid #006666;
                    border-radius:12px;padding:16px;
                    font-family:'DM Sans',sans-serif;font-size:13px;line-height:2.2;color:rgba(216,191,216,0.80);">
            <b style="color:white;">{len(movies):,}</b> movies &nbsp;&nbsp;
            <b style="color:white;">{len(ratings):,}</b> ratings<br>
            <b style="color:white;">{ratings['userId'].nunique():,}</b> users &nbsp;&nbsp;
            <b style="color:white;">{len(tags):,}</b> tags<br>
            Avg: <b style="color:#b2d8d8;">{ratings['rating'].mean():.2f} stars</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<div style="height:1px;background:rgba(216,191,216,0.12);margin:0 16px 16px;"></div>""", unsafe_allow_html=True)
        st.markdown("""<div style="padding:0 16px;font-family:'DM Sans',sans-serif;font-size:10px;
                       letter-spacing:1px;text-transform:uppercase;color:rgba(216,191,216,0.45);
                       margin-bottom:12px;">Model Parameters</div>""", unsafe_allow_html=True)

        demo_user   = st.selectbox("Demo User ID", options=sorted(ratings["userId"].unique())[:100], index=0)
        n_neighbors = st.slider("User-CF Neighbors (K)", 5, 50, 20)
        n_recs      = st.slider("Recommendations (N)", 5, 20, 10)
        svd_factors = st.slider("SVD Latent Factors", 10, 100, 50)

        st.markdown("""<div style="height:1px;background:rgba(216,191,216,0.12);margin:8px 16px 16px;"></div>""", unsafe_allow_html=True)
        st.markdown("""<div style="padding:0 16px;font-family:'DM Sans',sans-serif;font-size:10px;
                       letter-spacing:1px;text-transform:uppercase;color:rgba(216,191,216,0.45);
                       margin-bottom:12px;">Evaluation Settings</div>""", unsafe_allow_html=True)

        eval_k      = st.select_slider("Precision @ K", [5, 10, 15, 20], value=10)
        eval_thresh = st.select_slider("Relevance Threshold", [3.0, 3.5, 4.0, 4.5], value=4.0)
        eval_users  = st.slider("Test Users", 10, 60, 20)

    # ── Build matrix ───────────────────────────────────────────
    with st.spinner("Building user-item matrix..."):
        matrix    = build_matrix(ratings)
        user_ids  = list(matrix.index)
        movie_ids = list(matrix.columns)

    # ── Navigation anchor ──────────────────────────────────────
    st.markdown('<div id="overview"></div>', unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────
    tabs = st.tabs([
        "  Overview  ",
        "  User-Based CF  ",
        "  Item-Based CF  ",
        "  SVD Factorization  ",
        "  Explore Dataset  ",
        "  Evaluation  ",
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 0 — OVERVIEW
    # ══════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom:24px;">
          <div class="section-heading">Dataset at a Glance</div>
          <div class="section-subheading">MovieLens small dataset — 100K ratings across 9,742 movies</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Movies", f"{len(movies):,}")
        c2.metric("Ratings", f"{len(ratings):,}")
        c3.metric("Users", f"{ratings['userId'].nunique():,}")
        c4.metric("Tags", f"{len(tags):,}")
        c5.metric("Avg Rating", f"{ratings['rating'].mean():.2f} stars")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.divider()

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Rating Distribution")
            rdist = (ratings["rating"].value_counts().sort_index().reset_index())
            rdist.columns = ["rating", "count"]
            st.plotly_chart(rating_bar_chart(rdist), use_container_width=True)

        with col_right:
            st.markdown("#### Top 15 Genres")
            gc = defaultdict(int)
            for gs in movies["genres"].dropna():
                for g in gs.split("|"):
                    if g and g != "(no genres listed)": gc[g] += 1
            gdf = pd.DataFrame(list(gc.items()), columns=["Genre","Count"])\
                    .sort_values("Count", ascending=False).head(15)
            fig = px.bar(gdf, x="Count", y="Genre", orientation="h",
                         color="Count",
                         color_continuous_scale=PURPLE_SCALE)
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed", showgrid=False))
            fig.update_xaxes(gridcolor="rgba(69,44,99,0.07)", zeroline=False)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Top Rated Movies (min 50 ratings)")
            top_rated = (
                ratings.groupby("movieId")["rating"]
                .agg(avg_rating="mean", n_ratings="count")
                .query("n_ratings >= 50")
                .sort_values("avg_rating", ascending=False)
                .head(10).reset_index()
                .merge(movies[["movieId","title"]], on="movieId")
            )
            top_rated["avg_rating"] = top_rated["avg_rating"].round(2)
            top_rated["Rank"] = range(1, len(top_rated)+1)
            st.dataframe(
                top_rated[["Rank","title","avg_rating","n_ratings"]]
                .rename(columns={"title":"Title","avg_rating":"Avg Rating","n_ratings":"# Ratings"}),
                use_container_width=True, hide_index=True,
            )
        with col_b:
            st.markdown("#### Most Popular Movies")
            popular = (
                ratings.groupby("movieId")["rating"]
                .agg(n_ratings="count", avg_rating="mean")
                .sort_values("n_ratings", ascending=False)
                .head(10).reset_index()
                .merge(movies[["movieId","title"]], on="movieId")
            )
            popular["avg_rating"] = popular["avg_rating"].round(2)
            popular["Rank"] = range(1, len(popular)+1)
            st.dataframe(
                popular[["Rank","title","n_ratings","avg_rating"]]
                .rename(columns={"title":"Title","n_ratings":"# Ratings","avg_rating":"Avg Rating"}),
                use_container_width=True, hide_index=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 1 — USER-BASED CF
    # ══════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown('<div id="models"></div><div class="content-section">', unsafe_allow_html=True)
        st.markdown("""
        <div class="tab-description-card">
          <b>User-Based Collaborative Filtering</b><br>
          <span>Identifies users with similar rating patterns using Pearson similarity on
          mean-centered vectors, then predicts ratings for unseen movies via a weighted
          average of neighbour ratings.</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**Selected User:** `{demo_user}` &nbsp;&middot;&nbsp; {int((matrix.loc[demo_user] != 0).sum())} movies rated")
            run_ucf = st.button("Run User-CF Model", key="ucf_btn")

        if run_ucf:
            with st.spinner("Computing user similarity & generating recommendations..."):
                user_sim, umeans = compute_user_similarity(matrix)
                recs = ucf_recommend(demo_user, matrix, user_sim, user_ids,
                                     movie_ids, movies, n_neighbors, n_recs)
                u_idx  = user_ids.index(demo_user)
                sims   = user_sim[u_idx]
                top_nb = np.argsort(sims)[::-1][:8]
                sim_df = pd.DataFrame([{
                    "User ID": user_ids[i],
                    "Similarity": round(sims[i], 4),
                    "# Ratings": int((matrix.iloc[i] != 0).sum()),
                } for i in top_nb])

            st.divider()
            left, right = st.columns(2)
            with left:
                st.markdown("#### Most Similar Users")
                fig = px.bar(sim_df, x="User ID", y="Similarity",
                             color="Similarity",
                             color_continuous_scale=PURPLE_SCALE,
                             text="Similarity")
                fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
                fig.update_xaxes(type="category", showgrid=False)
                fig.update_yaxes(gridcolor="rgba(69,44,99,0.07)")
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Similar Users Table"):
                    st.dataframe(sim_df, use_container_width=True, hide_index=True)
            with right:
                st.markdown(f"#### Top {n_recs} Recommendations")
                if recs.empty:
                    st.info("No recommendations found for this user.")
                else:
                    st.plotly_chart(recommendation_chart(recs), use_container_width=True)
                    st.dataframe(recs, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 2 — ITEM-BASED CF
    # ══════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.markdown("""
        <div class="tab-description-card">
          <b>Item-Based Collaborative Filtering</b><br>
          <span>Computes cosine similarity between movie rating vectors. Scores unseen movies
          by their similarity to titles the user has already rated highly.</span>
        </div>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns([1, 1])
        with col_l:
            search_query = st.text_input("Search a movie to find similar titles", placeholder="e.g. Toy Story")
            run_icf_user = st.button("Get Recommendations for User", key="icf_user_btn")

        with st.spinner("Computing item-item similarity matrix..."):
            item_sim = compute_item_similarity(matrix)

        if run_icf_user:
            with st.spinner("Generating item-based recommendations..."):
                icf_recs = icf_recommend(demo_user, matrix, item_sim, movie_ids, movies, n_recs)
            st.divider()
            st.markdown(f"#### Item-CF Recommendations — User {demo_user}")
            if icf_recs.empty:
                st.info("No recommendations found.")
            else:
                st.plotly_chart(recommendation_chart(icf_recs), use_container_width=True)
                st.dataframe(icf_recs, use_container_width=True, hide_index=True)

        if search_query:
            matches = movies[movies["title"].str.contains(search_query, case=False, na=False)]
            if matches.empty:
                st.warning("No movies found matching that title.")
            else:
                chosen_title = st.selectbox("Select movie:", matches["title"].tolist())
                chosen_id    = matches[matches["title"] == chosen_title]["movieId"].values[0]
                if st.button("Find Similar Movies", key="icf_sim_btn"):
                    sim_movies = icf_similar_movies(chosen_id, item_sim, movie_ids, movies, n=12)
                    st.divider()
                    st.markdown(f"#### Movies Similar to *{chosen_title}*")
                    if sim_movies.empty:
                        st.info("Not enough rating data for this movie.")
                    else:
                        left, right = st.columns(2)
                        with left:
                            fig = px.bar(sim_movies, x="Similarity", y="Title", orientation="h",
                                         color="Similarity",
                                         color_continuous_scale=PURPLE_SCALE,
                                         text="Similarity")
                            fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                                              yaxis=dict(autorange="reversed", showgrid=False))
                            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
                            st.plotly_chart(fig, use_container_width=True)
                        with right:
                            st.dataframe(sim_movies, use_container_width=True, hide_index=True)
                        st.markdown("#### Genre Distribution of Similar Movies")
                        gcount = defaultdict(int)
                        for gs in sim_movies["Genres"].dropna():
                            for g in gs.split("|"):
                                if g: gcount[g] += 1
                        gdf2 = pd.DataFrame(list(gcount.items()), columns=["Genre","Count"])\
                                 .sort_values("Count", ascending=False)
                        fig2 = px.pie(gdf2, names="Genre", values="Count",
                                      color_discrete_sequence=BRAND_SEQ)
                        fig2.update_layout(**PLOTLY_LAYOUT)
                        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3 — SVD
    # ══════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.markdown("""
        <div class="tab-description-card">
          <b>SVD Matrix Factorization</b><br>
          <span>Decomposes the mean-centered user-item matrix R approximately equal to U·Sigma·Vt into k latent
          factors capturing hidden taste patterns. Predicted ratings are reconstructed
          as U·Sigma·Vt + user_mean.</span>
        </div>
        """, unsafe_allow_html=True)

        run_svd = st.button("Compute SVD & Recommend", key="svd_btn")

        if run_svd:
            with st.spinner(f"Computing SVD with k={svd_factors} latent factors..."):
                U, s, Vt, means = compute_svd(matrix, svd_factors)
                svd_recs = svd_recommend(demo_user, U, s, Vt, means,
                                         user_ids, movie_ids, matrix, movies, n_recs)
            st.divider()
            left, right = st.columns(2)
            with left:
                st.markdown("#### Singular Value Spectrum")
                st.plotly_chart(singular_value_chart(s[:30]), use_container_width=True)
                total = (s**2).sum()
                cum10 = (s[:10]**2).sum() / total * 100
                c1, c2, c3 = st.columns(3)
                c1.metric("Latent Factors", svd_factors)
                c2.metric("Var. Exp. (top 10)", f"{cum10:.1f}%")
                c3.metric("Top Sigma Value", f"{s[0]:.1f}")
                st.markdown("#### Top Singular Values")
                sdf = pd.DataFrame({
                    "Factor": range(1, min(15, len(s))+1),
                    "Singular Value": s[:15].round(3),
                    "Cumul. Variance %": (np.cumsum(s[:15]**2) / total * 100).round(2),
                })
                st.dataframe(sdf, use_container_width=True, hide_index=True)
            with right:
                st.markdown(f"#### SVD Recommendations — User {demo_user}")
                if svd_recs.empty:
                    st.info("No recommendations found.")
                else:
                    st.plotly_chart(recommendation_chart(svd_recs), use_container_width=True)
                    st.dataframe(svd_recs, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 4 — EXPLORE
    # ══════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown('<div id="dataset"></div><div class="content-section">', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom:20px;">
          <div class="section-heading">Explore the Dataset</div>
          <div class="section-subheading">Dive into individual movie profiles and user histories</div>
        </div>
        """, unsafe_allow_html=True)

        search_movie = st.text_input("Search for a movie profile", placeholder="e.g. Matrix, Inception...")

        if search_movie:
            matches = movies[movies["title"].str.contains(search_movie, case=False, na=False)]
            if not matches.empty:
                sel = st.selectbox("Pick a movie:", matches["title"].tolist(), key="explore_sel")
                sel_id    = matches[matches["title"] == sel]["movieId"].values[0]
                m_ratings = ratings[ratings["movieId"] == sel_id]["rating"]
                m_tags    = tags[tags["movieId"] == sel_id]["tag"].str.lower().value_counts().head(10)
                sel_genres = matches[matches["title"] == sel]["genres"].values[0]

                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Avg Rating", f"{m_ratings.mean():.2f}" if len(m_ratings) else "N/A")
                c2.metric("# Ratings", f"{len(m_ratings):,}")
                c3.metric("Std Dev", f"{m_ratings.std():.2f}" if len(m_ratings) > 1 else "N/A")
                c4.metric("Genres", str(sel_genres).count("|")+1 if sel_genres else 0)

                left, right = st.columns(2)
                with left:
                    st.markdown("**Rating Distribution**")
                    if len(m_ratings):
                        rdist2 = m_ratings.value_counts().sort_index().reset_index()
                        rdist2.columns = ["rating","count"]
                        st.plotly_chart(rating_bar_chart(rdist2), use_container_width=True)
                with right:
                    st.markdown("**Top Tags**")
                    if len(m_tags):
                        tdf = m_tags.reset_index(); tdf.columns = ["Tag","Uses"]
                        fig = px.bar(tdf, x="Uses", y="Tag", orientation="h", color="Uses",
                                     color_continuous_scale=TEAL_ACCENT_SCALE)
                        fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                                          yaxis=dict(autorange="reversed", showgrid=False))
                        fig.update_traces(marker_line_width=0)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No tags for this movie.")

        st.divider()
        st.markdown("### Top 40 User Tags")
        all_tags = tags["tag"].str.lower().value_counts().head(40).reset_index()
        all_tags.columns = ["Tag","Uses"]
        fig_tags = px.treemap(all_tags, path=["Tag"], values="Uses",
                              color="Uses",
                              color_continuous_scale=[[0,"#f0ebf8"],[0.3,"#D8BFD8"],[0.6,"#452c63"],[0.8,"#1d1160"],[1,"#004c4c"]])
        fig_tags.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_tags, use_container_width=True)

        st.divider()
        st.markdown("### User Rating Profile")
        explore_user = st.selectbox("Pick a user:", sorted(ratings["userId"].unique())[:100], key="explore_user")
        u_ratings = ratings[ratings["userId"] == explore_user].merge(
            movies[["movieId","title","genres"]], on="movieId"
        ).sort_values("rating", ascending=False)
        st.markdown(f"**User {explore_user}** has rated **{len(u_ratings)}** movies")
        st.dataframe(
            u_ratings[["title","genres","rating"]].rename(
                columns={"title":"Title","genres":"Genres","rating":"Rating"}
            ).head(20),
            use_container_width=True, hide_index=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 5 — EVALUATION
    # ══════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown('<div id="evaluation"></div><div class="content-section">', unsafe_allow_html=True)
        st.markdown("""
        <div class="tab-description-card">
          <b>Precision @ K Evaluation</b><br>
          <span>80/20 hold-out split per user. A recommendation is <i>relevant</i> if the user
          rated it at or above the threshold in the test set. Precision@K = hits in top-K / K.</span>
        </div>
        """, unsafe_allow_html=True)

        run_eval = st.button("Run Full Evaluation", key="eval_btn")

        if run_eval:
            progress = st.progress(0, text="Preparing evaluation...")
            eligible   = ratings.groupby("userId").filter(lambda x: len(x) >= 30)["userId"].unique()
            test_users = eligible[:eval_users]

            with st.spinner("Computing similarities (cached after first run)..."):
                user_sim, umeans = compute_user_similarity(matrix)
                item_sim2        = compute_item_similarity(matrix)
                U2, s2, Vt2, m2 = compute_svd(matrix, 30)

            ucf_p2, icf_p2, svd_p2 = [], [], []

            for i, uid in enumerate(test_users):
                progress.progress((i+1)/len(test_users),
                                   text=f"Evaluating user {uid} ({i+1}/{len(test_users)})...")
                ur  = ratings[ratings["userId"] == uid].sort_values("timestamp")
                sp  = int(len(ur) * 0.8)
                test_r = ur.iloc[sp:]
                rel    = test_r[test_r["rating"] >= eval_thresh]["movieId"].tolist()
                if not rel: continue
                try:
                    recs1 = ucf_recommend(uid, matrix, user_sim, user_ids, movie_ids, movies, n_neighbors, eval_k)
                    ids1  = [movies[movies["title"]==t]["movieId"].values[0]
                             for t in recs1["Title"].tolist() if len(movies[movies["title"]==t])>0]
                    ucf_p2.append(precision_at_k(ids1, rel, eval_k))
                except: pass
                try:
                    recs2 = icf_recommend(uid, matrix, item_sim2, movie_ids, movies, eval_k)
                    ids2  = [movies[movies["title"]==t]["movieId"].values[0]
                             for t in recs2["Title"].tolist() if len(movies[movies["title"]==t])>0]
                    icf_p2.append(precision_at_k(ids2, rel, eval_k))
                except: pass
                try:
                    recs3 = svd_recommend(uid, U2, s2, Vt2, m2, user_ids, movie_ids, matrix, movies, eval_k)
                    ids3  = [movies[movies["title"]==t]["movieId"].values[0]
                             for t in recs3["Title"].tolist() if len(movies[movies["title"]==t])>0]
                    svd_p2.append(precision_at_k(ids3, rel, eval_k))
                except: pass

            progress.empty()

            avg_ucf = np.mean(ucf_p2) if ucf_p2 else 0.0
            avg_icf = np.mean(icf_p2) if icf_p2 else 0.0
            avg_svd = np.mean(svd_p2) if svd_p2 else 0.0
            best    = max(["User-CF","Item-CF","SVD"],
                          key=lambda m: {"User-CF":avg_ucf,"Item-CF":avg_icf,"SVD":avg_svd}[m])

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"User-CF P@{eval_k}", f"{avg_ucf*100:.2f}%")
            c2.metric(f"Item-CF P@{eval_k}", f"{avg_icf*100:.2f}%")
            c3.metric(f"SVD P@{eval_k}",     f"{avg_svd*100:.2f}%")
            c4.metric("Best Model", best)

            st.divider()
            st.markdown("#### Algorithm Comparison")
            comp_df = pd.DataFrame({
                "Model": ["User-CF","Item-CF","SVD"],
                f"Precision@{eval_k}": [avg_ucf, avg_icf, avg_svd],
            })
            fig_comp = px.bar(comp_df, x="Model", y=f"Precision@{eval_k}",
                              color=f"Precision@{eval_k}",
                              color_continuous_scale=PURPLE_SCALE,
                              text=f"Precision@{eval_k}")
            fig_comp.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
            fig_comp.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                                   yaxis=dict(gridcolor="rgba(69,44,99,0.07)"),
                                   xaxis=dict(showgrid=False))
            st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown("#### Evaluation Parameters Used")
            st.dataframe(pd.DataFrame([{
                "K (cutoff)": eval_k,
                "Relevance Threshold": f">= {eval_thresh} stars",
                "Test Users": len(test_users),
                "UCF Neighbors": n_neighbors,
                "SVD Factors": 30,
            }]), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────
    st.markdown("""
    <div class="cineai-footer">
      CineAI &nbsp;&middot;&nbsp; MovieLens ML Engine &nbsp;&middot;&nbsp;
      <span>User-CF</span> &nbsp;&middot;&nbsp;
      <span>Item-CF</span> &nbsp;&middot;&nbsp;
      <span>SVD Matrix Factorization</span> &nbsp;&middot;&nbsp;
      Precision@K &nbsp;&middot;&nbsp; Built with Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
