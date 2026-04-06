"""
╔══════════════════════════════════════════════════════════════╗
║         CineAI — Movie Recommendation System                 ║
║         Streamlit Web Application                            ║
║         Run: streamlit run movie_recommender_app.py          ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
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
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# MASTER CSS — CineAI Brand System
# Palette: #191970 (midnight), #A7C7E7 (powder), #008080 (teal), #FFFFFF
# Fonts: Geist Sans (body), General Sans (display)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://api.fontshare.com/v2/css?f[]=general-sans@400,500,600,700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@300;400;500&display=swap');

/* Geist Sans via CDN fallback */
@font-face {
  font-family: 'Geist Sans';
  src: url('https://cdn.jsdelivr.net/npm/geist@1.3.0/dist/fonts/geist-sans/Geist-Regular.woff2') format('woff2');
  font-weight: 400; font-display: swap;
}
@font-face {
  font-family: 'Geist Sans';
  src: url('https://cdn.jsdelivr.net/npm/geist@1.3.0/dist/fonts/geist-sans/Geist-Medium.woff2') format('woff2');
  font-weight: 500; font-display: swap;
}
@font-face {
  font-family: 'Geist Sans';
  src: url('https://cdn.jsdelivr.net/npm/geist@1.3.0/dist/fonts/geist-sans/Geist-SemiBold.woff2') format('woff2');
  font-weight: 600; font-display: swap;
}
@font-face {
  font-family: 'Geist Sans';
  src: url('https://cdn.jsdelivr.net/npm/geist@1.3.0/dist/fonts/geist-sans/Geist-Bold.woff2') format('woff2');
  font-weight: 700; font-display: swap;
}

/* ── Root palette ── */
:root {
  --midnight:     #191970;
  --midnight-deep:#0d0d3d;
  --midnight-mid: #1e2485;
  --powder:       #A7C7E7;
  --powder-light: #c8dcf0;
  --powder-faint: #e8f2fa;
  --teal:         #008080;
  --teal-light:   #00a3a3;
  --teal-faint:   #e0f5f5;
  --white:        #FFFFFF;
  --off-white:    #f4f6fb;
  --text-1:       #0d1240;
  --text-2:       #3a4580;
  --text-3:       #7a88c0;
  --glass-light:  rgba(255,255,255,0.62);
  --glass-dark:   rgba(25,25,112,0.08);
  --border-light: rgba(167,199,231,0.35);
  --border-mid:   rgba(25,25,112,0.14);
  --shadow-blue:  rgba(25,25,112,0.12);
  --shadow-teal:  rgba(0,128,128,0.15);
}

/* ── Liquid glass utility ── */
.liquid-glass {
  background: rgba(255,255,255,0.08);
  background-blend-mode: luminosity;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: none;
  box-shadow: inset 0 1px 1px rgba(255,255,255,0.18), 0 8px 32px rgba(25,25,112,0.10);
  position: relative;
  overflow: hidden;
}
.liquid-glass::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  padding: 1.4px;
  background: linear-gradient(180deg,
    rgba(255,255,255,0.45) 0%,
    rgba(255,255,255,0.15) 20%,
    rgba(255,255,255,0) 40%,
    rgba(255,255,255,0) 60%,
    rgba(255,255,255,0.15) 80%,
    rgba(255,255,255,0.45) 100%);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  pointer-events: none;
}

/* ── Global resets ── */
html, body, [class*="css"] {
  font-family: 'Geist Sans', 'DM Sans', system-ui, sans-serif !important;
  color: var(--text-1) !important;
}

/* ── Main background ── */
.stApp {
  background: linear-gradient(160deg, #f0f4ff 0%, #f8faff 40%, #e8f5f5 80%, #f0f4ff 100%) !important;
  min-height: 100vh;
}

/* Noise texture overlay */
.stApp::after {
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.025'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 0;
  opacity: 0.4;
}

/* Subtle grid overlay */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(25,25,112,.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(25,25,112,.025) 1px, transparent 1px);
  background-size: 56px 56px;
  pointer-events: none;
  z-index: 0;
}

/* Main content above overlays */
.main .block-container {
  position: relative;
  z-index: 1;
  padding-top: 0 !important;
  max-width: 1280px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(25,25,112,0.96) 0%, rgba(13,13,61,0.98) 100%) !important;
  backdrop-filter: blur(20px);
  border-right: 1px solid rgba(167,199,231,0.15) !important;
  box-shadow: 4px 0 40px rgba(25,25,112,0.25) !important;
}
[data-testid="stSidebar"] * {
  color: var(--powder-light) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] b,
[data-testid="stSidebar"] strong {
  color: var(--white) !important;
}
[data-testid="stSidebar"] hr {
  border-color: rgba(167,199,231,0.15) !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div > div {
  background: var(--teal) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
  background: rgba(255,255,255,0.07) !important;
  border: 1px solid rgba(167,199,231,0.2) !important;
  border-radius: 10px !important;
  color: var(--powder-light) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label {
  color: rgba(167,199,231,0.7) !important;
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--midnight) 0%, var(--midnight-mid) 100%) !important;
  color: white !important;
  border: 1px solid rgba(167,199,231,0.25) !important;
  border-radius: 10px !important;
  font-family: 'Geist Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  letter-spacing: 0.3px;
  padding: 10px 24px !important;
  box-shadow: 0 4px 20px rgba(25,25,112,0.3), inset 0 1px 0 rgba(255,255,255,0.12) !important;
  transition: all 0.28s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
  width: 100%;
  position: relative;
  overflow: hidden;
}
.stButton > button::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(0,128,128,0.0) 0%, rgba(0,128,128,0.15) 100%);
  opacity: 0;
  transition: opacity 0.25s ease;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 32px rgba(25,25,112,0.4), 0 0 0 1px rgba(0,128,128,0.3) !important;
  border-color: rgba(0,128,128,0.5) !important;
}
.stButton > button:hover::after { opacity: 1; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Labels ── */
.stSelectbox label, .stSlider label, .stTextInput label,
.stNumberInput label, .stMultiSelect label {
  font-family: 'Geist Sans', sans-serif !important;
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
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px rgba(0,128,128,0.1) !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input {
  background: white !important;
  border: 1px solid var(--border-light) !important;
  border-radius: 10px !important;
  color: var(--text-1) !important;
  font-family: 'Geist Sans', sans-serif !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.stTextInput > div > div > input:focus {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px rgba(0,128,128,0.1) !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
  background: var(--midnight) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: rgba(255,255,255,0.72) !important;
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-light) !important;
  border-radius: 16px !important;
  padding: 18px 22px !important;
  box-shadow: 0 4px 24px var(--shadow-blue) !important;
  transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.25s ease !important;
}
[data-testid="metric-container"]:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 36px var(--shadow-blue) !important;
}
[data-testid="metric-container"] label {
  font-size: 10px !important;
  color: var(--text-3) !important;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  font-family: 'Geist Sans', sans-serif !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'General Sans', sans-serif !important;
  font-weight: 700 !important;
  font-size: 26px !important;
  color: var(--midnight) !important;
  background: linear-gradient(135deg, var(--midnight), var(--teal));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.65) !important;
  backdrop-filter: blur(14px);
  border: 1px solid var(--border-light) !important;
  border-radius: 14px !important;
  padding: 5px 6px !important;
  gap: 3px !important;
  box-shadow: 0 4px 20px var(--shadow-blue) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px !important;
  font-family: 'Geist Sans', sans-serif !important;
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
  background: rgba(25,25,112,0.05) !important;
  color: var(--midnight) !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--midnight) 0%, var(--midnight-mid) 100%) !important;
  color: white !important;
  box-shadow: 0 4px 14px rgba(25,25,112,0.3) !important;
}

/* ── Dataframe ── */
.stDataFrame {
  border: 1px solid var(--border-light) !important;
  border-radius: 14px !important;
  overflow: hidden;
  box-shadow: 0 4px 16px var(--shadow-blue) !important;
}

/* ── Divider ── */
hr { border-color: var(--border-light) !important; opacity: 0.5; }

/* ── Expander ── */
.streamlit-expanderHeader {
  background: rgba(255,255,255,0.65) !important;
  border: 1px solid var(--border-light) !important;
  border-radius: 10px !important;
  font-family: 'Geist Sans', sans-serif !important;
  font-weight: 500 !important;
  color: var(--text-1) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--teal) !important; }

/* ── Alerts ── */
.stAlert { border-radius: 12px !important; }

/* ── Progress bar ── */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--midnight), var(--teal)) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--powder-faint); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(var(--midnight), var(--teal));
  border-radius: 3px;
}

/* ══════════════════════════════════════════════════
   HERO SECTION
══════════════════════════════════════════════════ */
.cineai-hero-wrapper {
  position: relative;
  width: calc(100% + 8rem);
  margin-left: -4rem;
  margin-right: -4rem;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: #070720;
}
@media (max-width: 768px) {
  .cineai-hero-wrapper {
    width: 100%;
    margin-left: 0;
    margin-right: 0;
  }
}

/* Video background */
#hero-video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0;
  z-index: 0;
}

/* Dark overlay for readability */
.hero-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    to bottom,
    rgba(7,7,32,0.55) 0%,
    rgba(7,7,32,0.30) 40%,
    rgba(7,7,32,0.50) 75%,
    rgba(7,7,32,0.88) 100%
  );
  z-index: 1;
}

/* Blurred glow orb behind headline */
.hero-glow-orb {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -52%);
  width: 860px;
  height: 480px;
  background: rgba(7,7,32,0.72);
  filter: blur(80px);
  border-radius: 50%;
  z-index: 2;
  pointer-events: none;
}
@media (max-width: 900px) {
  .hero-glow-orb { width: 94vw; height: 320px; }
}

/* Content layer */
.hero-content {
  position: relative;
  z-index: 3;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

/* ── Navbar ── */
.hero-navbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 22px 48px;
  width: 100%;
  box-sizing: border-box;
}
@media (max-width: 768px) {
  .hero-navbar { padding: 18px 20px; }
}

.hero-nav-logo {
  font-family: 'General Sans', sans-serif;
  font-weight: 700;
  font-size: 22px;
  color: #ffffff;
  letter-spacing: -0.5px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.hero-nav-logo-icon {
  width: 36px;
  height: 36px;
  background: linear-gradient(135deg, #A7C7E7 0%, #008080 100%);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  box-shadow: 0 4px 16px rgba(0,128,128,0.4);
  flex-shrink: 0;
}

.hero-nav-links {
  display: flex;
  align-items: center;
  gap: 6px;
}
@media (max-width: 680px) {
  .hero-nav-links { display: none; }
}

.hero-nav-link {
  font-family: 'Geist Sans', sans-serif;
  font-weight: 400;
  font-size: 14px;
  color: rgba(255,255,255,0.82);
  padding: 7px 14px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  text-decoration: none;
  background: transparent;
  border: none;
  display: flex;
  align-items: center;
  gap: 4px;
}
.hero-nav-link:hover {
  background: rgba(255,255,255,0.08);
  color: white;
}

.hero-nav-cta {
  font-family: 'Geist Sans', sans-serif;
  font-weight: 500;
  font-size: 13px;
  color: #191970;
  background: white;
  padding: 8px 20px;
  border-radius: 50px;
  cursor: pointer;
  transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
  border: none;
  letter-spacing: 0.1px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}
.hero-nav-cta:hover {
  background: #A7C7E7;
  transform: translateY(-1px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}

/* Navbar divider */
.hero-navbar-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
  margin: 0 48px;
}
@media (max-width: 768px) {
  .hero-navbar-divider { margin: 0 20px; }
}

/* ── Hero body ── */
.hero-body {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px 48px;
  box-sizing: border-box;
}
@media (max-width: 768px) {
  .hero-body { padding: 30px 20px; }
}

.hero-body-inner {
  text-align: center;
  max-width: 860px;
}

.hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: 'Geist Sans', sans-serif;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 1.8px;
  text-transform: uppercase;
  color: rgba(167,199,231,0.85);
  background: rgba(167,199,231,0.08);
  border: 1px solid rgba(167,199,231,0.2);
  padding: 6px 16px;
  border-radius: 50px;
  margin-bottom: 28px;
  animation: fadeSlideDown 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.hero-eyebrow-dot {
  width: 6px; height: 6px;
  background: var(--teal);
  border-radius: 50%;
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.7); }
}

/* Headline */
.hero-headline {
  font-family: 'General Sans', sans-serif;
  font-weight: 700;
  font-size: clamp(52px, 8vw, 108px);
  line-height: 1.0;
  letter-spacing: -0.03em;
  color: #ffffff;
  margin: 0 0 4px 0;
  animation: fadeSlideUp 0.75s cubic-bezier(0.22, 1, 0.36, 1) 0.15s both;
}
.hero-headline-accent {
  background: linear-gradient(120deg, #A7C7E7 0%, #ffffff 40%, #008080 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  display: inline-block;
}

.hero-subtitle {
  font-family: 'Geist Sans', sans-serif;
  font-size: clamp(15px, 2vw, 18px);
  font-weight: 400;
  color: rgba(228,235,255,0.72);
  line-height: 1.7;
  max-width: 520px;
  margin: 18px auto 0;
  animation: fadeSlideUp 0.75s cubic-bezier(0.22, 1, 0.36, 1) 0.28s both;
}

/* CTA buttons row */
.hero-cta-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 14px;
  margin-top: 36px;
  flex-wrap: wrap;
  animation: fadeSlideUp 0.75s cubic-bezier(0.22, 1, 0.36, 1) 0.40s both;
}

.hero-btn-primary {
  font-family: 'Geist Sans', sans-serif;
  font-weight: 500;
  font-size: 14px;
  color: #191970;
  background: #ffffff;
  padding: 14px 32px;
  border-radius: 12px;
  border: none;
  cursor: pointer;
  letter-spacing: 0.1px;
  transition: all 0.28s cubic-bezier(0.34, 1.56, 0.64, 1);
  box-shadow: 0 4px 20px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.9);
  text-decoration: none;
  display: inline-block;
}
.hero-btn-primary:hover {
  background: #A7C7E7;
  transform: translateY(-2px);
  box-shadow: 0 10px 32px rgba(0,0,0,0.3);
}

.hero-btn-secondary {
  font-family: 'Geist Sans', sans-serif;
  font-weight: 400;
  font-size: 14px;
  color: rgba(255,255,255,0.85);
  background: rgba(255,255,255,0.06);
  padding: 14px 28px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.15);
  cursor: pointer;
  letter-spacing: 0.1px;
  transition: all 0.25s ease;
  display: flex;
  align-items: center;
  gap: 8px;
  text-decoration: none;
}
.hero-btn-secondary:hover {
  background: rgba(255,255,255,0.1);
  border-color: rgba(167,199,231,0.4);
  color: white;
}

/* Stats row */
.hero-stats {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 48px;
  margin-top: 52px;
  animation: fadeIn 1s cubic-bezier(0.22, 1, 0.36, 1) 0.6s both;
}
@media (max-width: 600px) {
  .hero-stats { gap: 24px; flex-wrap: wrap; }
}

.hero-stat {
  text-align: center;
}
.hero-stat-value {
  font-family: 'General Sans', sans-serif;
  font-weight: 700;
  font-size: 28px;
  color: white;
  letter-spacing: -0.5px;
  line-height: 1;
}
.hero-stat-label {
  font-family: 'Geist Sans', sans-serif;
  font-size: 11px;
  color: rgba(167,199,231,0.6);
  letter-spacing: 0.8px;
  text-transform: uppercase;
  margin-top: 4px;
}
.hero-stat-divider {
  width: 1px;
  height: 36px;
  background: rgba(167,199,231,0.2);
}

/* ── Marquee ── */
.hero-marquee-section {
  padding: 28px 48px 36px;
  animation: fadeIn 1s ease 0.8s both;
}
@media (max-width: 768px) {
  .hero-marquee-section { padding: 24px 20px 30px; }
}

.hero-marquee-label {
  font-family: 'Geist Sans', sans-serif;
  font-size: 11px;
  color: rgba(167,199,231,0.45);
  letter-spacing: 1px;
  text-transform: uppercase;
  text-align: center;
  margin-bottom: 16px;
}

.marquee-track-wrapper {
  overflow: hidden;
  mask-image: linear-gradient(90deg, transparent 0%, black 15%, black 85%, transparent 100%);
  -webkit-mask-image: linear-gradient(90deg, transparent 0%, black 15%, black 85%, transparent 100%);
}

.marquee-track {
  display: flex;
  gap: 40px;
  animation: marqueeScroll 28s linear infinite;
  width: max-content;
}
@media (max-width: 600px) {
  .marquee-track { animation-duration: 20s; }
}

@keyframes marqueeScroll {
  from { transform: translateX(0); }
  to   { transform: translateX(-50%); }
}

.marquee-item {
  display: flex;
  align-items: center;
  gap: 10px;
  color: rgba(255,255,255,0.55);
  font-family: 'Geist Sans', sans-serif;
  font-size: 14px;
  font-weight: 500;
  letter-spacing: 0.2px;
  white-space: nowrap;
  transition: color 0.2s ease;
}
.marquee-item:hover { color: rgba(255,255,255,0.85); }

.marquee-icon {
  width: 28px;
  height: 28px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(167,199,231,0.15);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  color: #A7C7E7;
  flex-shrink: 0;
  position: relative;
  overflow: hidden;
}
.marquee-icon::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  padding: 1px;
  background: linear-gradient(180deg, rgba(255,255,255,0.3) 0%, rgba(255,255,255,0) 50%);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
}

/* ══════════════════════════════════════════════════
   MAIN CONTENT (below hero)
══════════════════════════════════════════════════ */
.content-section {
  padding: 36px 0;
  animation: fadeIn 0.5s ease both;
}

/* Section heading style */
.section-heading {
  font-family: 'General Sans', sans-serif;
  font-weight: 600;
  font-size: 20px;
  color: var(--midnight);
  letter-spacing: -0.3px;
  margin: 0 0 6px 0;
}
.section-subheading {
  font-family: 'Geist Sans', sans-serif;
  font-size: 13px;
  color: var(--text-3);
  margin: 0 0 20px 0;
}

/* Info / description cards inside tabs */
.tab-description-card {
  background: linear-gradient(135deg, rgba(25,25,112,0.04) 0%, rgba(0,128,128,0.04) 100%);
  border: 1px solid rgba(25,25,112,0.10);
  border-left: 3px solid var(--teal);
  border-radius: 12px;
  padding: 18px 22px;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
}
.tab-description-card::before {
  content: '';
  position: absolute;
  top: 0; right: 0;
  width: 120px; height: 120px;
  background: radial-gradient(circle, rgba(0,128,128,0.06) 0%, transparent 70%);
  pointer-events: none;
}
.tab-description-card b {
  font-family: 'General Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 15px !important;
  color: var(--midnight) !important;
}
.tab-description-card span {
  font-family: 'Geist Sans', sans-serif !important;
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
@keyframes zoomIn {
  from { opacity: 0; transform: scale(0.94); }
  to   { opacity: 1; transform: scale(1); }
}
@keyframes slideInLeft {
  from { opacity: 0; transform: translateX(-20px); }
  to   { opacity: 1; transform: translateX(0); }
}

/* Staggered content animation on tab load */
.stTabs [data-baseweb="tab-panel"] > div > div:nth-child(1) { animation: fadeSlideUp 0.4s ease both; }
.stTabs [data-baseweb="tab-panel"] > div > div:nth-child(2) { animation: fadeSlideUp 0.4s ease 0.08s both; }
.stTabs [data-baseweb="tab-panel"] > div > div:nth-child(3) { animation: fadeSlideUp 0.4s ease 0.15s both; }
.stTabs [data-baseweb="tab-panel"] > div > div:nth-child(4) { animation: fadeSlideUp 0.4s ease 0.22s both; }

/* ── Section labels ── */
.stMarkdown h3 {
  font-family: 'General Sans', sans-serif !important;
  font-weight: 600 !important;
  color: var(--midnight) !important;
  letter-spacing: -0.2px !important;
}
.stMarkdown h4 {
  font-family: 'General Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 16px !important;
  color: var(--text-1) !important;
}

/* ── Footer ── */
.cineai-footer {
  text-align: center;
  padding: 20px 0 32px;
  font-family: 'Geist Sans', sans-serif;
  font-size: 12px;
  color: var(--text-3);
  letter-spacing: 0.3px;
  border-top: 1px solid var(--border-light);
  margin-top: 16px;
}
.cineai-footer span {
  color: var(--teal);
  font-weight: 500;
}

/* ── Responsive tweaks ── */
@media (max-width: 900px) {
  .hero-headline { font-size: clamp(40px, 9vw, 72px); }
  .main .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
}
@media (max-width: 600px) {
  .hero-headline { font-size: clamp(36px, 11vw, 56px); }
  .hero-stats { gap: 20px; }
  .hero-stat-divider { display: none; }
  .stTabs [data-baseweb="tab"] { font-size: 11px !important; padding: 7px 10px !important; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════
def render_hero():
    st.markdown("""
    <div class="cineai-hero-wrapper">

      <!-- Background video with JS fade loop -->
      <video id="hero-video" src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260328_065045_c44942da-53c6-4804-b734-f9e07fc22e08.mp4"
             muted playsinline preload="auto" aria-hidden="true"></video>

      <!-- Dark overlay -->
      <div class="hero-overlay"></div>

      <!-- Glow orb -->
      <div class="hero-glow-orb"></div>

      <!-- All content -->
      <div class="hero-content">

        <!-- Navbar -->
        <nav class="hero-navbar">
          <div class="hero-nav-logo">
            <div class="hero-nav-logo-icon">🎬</div>
            CineAI
          </div>
          <div class="hero-nav-links">
            <button class="hero-nav-link">Overview</button>
            <button class="hero-nav-link">Models</button>
            <button class="hero-nav-link">Evaluation</button>
            <button class="hero-nav-link">Dataset</button>
          </div>
          <button class="hero-nav-cta">Get Started ↓</button>
        </nav>
        <div class="hero-navbar-divider"></div>

        <!-- Hero body -->
        <div class="hero-body">
          <div class="hero-body-inner">

            <div class="hero-eyebrow">
              <span class="hero-eyebrow-dot"></span>
              MovieLens 100K · ML Recommendation Engine
            </div>

            <h1 class="hero-headline">
              Discover<br>
              <span class="hero-headline-accent">Cinema</span>
            </h1>

            <p class="hero-subtitle">
              Powered by User-CF, Item-CF &amp; SVD Matrix Factorization —
              intelligent film recommendations built on real viewing patterns.
            </p>

            <div class="hero-cta-row">
              <a href="#" class="hero-btn-primary">Explore Recommendations</a>
              <a href="#" class="hero-btn-secondary">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8" fill="currentColor" stroke="none"/></svg>
                How It Works
              </a>
            </div>

            <!-- Stats -->
            <div class="hero-stats">
              <div class="hero-stat">
                <div class="hero-stat-value">9,742</div>
                <div class="hero-stat-label">Movies</div>
              </div>
              <div class="hero-stat-divider"></div>
              <div class="hero-stat">
                <div class="hero-stat-value">100K+</div>
                <div class="hero-stat-label">Ratings</div>
              </div>
              <div class="hero-stat-divider"></div>
              <div class="hero-stat">
                <div class="hero-stat-value">610</div>
                <div class="hero-stat-label">Users</div>
              </div>
              <div class="hero-stat-divider"></div>
              <div class="hero-stat">
                <div class="hero-stat-value">3</div>
                <div class="hero-stat-label">ML Models</div>
              </div>
            </div>

          </div>
        </div>

        <!-- Marquee -->
        <div class="hero-marquee-section">
          <div class="hero-marquee-label">Powered by industry-grade algorithms</div>
          <div class="marquee-track-wrapper">
            <div class="marquee-track">
              <!-- Set A -->
              <div class="marquee-item"><div class="marquee-icon">U</div>User-CF</div>
              <div class="marquee-item"><div class="marquee-icon">I</div>Item-CF</div>
              <div class="marquee-item"><div class="marquee-icon">S</div>SVD Factorization</div>
              <div class="marquee-item"><div class="marquee-icon">P</div>Precision@K</div>
              <div class="marquee-item"><div class="marquee-icon">C</div>Cosine Similarity</div>
              <div class="marquee-item"><div class="marquee-icon">M</div>MovieLens</div>
              <div class="marquee-item"><div class="marquee-icon">L</div>Latent Factors</div>
              <div class="marquee-item"><div class="marquee-icon">R</div>Recall@K</div>
              <!-- Set B (duplicate for seamless loop) -->
              <div class="marquee-item"><div class="marquee-icon">U</div>User-CF</div>
              <div class="marquee-item"><div class="marquee-icon">I</div>Item-CF</div>
              <div class="marquee-item"><div class="marquee-icon">S</div>SVD Factorization</div>
              <div class="marquee-item"><div class="marquee-icon">P</div>Precision@K</div>
              <div class="marquee-item"><div class="marquee-icon">C</div>Cosine Similarity</div>
              <div class="marquee-item"><div class="marquee-icon">M</div>MovieLens</div>
              <div class="marquee-item"><div class="marquee-icon">L</div>Latent Factors</div>
              <div class="marquee-item"><div class="marquee-icon">R</div>Recall@K</div>
            </div>
          </div>
        </div>

      </div><!-- /hero-content -->
    </div><!-- /hero-wrapper -->

    <!-- Video fade-loop script -->
    <script>
    (function() {
      function initVideo() {
        var v = document.getElementById('hero-video');
        if (!v) { setTimeout(initVideo, 200); return; }

        var FADE_DURATION = 500;
        var rafId = null;

        function fadeIn() {
          var start = null;
          function step(ts) {
            if (!start) start = ts;
            var p = Math.min((ts - start) / FADE_DURATION, 1);
            v.style.opacity = p;
            if (p < 1) rafId = requestAnimationFrame(step);
          }
          rafId = requestAnimationFrame(step);
        }

        function fadeOut(cb) {
          var startOpacity = parseFloat(v.style.opacity) || 1;
          var start = null;
          function step(ts) {
            if (!start) start = ts;
            var p = Math.min((ts - start) / FADE_DURATION, 1);
            v.style.opacity = startOpacity * (1 - p);
            if (p < 1) { rafId = requestAnimationFrame(step); }
            else if (cb) cb();
          }
          rafId = requestAnimationFrame(step);
        }

        v.addEventListener('canplay', function() {
          v.play().then(fadeIn).catch(function(){});
        }, { once: true });

        v.addEventListener('timeupdate', function() {
          if (v.duration && v.currentTime >= v.duration - 0.6 && !v._fadingOut) {
            v._fadingOut = true;
            fadeOut(function() {
              v.pause();
              v.style.opacity = 0;
              v._fadingOut = false;
              setTimeout(function() { v.currentTime = 0; v.play().then(fadeIn).catch(function(){}); }, 120);
            });
          }
        });

        v.load();
      }
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initVideo);
      } else {
        initVideo();
      }
    })();
    </script>
    """, unsafe_allow_html=True)


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
# ML MODELS  (unchanged)
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
# CHART HELPERS  (updated palette)
# ══════════════════════════════════════════════════════════════
BRAND_SEQ = ["#e8f2fa","#A7C7E7","#6fa8c8","#3a7ca8","#1a5c8a","#191970","#0d0d3d","#008080"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Geist Sans", color="#3a4580", size=12),
    margin=dict(l=0, r=0, t=30, b=0),
)

def rating_bar_chart(df):
    colors = ["#A7C7E7" if v < 4 else "#191970" for v in df["rating"]]
    fig = go.Figure(go.Bar(
        x=df["rating"].astype(str) + "★",
        y=df["count"],
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="Rating: %{x}<br>Count: %{y:,}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(25,25,112,0.06)", zeroline=False)
    return fig


def singular_value_chart(s):
    total = (s**2).sum()
    cum   = np.cumsum(s**2) / total * 100
    k     = np.arange(1, len(s)+1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=k, y=s, name="Singular Value",
                         marker_color="#191970", marker_line_width=0, yaxis="y"))
    fig.add_trace(go.Scatter(x=k, y=cum, name="Cumulative Variance %",
                             line=dict(color="#008080", width=2.5), yaxis="y2"))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        yaxis=dict(title="Singular Value", gridcolor="rgba(25,25,112,0.06)"),
        yaxis2=dict(title="Cumul. Variance %", overlaying="y", side="right",
                    range=[0, 100], gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
        barmode="overlay",
    )
    return fig


def recommendation_chart(df, score_col="Predicted Rating"):
    fig = px.bar(df.head(10), x=score_col, y="Title", orientation="h",
                 color=score_col,
                 color_continuous_scale=[[0,"#e8f2fa"],[0.5,"#A7C7E7"],[1,"#191970"]],
                 text=score_col)
    fig.update_layout(**PLOTLY_LAYOUT,
                      yaxis=dict(autorange="reversed", showgrid=False),
                      xaxis=dict(gridcolor="rgba(25,25,112,0.06)", zeroline=False),
                      showlegend=False, coloraxis_showscale=False)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                      marker_line_width=0)
    return fig


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    # Hero
    render_hero()

    # Spacer
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # ── Load data ──────────────────────────────────────────────
    with st.spinner("Loading MovieLens dataset…"):
        try:
            movies, ratings, tags, links = load_data()
        except FileNotFoundError as e:
            st.error(f"Dataset files not found at `{DATA_PATH}`\n\n{e}")
            st.stop()

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding: 24px 16px 0;">
          <div style="font-family:'General Sans',sans-serif;font-weight:700;
                      font-size:20px;color:#fff;letter-spacing:-0.5px;margin-bottom:4px;">
            Configuration
          </div>
          <div style="font-size:12px;color:rgba(167,199,231,0.55);margin-bottom:24px;letter-spacing:0.3px;">
            Tune the ML models
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin: 0 16px 8px;
                    background:rgba(167,199,231,0.07);
                    border:1px solid rgba(167,199,231,0.14);
                    border-radius:12px;padding:16px;
                    font-family:'Geist Sans',sans-serif;font-size:13px;line-height:2.1;">
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin: 0 16px 20px;
                    background:rgba(167,199,231,0.07);
                    border:1px solid rgba(167,199,231,0.14);
                    border-left:3px solid #008080;
                    border-radius:12px;padding:16px;
                    font-family:'Geist Sans',sans-serif;font-size:13px;line-height:2.2;color:rgba(228,235,255,0.8);">
            🎬 <b style="color:white;">{len(movies):,}</b> movies<br>
            ⭐ <b style="color:white;">{len(ratings):,}</b> ratings<br>
            👤 <b style="color:white;">{ratings['userId'].nunique():,}</b> users<br>
            🏷️ <b style="color:white;">{len(tags):,}</b> tags<br>
            📊 Avg: <b style="color:#A7C7E7;">{ratings['rating'].mean():.2f}★</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<div style="height:1px;background:rgba(167,199,231,0.12);margin:0 16px 16px;"></div>""", unsafe_allow_html=True)

        st.markdown("""<div style="padding:0 16px;font-family:'Geist Sans',sans-serif;font-size:11px;
                       letter-spacing:0.8px;text-transform:uppercase;color:rgba(167,199,231,0.5);
                       margin-bottom:12px;">Model Parameters</div>""", unsafe_allow_html=True)

        demo_user   = st.selectbox("Demo User ID", options=sorted(ratings["userId"].unique())[:100], index=0)
        n_neighbors = st.slider("User-CF Neighbors (K)", 5, 50, 20)
        n_recs      = st.slider("Recommendations (N)", 5, 20, 10)
        svd_factors = st.slider("SVD Latent Factors", 10, 100, 50)

        st.markdown("""<div style="height:1px;background:rgba(167,199,231,0.12);margin:8px 16px 16px;"></div>""", unsafe_allow_html=True)
        st.markdown("""<div style="padding:0 16px;font-family:'Geist Sans',sans-serif;font-size:11px;
                       letter-spacing:0.8px;text-transform:uppercase;color:rgba(167,199,231,0.5);
                       margin-bottom:12px;">Evaluation Settings</div>""", unsafe_allow_html=True)

        eval_k      = st.select_slider("Precision @ K", [5, 10, 15, 20], value=10)
        eval_thresh = st.select_slider("Relevance Threshold ★", [3.0, 3.5, 4.0, 4.5], value=4.0)
        eval_users  = st.slider("Test Users", 10, 60, 20)

    # ── Build matrix ───────────────────────────────────────────
    with st.spinner("Building user-item matrix…"):
        matrix    = build_matrix(ratings)
        user_ids  = list(matrix.index)
        movie_ids = list(matrix.columns)

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
        c5.metric("Avg Rating", f"{ratings['rating'].mean():.2f}★")

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
                         color_continuous_scale=[[0,"#e8f2fa"],[0.5,"#A7C7E7"],[1,"#191970"]])
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed", showgrid=False))
            fig.update_xaxes(gridcolor="rgba(25,25,112,0.06)", zeroline=False)
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
                .rename(columns={"title":"Title","avg_rating":"Avg ★","n_ratings":"# Ratings"}),
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
                .rename(columns={"title":"Title","n_ratings":"# Ratings","avg_rating":"Avg ★"}),
                use_container_width=True, hide_index=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 1 — USER-BASED CF
    # ══════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
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
            st.markdown(f"**Selected User:** `{demo_user}` &nbsp;·&nbsp; {int((matrix.loc[demo_user] != 0).sum())} movies rated")
            run_ucf = st.button("▶  Run User-CF Model", key="ucf_btn")

        if run_ucf:
            with st.spinner("Computing user similarity & generating recommendations…"):
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
                             color_continuous_scale=[[0,"#e8f2fa"],[0.5,"#A7C7E7"],[1,"#191970"]],
                             text="Similarity")
                fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
                fig.update_xaxes(type="category", showgrid=False)
                fig.update_yaxes(gridcolor="rgba(25,25,112,0.06)")
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
            run_icf_user = st.button("▶  Get Recommendations for User", key="icf_user_btn")

        with st.spinner("Computing item-item similarity matrix…"):
            item_sim = compute_item_similarity(matrix)

        if run_icf_user:
            with st.spinner("Generating item-based recommendations…"):
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
                if st.button("🔎 Find Similar Movies", key="icf_sim_btn"):
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
                                         color_continuous_scale=[[0,"#e8f2fa"],[0.5,"#A7C7E7"],[1,"#191970"]],
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
          <span>Decomposes the mean-centered user-item matrix R ≈ U·Σ·Vᵀ into k latent
          factors capturing hidden taste patterns. Predicted ratings are reconstructed
          as Û·Σ·Vᵀ + user_mean.</span>
        </div>
        """, unsafe_allow_html=True)

        run_svd = st.button("▶  Compute SVD & Recommend", key="svd_btn")

        if run_svd:
            with st.spinner(f"Computing SVD with k={svd_factors} latent factors…"):
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
                c3.metric("Top Σ Value", f"{s[0]:.1f}")
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
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom:20px;">
          <div class="section-heading">Explore the Dataset</div>
          <div class="section-subheading">Dive into individual movie profiles and user histories</div>
        </div>
        """, unsafe_allow_html=True)

        search_movie = st.text_input("Search for a movie profile", placeholder="e.g. Matrix, Inception…")

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
                c1.metric("Avg Rating", f"{m_ratings.mean():.2f}★" if len(m_ratings) else "N/A")
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
                                     color_continuous_scale=[[0,"#e8f2fa"],[0.5,"#A7C7E7"],[1,"#191970"]])
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
                              color_continuous_scale=[[0,"#e8f2fa"],[0.4,"#A7C7E7"],[0.7,"#191970"],[1,"#008080"]])
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
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.markdown("""
        <div class="tab-description-card">
          <b>Precision @ K Evaluation</b><br>
          <span>80/20 hold-out split per user. A recommendation is <i>relevant</i> if the user
          rated it ≥ threshold★ in the test set. Precision@K = hits in top-K / K.</span>
        </div>
        """, unsafe_allow_html=True)

        run_eval = st.button("▶  Run Full Evaluation", key="eval_btn")

        if run_eval:
            progress = st.progress(0, text="Preparing evaluation…")
            eligible   = ratings.groupby("userId").filter(lambda x: len(x) >= 30)["userId"].unique()
            test_users = eligible[:eval_users]

            with st.spinner("Computing similarities (cached after first run)…"):
                user_sim, umeans = compute_user_similarity(matrix)
                item_sim2        = compute_item_similarity(matrix)
                U2, s2, Vt2, m2 = compute_svd(matrix, 30)

            ucf_p, icf_p, svd_p = [], [], []
            ucf_r, icf_r, svd_r = [], [], []

            for i, uid in enumerate(test_users):
                progress.progress((i+1)/len(test_users),
                                   text=f"Evaluating user {uid} ({i+1}/{len(test_users)})…")
                ur  = ratings[ratings["userId"] == uid].sort_values("timestamp")
                sp  = int(len(ur) * 0.8)
                tr  = ur.iloc[sp:]
                rel = tr[tr["rating"] >= eval_thresh]["movieId"].tolist()
                if not rel: continue
                try:
                    r1 = ucf_recommend(uid, matrix, user_sim, user_ids, movie_ids,
                                       movies, n_neighbors, eval_k)
                    ucf_p.append(precision_at_k(r1.reset_index()["index"].tolist(), rel, eval_k))
                    ucf_r.append(recall_at_k(r1.reset_index()["index"].tolist(), rel, eval_k))
                except: pass
                try:
                    r2   = icf_recommend(uid, matrix, item_sim2, movie_ids, movies, eval_k)
                    ids2 = [movies[movies["title"]==t]["movieId"].values[0]
                            for t in r2["Title"].tolist() if len(movies[movies["title"]==t])>0]
                    icf_p.append(precision_at_k(ids2, rel, eval_k))
                    icf_r.append(recall_at_k(ids2, rel, eval_k))
                except: pass
                try:
                    r3   = svd_recommend(uid, U2, s2, Vt2, m2, user_ids, movie_ids, matrix, movies, eval_k)
                    ids3 = [movies[movies["title"]==t]["movieId"].values[0]
                            for t in r3["Title"].tolist() if len(movies[movies["title"]==t])>0]
                    svd_p.append(precision_at_k(ids3, rel, eval_k))
                    svd_r.append(recall_at_k(ids3, rel, eval_k))
                except: pass

            progress.empty()

            ucf_p2, icf_p2, svd_p2 = [], [], []
            for uid in test_users:
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
                              color_continuous_scale=[[0,"#e8f2fa"],[0.5,"#A7C7E7"],[1,"#191970"]],
                              text=f"Precision@{eval_k}")
            fig_comp.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
            fig_comp.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                                   yaxis=dict(gridcolor="rgba(25,25,112,0.06)"),
                                   xaxis=dict(showgrid=False))
            st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown("#### Evaluation Parameters Used")
            st.dataframe(pd.DataFrame([{
                "K (cutoff)": eval_k,
                "Relevance Threshold": f"≥ {eval_thresh}★",
                "Test Users": len(test_users),
                "UCF Neighbors": n_neighbors,
                "SVD Factors": 30,
            }]), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────
    st.markdown("""
    <div class="cineai-footer">
      CineAI &nbsp;·&nbsp; MovieLens ML Engine &nbsp;·&nbsp;
      <span>User-CF</span> &nbsp;·&nbsp;
      <span>Item-CF</span> &nbsp;·&nbsp;
      <span>SVD Matrix Factorization</span> &nbsp;·&nbsp;
      Precision@K &nbsp;·&nbsp; Built with Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
