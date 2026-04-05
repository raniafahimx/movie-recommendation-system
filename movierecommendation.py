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
    page_title="CineAI · Movie Recommendation Engine by R•F",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (white & blue, glassmorphism, futuristic) ───────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --blue-deep:   #0a1628;
    --blue-mid:    #1a3a6b;
    --blue-accent: #2563eb;
    --blue-light:  #3b82f6;
    --blue-pale:   #dbeafe;
    --blue-faint:  #eff6ff;
    --text-1: #0a1628;
    --text-2: #4b6394;
    --text-3: #8fa3c8;
    --glass:  rgba(255,255,255,0.60);
    --border: rgba(37,99,235,0.14);
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-1) !important;
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg,#eef4ff 0%,#f8faff 50%,#e8f0fe 100%);
    min-height: 100vh;
}

/* Subtle grid overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(37,99,235,.03) 1px,transparent 1px),
        linear-gradient(90deg,rgba(37,99,235,.03) 1px,transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.72) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid var(--border);
    box-shadow: 4px 0 32px rgba(37,99,235,0.06);
}
[data-testid="stSidebar"] * { color: var(--text-1) !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--blue-accent) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.35), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    transition: all 0.25s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(37,99,235,0.45) !important;
}

/* ── Selectbox / Slider labels ── */
.stSelectbox label, .stSlider label, .stTextInput label,
.stNumberInput label, .stMultiSelect label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--text-2) !important;
    text-transform: uppercase;
    letter-spacing: .5px;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-1) !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: var(--blue-accent) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.65) !important;
    backdrop-filter: blur(12px);
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.07) !important;
}
[data-testid="metric-container"] label {
    font-size: 11px !important;
    color: var(--text-3) !important;
    text-transform: uppercase;
    letter-spacing: .5px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 28px !important;
    color: var(--blue-accent) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.65) !important;
    backdrop-filter: blur(12px);
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 5px !important;
    gap: 4px !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.06) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    color: var(--text-2) !important;
    padding: 8px 18px !important;
    border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: var(--blue-accent) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.65) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--blue-accent) !important; }

/* ── Toast / info / success ── */
.stAlert { border-radius: 10px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--blue-faint); }
::-webkit-scrollbar-thumb { background: var(--blue-light); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
def render_header():
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.65);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.8);
        border-radius: 18px;
        padding: 28px 36px;
        margin-bottom: 28px;
        box-shadow: 0 8px 40px rgba(37,99,235,0.08);
        display: flex;
        align-items: center;
        gap: 20px;
    ">
        <div style="
            width:52px;height:52px;
            background:#2563eb;
            border-radius:14px;
            display:flex;align-items:center;justify-content:center;
            font-size:24px;
            box-shadow:0 4px 16px rgba(37,99,235,0.4);
            flex-shrink:0;
        "></div>
        <div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:26px;
                        color:#0a1628;letter-spacing:-1px;line-height:1;">CineAI</div>
            <div style="font-size:13px;color:#4b6394;font-weight:300;margin-top:3px;">
                Movie Recommendation Engine · MovieLens 100K · User-CF · Item-CF · SVD
            </div>
        </div>
        <div style="margin-left:auto;">
            <span style="
                padding:6px 14px;
                background:rgba(37,99,235,0.08);
                border:1px solid rgba(37,99,235,0.2);
                border-radius:20px;
                font-size:12px;font-weight:500;color:#2563eb;
                letter-spacing:.5px;
            ">ML RECOMMENDATION SYSTEM</span>
        </div>
    </div>
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
# CHART HELPERS
# ══════════════════════════════════════════════════════════════

BLUE_SEQ = ["#dbeafe","#93c5fd","#60a5fa","#3b82f6","#2563eb","#1d4ed8","#1e40af","#1e3a8a"]
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#4b6394"),
    margin=dict(l=0, r=0, t=30, b=0),
)

def styled_bar(df, x, y, title="", color_col=None, orientation="v"):
    fig = px.bar(df, x=x, y=y, title=title, orientation=orientation,
                 color_discrete_sequence=["#2563eb"],
                 text=y if orientation == "h" else None)
    fig.update_layout(**PLOTLY_LAYOUT, title_font=dict(family="Syne", size=14, color="#0a1628"))
    fig.update_traces(marker_line_width=0, textposition="outside")
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(37,99,235,0.08)", zeroline=False)
    return fig

def rating_bar_chart(df):
    colors = ["#93c5fd" if v < 4 else "#2563eb" for v in df["rating"]]
    fig = go.Figure(go.Bar(
        x=df["rating"].astype(str) + "★",
        y=df["count"],
        marker_color=colors,
        marker_line_width=0,
    ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(37,99,235,0.08)", zeroline=False)
    return fig


def singular_value_chart(s):
    total = (s**2).sum()
    cum   = np.cumsum(s**2) / total * 100
    k     = np.arange(1, len(s)+1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=k, y=s, name="Singular Value",
                         marker_color="#2563eb", marker_line_width=0,
                         yaxis="y"))
    fig.add_trace(go.Scatter(x=k, y=cum, name="Cumulative Variance %",
                             line=dict(color="#ef4444", width=2),
                             yaxis="y2"))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        yaxis=dict(title="Singular Value", gridcolor="rgba(37,99,235,0.08)"),
        yaxis2=dict(title="Cumul. Variance %", overlaying="y", side="right",
                    range=[0, 100], gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
        barmode="overlay",
    )
    return fig


def recommendation_chart(df, score_col="Predicted Rating"):
    fig = px.bar(df.head(10), x=score_col, y="Title", orientation="h",
                 color=score_col,
                 color_continuous_scale=[[0,"#dbeafe"],[0.5,"#60a5fa"],[1,"#1d4ed8"]],
                 text=score_col)
    fig.update_layout(**PLOTLY_LAYOUT,
                      yaxis=dict(autorange="reversed", showgrid=False),
                      xaxis=dict(gridcolor="rgba(37,99,235,0.08)", zeroline=False),
                      showlegend=False, coloraxis_showscale=False)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    return fig


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    render_header()

    # ── Load data ──────────────────────────────────────────────
    with st.spinner("Loading MovieLens dataset..."):
        try:
            movies, ratings, tags, links = load_data()
        except FileNotFoundError as e:
            st.error(f"Could not find dataset files at `{DATA_PATH}`\n\n{e}")
            st.stop()

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Syne',sans-serif;font-weight:800;
                    font-size:18px;color:#0a1628;margin-bottom:4px;">
            Configuration
        </div>
        <div style="font-size:12px;color:#8fa3c8;margin-bottom:20px;">
            Tune the ML models
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Dataset Stats**")
        st.markdown(f"""
        <div style="background:rgba(37,99,235,0.06);border:1px solid rgba(37,99,235,0.15);
                    border-radius:10px;padding:14px;font-size:13px;line-height:2;">
            <b>{len(movies):,}</b> movies<br>
            <b>{len(ratings):,}</b> ratings<br>
            <b>{ratings['userId'].nunique():,}</b> users<br>
            <b>{len(tags):,}</b> tags<br>
            Avg: <b>{ratings['rating'].mean():.2f}★</b>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("**Model Parameters**")
        demo_user = st.selectbox(
            "Demo User ID",
            options=sorted(ratings["userId"].unique())[:100],
            index=0,
        )
        n_neighbors = st.slider("User-CF Neighbors (K)", 5, 50, 20)
        n_recs      = st.slider("Recommendations (N)", 5, 20, 10)
        svd_factors = st.slider("SVD Latent Factors", 10, 100, 50)

        st.divider()
        st.markdown("**Evaluation Settings**")
        eval_k     = st.select_slider("Precision @ K", [5, 10, 15, 20], value=10)
        eval_thresh = st.select_slider("Relevance Threshold ★", [3.0, 3.5, 4.0, 4.5], value=4.0)
        eval_users  = st.slider("Test Users", 10, 60, 20)

    # ── Build matrix & models ──────────────────────────────────
    with st.spinner("Building user-item matrix..."):
        matrix     = build_matrix(ratings)
        user_ids   = list(matrix.index)
        movie_ids  = list(matrix.columns)

    # ── Tabs ───────────────────────────────────────────────────
    tabs = st.tabs([
        "•Overview•",
        "•User-Based CF•",
        "•Item-Based CF•",
        "•SVD Factorization•",
        "•Explore Dataset•",
        "•Evaluation•",
    ])


    # ══════════════════════════════════════════════════════════
    # TAB 0 — OVERVIEW
    # ══════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("### Dataset at a Glance")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Movies", f"{len(movies):,}")
        c2.metric("Ratings", f"{len(ratings):,}")
        c3.metric("Users", f"{ratings['userId'].nunique():,}")
        c4.metric("Tags", f"{len(tags):,}")
        c5.metric("Avg Rating", f"{ratings['rating'].mean():.2f}★")

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Rating Distribution")
            rdist = (ratings["rating"].value_counts()
                     .sort_index().reset_index())
            rdist.columns = ["rating", "count"]
            st.plotly_chart(rating_bar_chart(rdist), use_container_width=True)

        with col_right:
            st.markdown("#### Top 15 Genres")
            gc = defaultdict(int)
            for gs in movies["genres"].dropna():
                for g in gs.split("|"):
                    if g and g != "(no genres listed)":
                        gc[g] += 1
            gdf = pd.DataFrame(list(gc.items()), columns=["Genre","Count"])\
                    .sort_values("Count", ascending=False).head(15)
            fig = px.bar(gdf, x="Count", y="Genre", orientation="h",
                         color="Count",
                         color_continuous_scale=[[0,"#dbeafe"],[1,"#1d4ed8"]])
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed", showgrid=False))
            fig.update_xaxes(gridcolor="rgba(37,99,235,0.08)", zeroline=False)
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


    # ══════════════════════════════════════════════════════════
    # TAB 1 — USER-BASED CF
    # ══════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("""
        <div style="background:rgba(37,99,235,0.05);border:1px solid rgba(37,99,235,0.15);
                    border-radius:12px;padding:16px 20px;margin-bottom:20px;">
            <b style="font-family:'Syne',sans-serif;color:#0a1628;">User-Based Collaborative Filtering</b><br>
            <span style="font-size:13px;color:#4b6394;">
            Finds users with similar rating patterns (Pearson similarity on mean-centered vectors),
            then predicts ratings for unseen movies using a weighted average of neighbor ratings.
            </span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**Selected User:** `{demo_user}` · {int((matrix.loc[demo_user] != 0).sum())} movies rated")
            run_ucf = st.button("▶ Run User-CF Model", key="ucf_btn")

        if run_ucf:
            with st.spinner("Computing user similarity & generating recommendations..."):
                user_sim, umeans = compute_user_similarity(matrix)
                recs = ucf_recommend(demo_user, matrix, user_sim, user_ids,
                                     movie_ids, movies, n_neighbors, n_recs)

                # Similar users
                u_idx   = user_ids.index(demo_user)
                sims    = user_sim[u_idx]
                top_nb  = np.argsort(sims)[::-1][:8]
                sim_df  = pd.DataFrame([{
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
                             color_continuous_scale=[[0,"#dbeafe"],[1,"#1d4ed8"]],
                             text="Similarity")
                fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig.update_xaxes(type="category", showgrid=False)
                fig.update_yaxes(gridcolor="rgba(37,99,235,0.08)")
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


    # ══════════════════════════════════════════════════════════
    # TAB 2 — ITEM-BASED CF
    # ══════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("""
        <div style="background:rgba(37,99,235,0.05);border:1px solid rgba(37,99,235,0.15);
                    border-radius:12px;padding:16px 20px;margin-bottom:20px;">
            <b style="font-family:'Syne',sans-serif;color:#0a1628;">Item-Based Collaborative Filtering</b><br>
            <span style="font-size:13px;color:#4b6394;">
            Computes cosine similarity between movie rating vectors.
            Scores unseen movies by their similarity to movies the user has already rated highly.
            </span>
        </div>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns([1, 1])
        with col_l:
            search_query = st.text_input("Search a movie to find similar titles", placeholder="e.g. Toy Story")
            run_icf_user = st.button("▶ Get Recommendations for User", key="icf_user_btn")

        with st.spinner("Computing item-item similarity matrix..."):
            item_sim = compute_item_similarity(matrix)

        if run_icf_user:
            with st.spinner("Generating item-based recommendations..."):
                icf_recs = icf_recommend(demo_user, matrix, item_sim, movie_ids, movies, n_recs)

            st.divider()
            st.markdown(f"#### Item-CF Recommendations for User {demo_user}")
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
                            fig = px.bar(sim_movies, x="Similarity", y="Title",
                                         orientation="h",
                                         color="Similarity",
                                         color_continuous_scale=[[0,"#dbeafe"],[1,"#1d4ed8"]],
                                         text="Similarity")
                            fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                                              yaxis=dict(autorange="reversed", showgrid=False))
                            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                            st.plotly_chart(fig, use_container_width=True)
                        with right:
                            st.dataframe(sim_movies, use_container_width=True, hide_index=True)

                        # Genre breakdown
                        st.markdown("#### Genre Distribution of Similar Movies")
                        gcount = defaultdict(int)
                        for gs in sim_movies["Genres"].dropna():
                            for g in gs.split("|"):
                                if g: gcount[g] += 1
                        gdf2 = pd.DataFrame(list(gcount.items()), columns=["Genre","Count"])\
                                 .sort_values("Count", ascending=False)
                        fig2 = px.pie(gdf2, names="Genre", values="Count",
                                      color_discrete_sequence=BLUE_SEQ)
                        fig2.update_layout(**PLOTLY_LAYOUT)
                        st.plotly_chart(fig2, use_container_width=True)


    # ══════════════════════════════════════════════════════════
    # TAB 3 — SVD
    # ══════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("""
        <div style="background:rgba(37,99,235,0.05);border:1px solid rgba(37,99,235,0.15);
                    border-radius:12px;padding:16px 20px;margin-bottom:20px;">
            <b style="font-family:'Syne',sans-serif;color:#0a1628;">SVD Matrix Factorization</b><br>
            <span style="font-size:13px;color:#4b6394;">
            Decomposes the mean-centered user-item matrix R ≈ U·Σ·Vᵀ into k latent factors
            capturing hidden patterns of taste. Predicted ratings are reconstructed as Û·Σ·Vᵀ + user_mean.
            </span>
        </div>
        """, unsafe_allow_html=True)

        run_svd = st.button("▶ Compute SVD & Recommend", key="svd_btn")

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
                cum   = (s**2).sum() / total * 100
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
                st.markdown(f"#### SVD Recommendations for User {demo_user}")
                if svd_recs.empty:
                    st.info("No recommendations found.")
                else:
                    st.plotly_chart(recommendation_chart(svd_recs), use_container_width=True)
                    st.dataframe(svd_recs, use_container_width=True, hide_index=True)


    # ══════════════════════════════════════════════════════════
    # TAB 4 — EXPLORE
    # ══════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### Explore the Dataset")

        search_movie = st.text_input("Search for a movie profile", placeholder="e.g. Matrix, Inception...")

        if search_movie:
            matches = movies[movies["title"].str.contains(search_movie, case=False, na=False)]
            if not matches.empty:
                sel = st.selectbox("Pick a movie:", matches["title"].tolist(), key="explore_sel")
                sel_id = matches[matches["title"] == sel]["movieId"].values[0]
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
                        tdf = m_tags.reset_index()
                        tdf.columns = ["Tag","Uses"]
                        fig = px.bar(tdf, x="Uses", y="Tag", orientation="h",
                                     color="Uses",
                                     color_continuous_scale=[[0,"#dbeafe"],[1,"#1d4ed8"]])
                        fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                                          yaxis=dict(autorange="reversed", showgrid=False))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No tags for this movie.")

        st.divider()
        st.markdown("### Top 40 User Tags")
        all_tags = tags["tag"].str.lower().value_counts().head(40).reset_index()
        all_tags.columns = ["Tag","Uses"]
        fig_tags = px.treemap(all_tags, path=["Tag"], values="Uses",
                              color="Uses",
                              color_continuous_scale=[[0,"#dbeafe"],[0.5,"#60a5fa"],[1,"#1d4ed8"]])
        fig_tags.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_tags, use_container_width=True)

        st.divider()
        st.markdown("### User Rating Profile")
        explore_user = st.selectbox("Pick a user:", sorted(ratings["userId"].unique())[:100],
                                    key="explore_user")
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


    # ══════════════════════════════════════════════════════════
    # TAB 5 — EVALUATION
    # ══════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("""
        <div style="background:rgba(37,99,235,0.05);border:1px solid rgba(37,99,235,0.15);
                    border-radius:12px;padding:16px 20px;margin-bottom:20px;">
            <b style="font-family:'Syne',sans-serif;color:#0a1628;">Precision @ K Evaluation</b><br>
            <span style="font-size:13px;color:#4b6394;">
            80/20 hold-out split per user. A recommendation is <i>relevant</i> if the user rated it
            ≥ threshold ★ in the test set. Precision@K = hits in top-K / K.
            </span>
        </div>
        """, unsafe_allow_html=True)

        run_eval = st.button("▶ Run Full Evaluation", key="eval_btn")

        if run_eval:
            progress = st.progress(0, text="Preparing evaluation...")
            eligible = (ratings.groupby("userId")
                        .filter(lambda x: len(x) >= 30)["userId"].unique())
            test_users = eligible[:eval_users]

            with st.spinner("Computing similarities (cached after first run)..."):
                user_sim, umeans = compute_user_similarity(matrix)
                item_sim2        = compute_item_similarity(matrix)
                U2, s2, Vt2, m2 = compute_svd(matrix, 30)

            ucf_p, icf_p, svd_p = [], [], []
            ucf_r, icf_r, svd_r = [], [], []

            for i, uid in enumerate(test_users):
                progress.progress((i+1)/len(test_users),
                                   text=f"Evaluating user {uid} ({i+1}/{len(test_users)})...")
                ur = ratings[ratings["userId"] == uid].sort_values("timestamp")
                sp = int(len(ur) * 0.8)
                tr = ur.iloc[sp:]
                rel = tr[tr["rating"] >= eval_thresh]["movieId"].tolist()
                if not rel: continue
                try:
                    r1 = ucf_recommend(uid, matrix, user_sim, user_ids, movie_ids,
                                       movies, n_neighbors, eval_k)
                    ucf_p.append(precision_at_k(r1["movie_id"].tolist() if "movie_id" in r1 else
                                                r1.reset_index()["index"].tolist(), rel, eval_k))
                    ucf_r.append(recall_at_k(r1["movie_id"].tolist() if "movie_id" in r1 else
                                             r1.reset_index()["index"].tolist(), rel, eval_k))
                except: pass
                try:
                    r2 = icf_recommend(uid, matrix, item_sim2, movie_ids, movies, eval_k)
                    icf_p.append(precision_at_k([], rel, eval_k))
                    icf_r.append(0.0)
                except: pass
                try:
                    r3 = svd_recommend(uid, U2, s2, Vt2, m2, user_ids, movie_ids,
                                       matrix, movies, eval_k)
                    svd_p.append(precision_at_k([], rel, eval_k))
                    svd_r.append(0.0)
                except: pass

            progress.empty()

            # Recompute properly
            ucf_p2, icf_p2, svd_p2 = [], [], []
            for uid in test_users:
                ur = ratings[ratings["userId"] == uid].sort_values("timestamp")
                sp = int(len(ur) * 0.8)
                train_ids = set(ur.iloc[:sp]["movieId"])
                test_r    = ur.iloc[sp:]
                rel       = test_r[test_r["rating"] >= eval_thresh]["movieId"].tolist()
                if not rel: continue
                try:
                    recs1 = ucf_recommend(uid, matrix, user_sim, user_ids,
                                          movie_ids, movies, n_neighbors, eval_k)
                    ids1  = [movies[movies["title"] == t]["movieId"].values[0]
                             for t in recs1["Title"].tolist()
                             if len(movies[movies["title"] == t]) > 0]
                    ucf_p2.append(precision_at_k(ids1, rel, eval_k))
                except: pass
                try:
                    recs2 = icf_recommend(uid, matrix, item_sim2, movie_ids, movies, eval_k)
                    ids2  = [movies[movies["title"] == t]["movieId"].values[0]
                             for t in recs2["Title"].tolist()
                             if len(movies[movies["title"] == t]) > 0]
                    icf_p2.append(precision_at_k(ids2, rel, eval_k))
                except: pass
                try:
                    recs3 = svd_recommend(uid, U2, s2, Vt2, m2, user_ids,
                                          movie_ids, matrix, movies, eval_k)
                    ids3  = [movies[movies["title"] == t]["movieId"].values[0]
                             for t in recs3["Title"].tolist()
                             if len(movies[movies["title"] == t]) > 0]
                    svd_p2.append(precision_at_k(ids3, rel, eval_k))
                except: pass

            avg_ucf = np.mean(ucf_p2) if ucf_p2 else 0.0
            avg_icf = np.mean(icf_p2) if icf_p2 else 0.0
            avg_svd = np.mean(svd_p2) if svd_p2 else 0.0
            best    = max(["User-CF","Item-CF","SVD"],
                          key=lambda m: {"User-CF":avg_ucf,"Item-CF":avg_icf,"SVD":avg_svd}[m])

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"User-CF Precision@{eval_k}", f"{avg_ucf*100:.2f}%")
            c2.metric(f"Item-CF Precision@{eval_k}", f"{avg_icf*100:.2f}%")
            c3.metric(f"SVD Precision@{eval_k}",     f"{avg_svd*100:.2f}%")
            c4.metric("Best Model", best)

            st.divider()
            st.markdown("#### Algorithm Comparison")
            comp_df = pd.DataFrame({
                "Model":       ["User-CF", "Item-CF", "SVD"],
                f"Precision@{eval_k}": [avg_ucf, avg_icf, avg_svd],
            })
            fig_comp = px.bar(comp_df, x="Model", y=f"Precision@{eval_k}",
                              color=f"Precision@{eval_k}",
                              color_continuous_scale=[[0,"#dbeafe"],[1,"#1d4ed8"]],
                              text=f"Precision@{eval_k}")
            fig_comp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_comp.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                                   yaxis=dict(gridcolor="rgba(37,99,235,0.08)"),
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


    # ── Footer ─────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="text-align:center;font-size:12px;color:#8fa3c8;padding:8px;">
        CineAI · MovieLens ML Engine ·
        User-Based CF · Item-Based CF · SVD Matrix Factorization · Precision@K
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
