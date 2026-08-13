import io
import re
import html
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Social Text Intelligence Lab", page_icon="🧠", layout="wide")

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
      [data-testid="stMetricValue"] {font-size: 1.65rem;}
      .hero {
        padding: 1.15rem 1.35rem; border: 1px solid rgba(128,128,128,.20);
        border-radius: 18px; margin-bottom: .8rem;
        background: linear-gradient(120deg, rgba(79,70,229,.10), rgba(14,165,233,.06));
      }
      .small-note {opacity:.74; font-size:.9rem;}
      .topic-card {padding:.8rem 1rem;border:1px solid rgba(128,128,128,.2);border-radius:14px;margin:.35rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='hero'>
      <h1 style='margin:0'>🧠 Social Text Intelligence Lab</h1>
      <p style='margin:.35rem 0 0 0'>Interactive topic modelling, sentiment analysis and text mining for social-media and review data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Subject lenses
# -----------------------------
SUBJECT_LENSES = {
    "Brand Personalisation": {
        "description": "How customers discuss tailoring, relevance, individual treatment, recommendations and customised experiences.",
        "keywords": [
            "personalized", "personalised", "personalization", "personalisation", "personal", "customized",
            "customised", "customization", "customisation", "tailored", "tailor", "recommendation",
            "recommended", "relevant", "preference", "preferences", "individual", "individualized",
            "individualised", "for me", "my needs", "my taste", "profile", "targeted", "unique experience",
            "recognize me", "recognise me", "suggested", "algorithm", "recommendations"
        ],
    },
    "Value Co-Creation": {
        "description": "How customers participate in creating value through interaction, feedback, collaboration and shared experience.",
        "keywords": [
            "co-create", "co creation", "cocreation", "collaborate", "collaboration", "participate", "participation",
            "community", "feedback", "contribute", "contribution", "together", "interaction", "engage", "engagement",
            "involve", "involvement", "customer input", "user input", "suggestion", "ideas", "shared value",
            "experience", "relationship", "dialogue", "conversation", "help improve", "build together"
        ],
    },
    "Brand Salience": {
        "description": "How strongly and readily a brand is noticed, recalled, recognised and mentally available in relevant situations.",
        "keywords": [
            "remember", "recall", "recognize", "recognise", "recognition", "awareness", "aware", "notice", "noticed",
            "top of mind", "first brand", "think of", "familiar", "familiarity", "memorable", "iconic", "visible",
            "visibility", "brand name", "logo", "distinctive", "stand out", "stands out", "mental availability",
            "comes to mind", "heard of", "known brand"
        ],
    },
    "Custom Lens": {
        "description": "Create your own construct dictionary for exploratory theory-driven analysis.",
        "keywords": [],
    },
}

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv_bytes(raw: bytes) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

@st.cache_data(show_spinner=False)
def load_csv_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def likely_text_columns(df: pd.DataFrame):
    candidates = []
    preferred = ["text", "comment", "comments", "review", "content", "body", "message", "caption", "title", "post"]
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().astype(str).head(500)
            avg_len = sample.str.len().mean() if len(sample) else 0
            score = avg_len
            if str(col).lower().strip() in preferred:
                score += 1000
            if any(k in str(col).lower() for k in preferred):
                score += 300
            candidates.append((col, score))
    return [c for c, _ in sorted(candidates, key=lambda x: x[1], reverse=True)]

def clean_text(text: str, keep_hashtags=True) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    if keep_hashtags:
        text = re.sub(r"#(\w+)", r" \1 ", text)
    else:
        text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^\w\s'’-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"[_\d]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def sentiment_label(score):
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"

@st.cache_data(show_spinner=False)
def add_sentiment(df: pd.DataFrame, text_col: str):
    analyzer = SentimentIntensityAnalyzer()
    out = df.copy()
    scores = out[text_col].fillna("").astype(str).apply(analyzer.polarity_scores)
    out["sentiment_compound"] = scores.apply(lambda x: x["compound"])
    out["sentiment_positive"] = scores.apply(lambda x: x["pos"])
    out["sentiment_neutral"] = scores.apply(lambda x: x["neu"])
    out["sentiment_negative"] = scores.apply(lambda x: x["neg"])
    out["sentiment_label"] = out["sentiment_compound"].apply(sentiment_label)
    return out

def extract_ngrams(texts, ngram_range=(1, 1), top_n=25, min_df=2, max_df=0.95):
    try:
        vec = CountVectorizer(stop_words="english", ngram_range=ngram_range, min_df=min_df, max_df=max_df)
        X = vec.fit_transform(texts)
        counts = np.asarray(X.sum(axis=0)).ravel()
        terms = vec.get_feature_names_out()
        idx = counts.argsort()[::-1][:top_n]
        return pd.DataFrame({"term": terms[idx], "frequency": counts[idx]})
    except ValueError:
        return pd.DataFrame(columns=["term", "frequency"])

def topic_terms(model, feature_names, n_top_words=10):
    records = []
    for topic_idx, topic in enumerate(model.components_):
        top_ids = topic.argsort()[::-1][:n_top_words]
        for rank, idx in enumerate(top_ids, 1):
            records.append({"topic": topic_idx + 1, "rank": rank, "term": feature_names[idx], "weight": float(topic[idx])})
    return pd.DataFrame(records)

@st.cache_data(show_spinner=False)
def run_sklearn_topics(texts, method, n_topics, n_words, min_df, max_df, max_features):
    texts = pd.Series(texts).fillna("").astype(str).tolist()
    if method == "LDA":
        vectorizer = CountVectorizer(stop_words="english", min_df=min_df, max_df=max_df, max_features=max_features, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch", max_iter=20)
        doc_topic = model.fit_transform(X)
    else:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=min_df, max_df=max_df, max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
        X = vectorizer.fit_transform(texts)
        model = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=400)
        doc_topic = model.fit_transform(X)
    terms = topic_terms(model, vectorizer.get_feature_names_out(), n_words)
    dominant = doc_topic.argmax(axis=1) + 1
    strength = doc_topic.max(axis=1)
    return terms, doc_topic, dominant, strength

def subject_relevance(texts, keywords):
    if not keywords:
        return np.zeros(len(texts)), np.zeros(len(texts), dtype=int)
    docs = pd.Series(texts).fillna("").astype(str).str.lower().tolist()
    keyword_doc = " ".join(keywords)
    corpus = docs + [keyword_doc]
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(corpus)
        semantic = cosine_similarity(X[:-1], X[-1]).ravel()
    except Exception:
        semantic = np.zeros(len(docs))
    pattern = re.compile("|".join(re.escape(k.lower()) for k in sorted(keywords, key=len, reverse=True)), flags=re.I)
    counts = np.array([len(pattern.findall(t)) for t in docs])
    if semantic.max() > 0:
        semantic = semantic / semantic.max()
    lexical = np.log1p(counts)
    if lexical.max() > 0:
        lexical = lexical / lexical.max()
    score = 100 * (0.65 * semantic + 0.35 * lexical)
    return np.round(score, 1), counts

def make_download(df):
    return df.to_csv(index=False).encode("utf-8")

# -----------------------------
# Sidebar: data and settings
# -----------------------------
with st.sidebar:
    st.header("1 · Data")
    source_mode = st.radio("Load CSV from", ["Upload file", "CSV URL"], horizontal=True)
    df = None
    data_name = None
    try:
        if source_mode == "Upload file":
            uploaded = st.file_uploader("Choose a CSV file", type=["csv"], help="Upload scraped/exported social-media, review or forum text data.")
            if uploaded is not None:
                df = load_csv_bytes(uploaded.getvalue())
                data_name = uploaded.name
        else:
            url = st.text_input("Public CSV URL", placeholder="https://.../data.csv")
            if url:
                parsed = urlparse(url)
                if parsed.scheme in ("http", "https"):
                    df = load_csv_url(url)
                    data_name = url.split("/")[-1] or "remote.csv"
                else:
                    st.warning("Please use an http(s) CSV URL.")
    except Exception as e:
        st.error(f"Could not load CSV: {e}")

    st.header("2 · Research lens")
    lens_name = st.radio("Subject area", list(SUBJECT_LENSES.keys()))
    lens = SUBJECT_LENSES[lens_name].copy()
    if lens_name == "Custom Lens":
        custom_name = st.text_input("Custom construct name", value="Custom construct")
        custom_kw = st.text_area("Keywords / phrases (comma-separated)", placeholder="keyword, phrase, another term")
        lens["description"] = f"Custom construct: {custom_name}"
        lens["keywords"] = [x.strip() for x in custom_kw.split(",") if x.strip()]
    st.caption(lens["description"])

if df is None:
    st.info("Upload a CSV (or provide a public CSV URL) to begin. Your file should contain at least one column with social-media posts, comments, captions or reviews.")
    st.markdown("**Good input examples:** YouTube comments, Reddit posts/comments, App Store/Google Play reviews, Trustpilot reviews, X/Twitter exports, Instagram captions/comments, Facebook comments, survey verbatims or customer-support text.")
    st.stop()

if df.empty:
    st.error("The CSV is empty.")
    st.stop()

text_candidates = likely_text_columns(df)
if not text_candidates:
    st.error("No text-like columns were detected in the CSV.")
    st.stop()

with st.sidebar:
    st.header("3 · Analysis setup")
    text_col = st.selectbox("Text column", df.columns.tolist(), index=df.columns.get_loc(text_candidates[0]) if text_candidates[0] in df.columns else 0)
    group_candidates = ["None"] + [c for c in df.columns if c != text_col]
    default_group_idx = 0
    for i, c in enumerate(group_candidates):
        if str(c).lower() in {"platform", "source", "channel", "brand", "product", "app"}:
            default_group_idx = i; break
    group_col = st.selectbox("Optional grouping column", group_candidates, index=default_group_idx)
    keep_hashtags = st.checkbox("Keep hashtag words", value=True)

# Base processing
work = df.copy()
work[text_col] = work[text_col].fillna("").astype(str)
work = work[work[text_col].str.strip().ne("")].copy()
work["clean_text"] = work[text_col].apply(lambda x: clean_text(x, keep_hashtags=keep_hashtags))
work["word_count"] = work["clean_text"].str.split().str.len()
work["character_count"] = work[text_col].str.len()
work = add_sentiment(work, text_col)
rel_score, rel_hits = subject_relevance(work["clean_text"], lens["keywords"])
work["subject_relevance"] = rel_score
work["subject_keyword_hits"] = rel_hits

# Filters
with st.sidebar:
    st.header("4 · Filters")
    min_words = int(work["word_count"].min()) if len(work) else 0
    max_words = int(work["word_count"].max()) if len(work) else 1
    if max_words > min_words:
        word_range = st.slider("Word count", min_words, max_words, (min_words, max_words))
    else:
        word_range = (min_words, max_words)
    sentiments = st.multiselect("Sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
    relevance_min = st.slider("Minimum subject relevance", 0, 100, 0, help="Theory-guided relevance score based on the selected construct dictionary.")

filtered = work[
    work["word_count"].between(word_range[0], word_range[1])
    & work["sentiment_label"].isin(sentiments)
    & (work["subject_relevance"] >= relevance_min)
].copy()

if filtered.empty:
    st.warning("No rows remain after filtering. Adjust the filters in the sidebar.")
    st.stop()

# -----------------------------
# KPI strip
# -----------------------------
cols = st.columns(6)
cols[0].metric("Texts", f"{len(filtered):,}")
cols[1].metric("Avg words", f"{filtered['word_count'].mean():.1f}")
cols[2].metric("Positive", f"{(filtered['sentiment_label']=='Positive').mean():.0%}")
cols[3].metric("Negative", f"{(filtered['sentiment_label']=='Negative').mean():.0%}")
cols[4].metric("Avg sentiment", f"{filtered['sentiment_compound'].mean():.2f}")
cols[5].metric(f"{lens_name} ≥50", f"{(filtered['subject_relevance']>=50).mean():.0%}")

st.caption(f"Dataset: **{data_name}** · Analysing **{text_col}** · Lens: **{lens_name}** · {len(filtered):,} of {len(work):,} usable texts currently shown")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "⛏️ Text Mining", "🙂 Sentiment", "🧩 Topic Modelling", "🎯 Subject Lens", "🔎 Data Explorer"
])

with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        sent_counts = filtered["sentiment_label"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        fig = px.bar(sent_counts, x="Sentiment", y="Count", title="Sentiment distribution", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(filtered, x="word_count", nbins=30, title="Text length distribution", labels={"word_count":"Words per text"})
        st.plotly_chart(fig, use_container_width=True)

    if group_col != "None":
        g = filtered[group_col].fillna("Missing").astype(str)
        top_groups = g.value_counts().head(20).index
        tmp = filtered[g.isin(top_groups)].copy()
        tmp[group_col] = tmp[group_col].fillna("Missing").astype(str)
        summary = tmp.groupby(group_col).agg(
            texts=(text_col, "size"), avg_sentiment=("sentiment_compound", "mean"), avg_relevance=("subject_relevance", "mean")
        ).reset_index().sort_values("texts", ascending=False)
        fig = px.scatter(summary, x="avg_sentiment", y="avg_relevance", size="texts", hover_name=group_col,
                         title=f"Groups: sentiment × {lens_name} relevance", labels={"avg_sentiment":"Average sentiment", "avg_relevance":"Average relevance"})
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Term and phrase mining")
    c1, c2, c3 = st.columns(3)
    ngram_n = c1.selectbox("Phrase length", [1, 2, 3], format_func=lambda x: {1:"Unigrams",2:"Bigrams",3:"Trigrams"}[x])
    top_n = c2.slider("Top terms", 10, 60, 25)
    min_df_tm = c3.slider("Minimum document frequency", 1, min(20, max(1, len(filtered)//10)), 2 if len(filtered) >= 20 else 1)
    terms = extract_ngrams(filtered["clean_text"], (ngram_n, ngram_n), top_n=top_n, min_df=min_df_tm)
    c1, c2 = st.columns([1.15, .85])
    with c1:
        if len(terms):
            fig = px.bar(terms.sort_values("frequency"), x="frequency", y="term", orientation="h", title="Most frequent terms / phrases")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough repeated terms for the current settings.")
    with c2:
        st.markdown("#### Word cloud")
        cloud_text = " ".join(filtered["clean_text"].tolist())
        if cloud_text.strip():
            wc = WordCloud(width=1000, height=650, background_color="white", stopwords=set(), collocations=False, max_words=150).generate(cloud_text)
            fig_wc, ax = plt.subplots(figsize=(9, 5.8))
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            st.pyplot(fig_wc, use_container_width=True)
            plt.close(fig_wc)

    st.markdown("#### Terms associated with positive vs negative texts")
    pos_terms = extract_ngrams(filtered.loc[filtered.sentiment_label=="Positive", "clean_text"], (1,2), 15, 1)
    neg_terms = extract_ngrams(filtered.loc[filtered.sentiment_label=="Negative", "clean_text"], (1,2), 15, 1)
    a,b = st.columns(2)
    a.dataframe(pos_terms, use_container_width=True, hide_index=True)
    b.dataframe(neg_terms, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Sentiment intelligence")
    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered, x="sentiment_compound", nbins=40, title="VADER compound sentiment", labels={"sentiment_compound":"Compound score (−1 to +1)"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(filtered, x="sentiment_label", y="sentiment_compound", points="outliers", title="Sentiment score by class")
        st.plotly_chart(fig, use_container_width=True)

    if group_col != "None":
        agg = filtered.groupby(group_col).agg(texts=(text_col,"size"), sentiment=("sentiment_compound","mean")).reset_index()
        agg = agg[agg.texts >= max(1, int(np.percentile(agg.texts, 25)))].sort_values("sentiment")
        fig = px.bar(agg.tail(25), x="sentiment", y=group_col, orientation="h", title=f"Average sentiment by {group_col}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Most positive and most negative texts")
    c1,c2 = st.columns(2)
    show_cols = [text_col, "sentiment_compound", "subject_relevance"]
    c1.dataframe(filtered.nlargest(10, "sentiment_compound")[show_cols], hide_index=True, use_container_width=True)
    c2.dataframe(filtered.nsmallest(10, "sentiment_compound")[show_cols], hide_index=True, use_container_width=True)

with tab4:
    st.subheader("Unsupervised topic discovery")
    st.caption("Use NMF for fast, interpretable TF-IDF topics; use LDA for probabilistic topics. BERTopic can be enabled in the Advanced deployment described in the README.")
    c1,c2,c3,c4 = st.columns(4)
    method = c1.radio("Model", ["NMF", "LDA"], horizontal=True)
    upper_topics = min(20, max(2, len(filtered)//5))
    n_topics = c2.slider("Number of topics", 2, upper_topics, min(6, upper_topics))
    n_words = c3.slider("Words per topic", 5, 20, 10)
    max_features = c4.selectbox("Vocabulary cap", [2000, 5000, 10000, 20000], index=1)
    min_df_topic = 2 if len(filtered) >= 30 else 1

    if len(filtered) < max(10, n_topics*2):
        st.warning("Topic models are more stable with more documents. Consider using at least 20–50 texts for exploratory modelling and substantially more for research inference.")

    if st.button("Run topic model", type="primary", use_container_width=True):
        try:
            with st.spinner("Fitting topic model…"):
                terms_df, doc_topic, dominant, strength = run_sklearn_topics(
                    filtered["clean_text"], method, n_topics, n_words, min_df_topic, 0.95, max_features
                )
            st.session_state["topic_result"] = {
                "signature": (method, n_topics, n_words, len(filtered), lens_name),
                "terms": terms_df, "doc_topic": doc_topic, "dominant": dominant, "strength": strength,
                "index": filtered.index.tolist(),
            }
        except Exception as e:
            st.error(f"Topic model could not be fitted: {e}")

    result = st.session_state.get("topic_result")
    if result:
        topic_terms_df = result["terms"]
        dominant = result["dominant"]
        strength = result["strength"]
        doc_topic = result["doc_topic"]
        if len(dominant) == len(filtered):
            topic_data = filtered.copy()
            topic_data["topic"] = dominant
            topic_data["topic_strength"] = strength
            labels = topic_terms_df.groupby("topic")["term"].apply(lambda s: ", ".join(s.head(5))).to_dict()
            topic_data["topic_label"] = topic_data["topic"].map(lambda x: f"T{x}: {labels.get(x,'')}")

            counts = topic_data["topic_label"].value_counts().reset_index()
            counts.columns = ["Topic", "Texts"]
            fig = px.bar(counts.sort_values("Texts"), x="Texts", y="Topic", orientation="h", title="Topic prevalence")
            st.plotly_chart(fig, use_container_width=True)

            selected_topic = st.selectbox("Explore a topic", sorted(topic_data["topic"].unique()), format_func=lambda x: f"Topic {x} — {labels.get(x,'')}")
            tt = topic_terms_df[topic_terms_df.topic == selected_topic].sort_values("weight")
            c1,c2 = st.columns([.8,1.2])
            with c1:
                fig = px.bar(tt, x="weight", y="term", orientation="h", title=f"Topic {selected_topic} terms")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("#### Representative texts")
                rep = topic_data[topic_data.topic == selected_topic].nlargest(12, "topic_strength")[[text_col,"topic_strength","sentiment_compound","subject_relevance"]]
                st.dataframe(rep, hide_index=True, use_container_width=True)

            # Topic × sentiment and relevance
            agg = topic_data.groupby("topic_label").agg(texts=(text_col,"size"), sentiment=("sentiment_compound","mean"), relevance=("subject_relevance","mean")).reset_index()
            fig = px.scatter(agg, x="sentiment", y="relevance", size="texts", hover_name="topic_label", title=f"Topics: sentiment × {lens_name} relevance")
            st.plotly_chart(fig, use_container_width=True)

            # 2D document map from topic-membership matrix
            if doc_topic.shape[1] >= 2 and len(topic_data) >= 3:
                coords = PCA(n_components=2, random_state=42).fit_transform(normalize(doc_topic))
                map_df = pd.DataFrame({"x":coords[:,0], "y":coords[:,1], "Topic":topic_data["topic_label"].values,
                                       "Sentiment":topic_data["sentiment_label"].values,
                                       "Relevance":topic_data["subject_relevance"].values,
                                       "Text":topic_data[text_col].str.slice(0,180).values})
                fig = px.scatter(map_df, x="x", y="y", color="Topic", hover_data={"Text":True,"Sentiment":True,"Relevance":True,"x":False,"y":False},
                                 title="Interactive document-topic map (PCA of topic memberships)")
                st.plotly_chart(fig, use_container_width=True)

            export_topics = topic_data.copy()
            st.download_button("⬇️ Download topic-assigned data", make_download(export_topics), "topic_model_results.csv", "text/csv", use_container_width=True)
        else:
            st.info("Your filters changed since the topic model was run. Run the topic model again to refresh the results.")

with tab5:
    st.subheader(f"Theory-guided lens · {lens_name}")
    st.write(lens["description"])
    if lens["keywords"]:
        st.markdown("**Current construct dictionary:** " + ", ".join(f"`{k}`" for k in lens["keywords"]))
    else:
        st.info("Add keywords in the sidebar to activate the custom construct lens.")

    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered, x="subject_relevance", nbins=25, title=f"{lens_name} relevance distribution", labels={"subject_relevance":"Relevance score (0–100)"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rel_band = pd.cut(filtered["subject_relevance"], bins=[-0.1,10,25,50,75,100], labels=["Very low","Low","Moderate","High","Very high"])
        tmp = pd.DataFrame({"band": rel_band, "sentiment": filtered["sentiment_compound"]}).groupby("band", observed=True).agg(texts=("sentiment","size"), avg_sentiment=("sentiment","mean")).reset_index()
        fig = px.bar(tmp, x="band", y="avg_sentiment", text="texts", title="Sentiment across relevance bands", labels={"band":"Relevance band","avg_sentiment":"Average sentiment"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Highest-relevance texts")
    high = filtered.nlargest(25, "subject_relevance")[[text_col,"subject_relevance","subject_keyword_hits","sentiment_label","sentiment_compound"]]
    st.dataframe(high, use_container_width=True, hide_index=True)

    if group_col != "None":
        g = filtered.groupby(group_col).agg(texts=(text_col,"size"), relevance=("subject_relevance","mean"), sentiment=("sentiment_compound","mean")).reset_index()
        g = g[g.texts >= 2].nlargest(25, "relevance")
        fig = px.bar(g.sort_values("relevance"), x="relevance", y=group_col, orientation="h", hover_data=["texts","sentiment"], title=f"{lens_name} relevance by {group_col}")
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("Filtered data explorer")
    query = st.text_input("Search within text", placeholder="Type a word or phrase…")
    explorer = filtered.copy()
    if query:
        explorer = explorer[explorer[text_col].str.contains(query, case=False, na=False, regex=False)]
    default_cols = [text_col, "sentiment_label", "sentiment_compound", "subject_relevance", "word_count"]
    if group_col != "None" and group_col not in default_cols:
        default_cols.insert(0, group_col)
    selected_cols = st.multiselect("Columns to display", explorer.columns.tolist(), default=[c for c in default_cols if c in explorer.columns])
    st.dataframe(explorer[selected_cols] if selected_cols else explorer, use_container_width=True, hide_index=True, height=560)
    st.download_button("⬇️ Download all analysed rows", make_download(filtered), "analysed_social_text.csv", "text/csv", use_container_width=True)

st.divider()
st.caption("Research note: automated sentiment and topic models are exploratory measurement tools. Validate construct dictionaries, topic labels and classification quality on a human-coded sample before making confirmatory claims.")
