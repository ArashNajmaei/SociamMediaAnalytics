# Social Text Intelligence Lab Pro

A GitHub-ready Streamlit workbench for research and analyst use with scraped/exported YouTube, Reddit, app-review, forum, social-media and customer-review text.

## Pro features

- Local CSV upload or public CSV URL.
- Automatic text/date column suggestions.
- Theory lenses: **Brand Personalisation**, **Value Co-Creation**, **Brand Salience**, or a custom construct.
- VADER sentiment plus transparent multi-emotion lexicon analysis.
- Unigrams, bigrams, trigrams, word clouds and positive/negative lexical contrast.
- **NMF, LDA, and optional BERTopic** topic discovery.
- Automatic descriptive topic names from high-weight terms.
- **NPMI topic coherence** diagnostics.
- **Repeated-seed topic stability** for NMF/LDA using optimal topic matching and top-word Jaccard similarity.
- Topic prevalence, representative documents, sentiment/relevance positioning and interactive document maps.
- **Temporal topic evolution** when a date column exists.
- **Cross-platform / cross-brand topic composition** when a grouping column exists.
- **Term co-occurrence networks** with interactive Plotly visualisation.
- **KWIC (keyword-in-context)** concordance explorer.
- Searchable enriched data table.
- Downloadable analysed CSV, topic outputs, diagnostics, summaries, metadata, and a combined **research export ZIP**.

## Recommended CSV structure

Only a text field is compulsory. For richer analysis:

```csv
platform,brand,date,text
YouTube,Nike,2026-01-01,"I love how the recommendations fit my style"
Reddit,Nike,2026-01-02,"The community should have more input into product ideas"
App Review,Nike,2026-01-03,"The logo is instantly recognisable"
```

## Standard deployment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Community Cloud, push `app.py`, `requirements.txt`, `.streamlit/config.toml` and any optional sample data to GitHub, then select `app.py` as the entry point.

## BERTopic deployment

BERTopic is optional because transformer embeddings + UMAP + HDBSCAN require substantially more RAM, download time and startup time.

For a host with sufficient resources:

```bash
pip install -r requirements-bertopic.txt
streamlit run app.py
```

The app automatically exposes BERTopic in the same Topic Lab. If BERTopic dependencies are absent, NMF/LDA remain fully functional.

For Streamlit Community Cloud, if you want BERTopic to install automatically, rename `requirements-bertopic.txt` to `requirements.txt`. Be aware that free/community resource limits may be restrictive for larger corpora.

## Research architecture

### 1. Unsupervised discovery is separate from theory-guided measurement

The selected subject-area dictionary does **not** define NMF/LDA/BERTopic topics. Topics are discovered from the corpus first. The construct relevance score is calculated separately so researchers can assess how emergent discourse relates to Brand Personalisation, Value Co-Creation, Brand Salience or a custom construct.

### 2. Construct relevance

Each text receives a 0–100 exploratory relevance score combining:

- TF-IDF similarity to the construct dictionary; and
- direct dictionary keyword/phrase hits.

For journal research, validate the dictionary on a manually coded sample and report its precision/recall or agreement.

### 3. Topic coherence

The app calculates **NPMI coherence** from pairwise document co-occurrence of each topic's leading terms. Use coherence comparatively across alternative specifications; do not treat any one value as a universal pass/fail threshold.

### 4. Topic stability

NMF/LDA are fitted repeatedly using different random seeds. Topics are optimally matched to the baseline run and scored by Jaccard similarity of their leading terms. This provides a transparent robustness diagnostic for whether broadly similar themes recur.

### 5. Emotion analysis

Emotion scores use a compact transparent in-app lexicon for Joy, Trust, Anger, Fear, Sadness, Surprise, Disgust and Anticipation. This avoids external model downloads and makes the measurement auditable. It is intentionally exploratory. Researchers needing validated emotion measurement should substitute or validate a field-appropriate dictionary/model.

## Recommended research workflow

1. Preserve a raw immutable data export.
2. Document scraping/API inclusion and exclusion rules.
3. Remove duplicates, bots/spam and non-relevant records where defensible.
4. Run descriptive text and sentiment/emotion exploration.
5. Compare plausible topic counts and NMF/LDA/BERTopic solutions.
6. Review coherence, stability, prevalence and representative/deviant texts.
7. Name topics using both high-weight terms and representative documents, not top terms alone.
8. Validate automated constructs/sentiment/emotions on a stratified human-coded sample.
9. Compare topics across platform/brand and over time only where sample sizes are adequate.
10. Export evidence tables and retain analysis settings for reproducibility.

## Privacy and platform governance

Do not publicly deploy identifiable, restricted or sensitive scraped data unless collection, storage, processing and display comply with applicable platform terms, ethics approval, privacy requirements and institutional research governance. Where possible, analyse de-identified text and avoid redistributing raw usernames or identifiers.
