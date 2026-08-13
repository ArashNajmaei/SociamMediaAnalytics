# Streamlit Cloud deployment

1. Upload all files and the `.streamlit` folder from this package to the **root** of your GitHub repository.
2. Confirm the repository root contains `app.py` and `requirements.txt` side by side.
3. In Streamlit Community Cloud, choose the repository and set **Main file path** to `app.py`.
4. Deploy or reboot the app.

## Standard deployment

Streamlit Cloud automatically installs packages from `requirements.txt`. This is the recommended first deployment and includes NMF and LDA topic modelling, sentiment analysis, text mining, visualisation, networks and exports.

## BERTopic deployment

BERTopic is intentionally separated because it is much heavier. To deploy BERTopic, replace the contents of `requirements.txt` with the combined dependencies in `requirements-bertopic.txt`, or install those optional dependencies in a hosting environment with sufficient memory/CPU.

## Test data

- `sample_social_data.csv` — small quick test file.
- `mock_social_media_text_data.csv` — 600-row mock dataset covering Brand Personalisation, Value Co-Creation, Brand Salience, service, price, privacy and usability.

For the 600-row mock file, use `text` as the text column, `date` as the date field, and `platform` or `brand` as a grouping field.
