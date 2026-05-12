import streamlit as st
import joblib
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline

st.set_page_config(page_title="Fake News Classifier", page_icon="📰", layout="wide")

BASE_DIR = Path(__file__).parent

LABEL_MAPPING = {
    0: "Fake News",
    1: "Real News"
}


@st.cache_resource
def load_sklearn_models():
    return {
        "Logistic Regression": {
            "model": joblib.load(BASE_DIR / "log_model.pkl"),
            "vectorizer": joblib.load(BASE_DIR / "log_model_vectorizer.pkl"),
        },
        "Random Forest": {
            "model": joblib.load(BASE_DIR / "forest_model.pkl"),
            "vectorizer": joblib.load(BASE_DIR / "forest_model_vectorizer.pkl"),
        },
        "Naive Bayes": {
            "model": joblib.load(BASE_DIR / "bayes_model.pkl"),
            "vectorizer": joblib.load(BASE_DIR / "bayes_model_vectorizer.pkl"),
        },
        "Passive Aggressive": {
            "model": joblib.load(BASE_DIR / "passive_model.pkl"),
            "vectorizer": joblib.load(BASE_DIR / "passive_model_vectorizer.pkl"),
        },
    }


@st.cache_resource
def load_bert_pipeline():
    bert_model = BertForSequenceClassification.from_pretrained(
        "davidli33/bertclassifierbest"
    )
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return TextClassificationPipeline(
        model=bert_model,
        tokenizer=bert_tokenizer,
        truncation=True,
        max_length=512,
    )


def show_result(prediction, confidence):
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Prediction", value=prediction)

    with col2:
        st.metric(label="Confidence", value=f"{confidence:.2f}%")


st.sidebar.title("📰 Fake News Classifier")
st.sidebar.write(
    """
Enter a news article below, and the model will detect if it is **Real News**
or **Fake News** along with the confidence score.
"""
)

model_option = st.sidebar.radio(
    "Choose the model:",
    (
        "Logistic Regression",
        "Random Forest",
        "Naive Bayes",
        "Passive Aggressive",
        "BERT",
    ),
)

st.header("Enter Article Text")

user_input = st.text_area(
    "Article text",
    height=300,
    label_visibility="collapsed",
    placeholder="Paste a news article here...",
)

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter text to classify.")
        st.stop()

    with st.spinner("Classifying..."):
        if model_option == "BERT":
            bert_pipe = load_bert_pipeline()
            result = bert_pipe(user_input)[0]

            label = result["label"]
            confidence = result["score"] * 100

            if label == "LABEL_0":
                prediction = "Real News"
            else:
                prediction = "Fake News"

            show_result(prediction, confidence)

        else:
            sklearn_models = load_sklearn_models()

            model = sklearn_models[model_option]["model"]
            vectorizer = sklearn_models[model_option]["vectorizer"]

            transformed_input = vectorizer.transform([user_input])
            prediction_raw = model.predict(transformed_input)[0]
            prediction = LABEL_MAPPING[prediction_raw]

            if model_option == "Passive Aggressive":
                decision_score = model.decision_function(transformed_input)[0]
                confidence = 1 / (1 + np.exp(-decision_score))
                confidence = confidence * 100
            else:
                probabilities = model.predict_proba(transformed_input)[0]
                confidence = np.max(probabilities) * 100

            show_result(prediction, confidence)

st.markdown("---")
st.write("Made with ❤️ using Streamlit")