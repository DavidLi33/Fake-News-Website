import streamlit as st
import joblib
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TextClassificationPipeline

# Load models and vectorizers
vectorizer_lr = joblib.load('log_model_vectorizer.pkl')
vectorizer_rf = joblib.load('forest_model_vectorizer.pkl')
vectorizer_bayes = joblib.load('bayes_model_vectorizer.pkl')
vectorizer_passive = joblib.load('passive_model_vectorizer.pkl')

log_model = joblib.load('log_model.pkl')
forest_model = joblib.load('forest_model.pkl')
bayes_model = joblib.load('bayes_model.pkl')
passive_model = joblib.load('passive_model.pkl')

# Load BERT model and tokenizer
bert_model = BertForSequenceClassification.from_pretrained("davidli33/bertclassifiersample")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_pipe = TextClassificationPipeline(model=bert_model, tokenizer=bert_tokenizer)

# Set up Streamlit app
st.set_page_config(page_title="Fake News Classifier", page_icon="📰", layout='wide')

st.sidebar.title("📰 Fake News Classifier")
st.sidebar.write("""
Enter a news article below, and the model will detect if it's **Real News** or **Fake News** along with the confidence score.
""")

model_option = st.sidebar.radio("Choose the model:", ("Logistic Regression", "Random Forest", "Naive Bayes", "Passive Aggressive", "BERT"))

st.header("Enter Article Text")

user_input = st.text_area('', height=300)

if st.button('Classify'):
    if user_input.strip():
        if model_option == "Logistic Regression":
            model = log_model
            vectorizer = vectorizer_lr
        elif model_option == "Random Forest":
            model = forest_model
            vectorizer = vectorizer_rf
        elif model_option == "Naive Bayes":
            model = bayes_model
            vectorizer = vectorizer_bayes
        elif model_option == "Passive Aggressive":
            model = passive_model
            vectorizer = vectorizer_passive
        elif model_option == "BERT":
            model = bert_pipe
        if model_option == "BERT":
            max_length = 512
            user_input_trimmed = user_input[:max_length]
            result = bert_pipe(user_input_trimmed)
            label = result[0].get('label')
            confidence_val = result[0].get('score')
            confidence_val = round(confidence_val * 100, 2)  # Convert to percentage and round to 2 decimal places
            if label == "LABEL_0":
                label_val = "Real News"
                color_val = "green"
            else:
                label_val = "Fake News"
                color_val = "red"
            st.markdown('---')
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Prediction", value=label_val)
            with col2:
                st.metric(label="Confidence", value=f"{confidence_val}%")  # Add % sign
        elif model_option == "Passive Aggressive":
            user_input_transformed = vectorizer.transform([user_input])
            prediction = model.predict(user_input_transformed)[0]
            confidence = model.decision_function(user_input_transformed)[0]
            if confidence > 0:
                confidence = 1 / (1 + np.exp(-confidence))  
            else:
                confidence = np.exp(confidence) / (1 + np.exp(confidence))
            confidence = confidence * 100  
            label_mapping = {0: 'Fake News', 1: 'Real News'}
            predicted_class_label = label_mapping[prediction]
            st.markdown('---')
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Prediction", value=predicted_class_label)
            with col2:
                st.metric(label="Confidence", value=f"{confidence:.2f}%")
        else:
            user_input_transformed = vectorizer.transform([user_input])
            prediction = model.predict(user_input_transformed)[0]
            probabilities = model.predict_proba(user_input_transformed)[0]
            confidence = np.max(probabilities) * 100
            label_mapping = {0: 'Fake News', 1: 'Real News'}
            predicted_class_label = label_mapping[prediction]
            st.markdown('---')
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Prediction", value=predicted_class_label)
            with col2:
                st.metric(label="Confidence", value=f"{confidence:.2f}%")
    else:
        st.warning('Please enter text to classify.')

# Footer
st.markdown('---')
st.write('Made with ❤️ using Streamlit')