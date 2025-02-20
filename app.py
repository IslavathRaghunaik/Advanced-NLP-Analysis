import streamlit as st
import torch
from transformers import pipeline
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Initialize transformers pipeline
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization")

# Emoji Dictionary
emoji_dict = {
    "happy": "😊", "sad": "😢", "angry": "😡", "love": "❤️",
    "laugh": "😂", "surprise": "😮", "cool": "😎", "sleepy": "😴",
    "food": "🍔", "coffee": "☕", "money": "💰", "fire": "🔥",
    "clap": "👏", "star": "⭐", "heart": "💖", "idea": "💡"
}

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    processed_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token.isalnum() and token not in stop_words
    ]
    
    return " ".join(processed_tokens)

def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']

def generate_summary(text):
    if len(text.split()) > 100:  # Only summarize longer texts
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    return "Text is too short to summarize."

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def analyze_text_structure(text):
    doc = nlp(text)
    sentences = len(list(doc.sents))
    words = len([token for token in doc if not token.is_punct])
    return {
        "sentences": sentences,
        "words": words,
        "average_words_per_sentence": round(words/sentences if sentences > 0 else 0, 2)
    }

def convert_text_to_emojis(text):
    words = text.split()
    converted_text = " ".join([emoji_dict.get(word.lower(), word) for word in words])
    return converted_text

# Streamlit UI
st.set_page_config(page_title="Advanced NLP Analysis Tool", layout="wide")

st.title("🤖 Advanced NLP Analysis Tool")
st.write("""
This tool provides comprehensive natural language processing capabilities including:
- Text Preprocessing
- Sentiment Analysis
- Text Summarization
- Named Entity Recognition
- Text Structure Analysis
- Text-to-Emojis Conversion 🎭
""")

# Input text area
text_input = st.text_area("Enter your text here:", height=200)

if text_input:
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Preprocessing", "Sentiment", "Summary", "Entities", "Structure", "Emojis"
    ])
    
    with tab1:
        st.subheader("Text Preprocessing.")
        processed_text = preprocess_text(text_input)
        st.write("Original text length:", len(text_input.split()))
        st.write("Processed text length:", len(processed_text.split()))
        st.write("Processed text:")
        st.write(processed_text)
    
    with tab2:
        st.subheader("Sentiment Analysis")
        sentiment, score = analyze_sentiment(text_input)
        st.write(f"Sentiment: {sentiment}")
        st.progress(score)
        st.write(f"Confidence: {score:.2%}")
    
    with tab3:
        st.subheader("Text Summary")
        summary = generate_summary(text_input)
        st.write(summary)
    
    with tab4:
        st.subheader("Named Entity Recognition")
        entities = extract_entities(text_input)
        if entities:
            df = pd.DataFrame(entities, columns=["Entity", "Type"])
            st.dataframe(df)
        else:
            st.write("No entities found.")
    
    with tab5:
        st.subheader("Text Structure Analysis")
        structure = analyze_text_structure(text_input)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sentences", structure["sentences"])
        with col2:
            st.metric("Words", structure["words"])
        with col3:
            st.metric("Avg Words/Sentence", structure["average_words_per_sentence"])
    
    with tab6:
        st.subheader("Text-to-Emojis Conversion 🎭")
        emoji_text = convert_text_to_emojis(text_input)
        st.write("Converted Text with Emojis:")
        st.write(emoji_text)
