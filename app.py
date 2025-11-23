import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import time
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Mobile Legends Sentiment Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    h1 { color: #667eea; font-weight: 800; }
    h2 { color: #334155; font-weight: 700; }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white;
        border-radius: 12px;
        font-weight: 600;
    }
    .metric-card { background: #f8fafc; padding: 20px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# Initialize preprocessing tools
@st.cache_resource
def load_preprocessing_tools():
    try:
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stop_words = set(stopwords.words('indonesian'))
        return stemmer, stop_words
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# Load models
@st.cache_resource
def load_models():
    try:
        with open('models/naive_bayes_model.pkl', 'rb') as f:
            nb_model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        
        lstm_model = keras.models.load_model('models/lstm_model.h5', compile=False)
        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        with open('models/label_map.pkl', 'rb') as f:
            label_map = pickle.load(f)
        
        return nb_model, tfidf, lstm_model, tokenizer, label_map
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load data
@st.cache_data
def load_data():
    try:
        comparison = pd.read_csv('data/algorithm_comparison.csv')
        cleaned_data = pd.read_csv('data/cleaned_data.csv')
        return comparison, cleaned_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Preprocessing functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text, stemmer, stop_words):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Prediction functions
def predict_naive_bayes(text, nb_model, tfidf):
    text_tfidf = tfidf.transform([text])
    prediction = nb_model.predict(text_tfidf)[0]
    probabilities = nb_model.predict_proba(text_tfidf)[0]
    return prediction, probabilities

def predict_lstm(text, lstm_model, tokenizer, max_len=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    probabilities = lstm_model.predict(padded, verbose=0)[0]
    prediction = np.argmax(probabilities)
    return prediction, probabilities

# Load everything
try:
    stemmer, stop_words = load_preprocessing_tools()
    nb_model, tfidf, lstm_model, tokenizer, label_map = load_models()
    comparison_df, cleaned_data = load_data()
    reverse_label_map = {v: k for k, v in label_map.items()}
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    st.stop()

# Header
st.title("🎮 Mobile Legends Sentiment Analysis")
st.markdown("### 🤖 Perbandingan Naive Bayes vs LSTM")

# Sidebar
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Dashboard", "🔮 Real-Time Prediction", "📊 Algorithm Comparison", "☁️ Word Cloud"]
)

# ========== PAGE 1: DASHBOARD ==========
if page == "🏠 Dashboard":
    st.header("📈 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", f"{len(cleaned_data):,}")
    with col2:
        try:
            nb_acc = float(comparison_df[comparison_df['Metric'] == 'Accuracy']['Naive Bayes'].values[0])
            st.metric("NB Accuracy", f"{nb_acc:.1%}")
        except:
            st.metric("NB Accuracy", "N/A")
    with col3:
        try:
            lstm_acc = float(comparison_df[comparison_df['Metric'] == 'Accuracy']['LSTM'].values[0])
            st.metric("LSTM Accuracy", f"{lstm_acc:.1%}")
        except:
            st.metric("LSTM Accuracy", "N/A")
    with col4:
        st.metric("Status", "✅ Running")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sentiment Distribution")
        try:
            sentiment_counts = cleaned_data['sentiment'].value_counts()
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                        color_discrete_map={'positive': '#10b981', 'neutral': '#f59e0b', 'negative': '#ef4444'},
                        hole=0.4)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Chart unavailable: {str(e)[:50]}")
    
    with col2:
        st.subheader("⭐ Rating Distribution")
        try:
            rating_counts = cleaned_data['score'].value_counts().sort_index()
            fig = px.bar(x=rating_counts.index, y=rating_counts.values,
                        labels={'x': 'Rating', 'y': 'Count'})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Chart unavailable: {str(e)[:50]}")
    
    st.markdown("---")
    
    st.subheader("📋 Algorithm Metrics")
    try:
        # Convert to float to avoid string comparison errors
        comparison_display = comparison_df.copy()
        for col in ['Naive Bayes', 'LSTM']:
            comparison_display[col] = comparison_display[col].astype(float)
        st.dataframe(comparison_display.style.format({
            'Naive Bayes': '{:.4f}',
            'LSTM': '{:.4f}'
        }), use_container_width=True)
    except:
        st.dataframe(comparison_df)

# ========== PAGE 2: REAL-TIME PREDICTION ==========
elif page == "🔮 Real-Time Prediction":
    st.header("🔮 Real-Time Sentiment Prediction")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_area(
            "✍️ Tulis Review:",
            placeholder="Contoh: Game sangat seru dan grafis bagus...",
            height=120
        )
    
    with col2:
        st.markdown("### Sample Review")
        if st.button("😊 Positive"):
            user_input = "Game sangat seru dan menyenangkan! Grafisnya bagus!"
        if st.button("😐 Neutral"):
            user_input = "Game lumayan, ada plus minusnya"
        if st.button("😞 Negative"):
            user_input = "Game jelek, sering lag dan tidak adil"
    
    if st.button("🚀 PREDICT", use_container_width=True):
        if user_input.strip():
            with st.spinner("Processing..."):
                cleaned_input = preprocess_text(user_input, stemmer, stop_words)
                
                # Naive Bayes
                start = time.time()
                pred_nb, prob_nb = predict_naive_bayes(cleaned_input, nb_model, tfidf)
                time_nb = (time.time() - start) * 1000
                sentiment_nb = reverse_label_map[pred_nb]
                
                # LSTM
                start = time.time()
                pred_lstm, prob_lstm = predict_lstm(cleaned_input, lstm_model, tokenizer)
                time_lstm = (time.time() - start) * 1000
                sentiment_lstm = reverse_label_map[pred_lstm]
                
                st.success("✅ Prediction Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🤖 Naive Bayes")
                    st.success(f"**{sentiment_nb.upper()}**")
                    st.metric("Confidence", f"{prob_nb[pred_nb]:.1%}")
                    st.metric("Speed", f"{time_nb:.2f} ms")
                
                with col2:
                    st.markdown("### 🧠 LSTM")
                    st.success(f"**{sentiment_lstm.upper()}**")
                    st.metric("Confidence", f"{prob_lstm[pred_lstm]:.1%}")
                    st.metric("Speed", f"{time_lstm:.2f} ms")
                
                st.markdown("---")
                
                if sentiment_nb == sentiment_lstm:
                    st.info(f"✅ **BOTH AGREE**: {sentiment_nb.upper()}")
                else:
                    st.warning(f"⚠️ **DISAGREE**: NB={sentiment_nb}, LSTM={sentiment_lstm}")
        else:
            st.error("Please enter review text!")

# ========== PAGE 3: ALGORITHM COMPARISON ==========
elif page == "📊 Algorithm Comparison":
    st.header("⚔️ Algorithm Comparison")
    
    st.subheader("📋 Metrics")
    try:
        comparison_display = comparison_df.copy()
        for col in ['Naive Bayes', 'LSTM']:
            comparison_display[col] = comparison_display[col].astype(float)
        st.dataframe(comparison_display, use_container_width=True)
    except:
        st.dataframe(comparison_df)
    
    st.markdown("---")
    
    try:
        metrics = comparison_df['Metric'].tolist()
        nb_scores = [float(x) for x in comparison_df['Naive Bayes'].tolist()]
        lstm_scores = [float(x) for x in comparison_df['LSTM'].tolist()]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Naive Bayes', x=metrics, y=nb_scores, marker_color='#6366f1'))
        fig.add_trace(go.Bar(name='LSTM', x=metrics, y=lstm_scores, marker_color='#8b5cf6'))
        fig.update_layout(barmode='group', height=400, yaxis=dict(range=[0, 1.1]))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Chart error: {str(e)[:50]}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Strengths NB")
        st.markdown("- ⚡ Lightning fast\n- 📚 Simple & interpretable\n- 💾 Lightweight")
    
    with col2:
        st.markdown("### ✅ Strengths LSTM")
        st.markdown("- 🎯 Higher accuracy\n- 🧠 Context-aware\n- 📝 Handles complexity")

# ========== PAGE 4: WORD CLOUD ==========
elif page == "☁️ Word Cloud":
    st.header("☁️ Word Cloud Analysis")
    
    sentiment = st.selectbox("Choose Sentiment:", ['positive', 'neutral', 'negative'])
    
    try:
        sentiment_text = ' '.join(cleaned_data[cleaned_data['sentiment'] == sentiment]['cleaned_text'])
        
        if sentiment_text.strip():
            wordcloud = WordCloud(width=1200, height=600, background_color='white').generate(sentiment_text)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning(f"No {sentiment} reviews found")
    except Exception as e:
        st.error(f"Error: {str(e)[:100]}")

# Footer
st.markdown("---")
st.markdown("<center>🎮 Mobile Legends Sentiment Analysis | Naive Bayes vs LSTM</center>", unsafe_allow_html=True)