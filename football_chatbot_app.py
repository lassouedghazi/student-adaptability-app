import pandas as pd
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Load NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the football players dataset from text file
with open('football_players.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

# Tokenize the text into sentences
sentences = sent_tokenize(text_data)

# Preprocess the data
lemmatizer = WordNetLemmatizer()

def preprocess(sentence):
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word.isalpha()]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Function to find synonyms using WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

# Function to expand the query with synonyms
def expand_query(query):
    words = query.split()
    expanded_words = []
    for word in words:
        expanded_words.append(word)
        synonyms = get_synonyms(word)
        expanded_words.extend(synonyms)
    return " ".join(set(expanded_words))

# Function to find the most relevant sentences using TF-IDF and cosine similarity
@st.cache_data
def get_most_relevant_sentences(query, num_responses=3):
    query_processed = preprocess(query)
    expanded_query = expand_query(query_processed)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    sentences_combined = sentences + [expanded_query]

    # Transform the sentences into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(sentences_combined)

    # Compute cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Get indices of the most relevant sentences
    most_relevant_indices = cosine_similarities[0].argsort()[-num_responses:][::-1]

    return [sentences[i] for i in most_relevant_indices]

# Set Streamlit page configuration
st.set_page_config(page_title="⚽ Football Player Chatbot by Ghazi Lassoued", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;
        font-family: 'Arial', sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background-color: white;
        color: red;
        border-radius: 10px;
        border-bottom: 5px solid red;
        margin-bottom: 20px;
    }
    .header .title {
        font-weight: bold;
        font-size: 36px;
    }
    .header .subtitle {
        font-size: 24px;
        color: #333;
        margin-top: 5px;
    }
    .header .author {
        font-size: 18px;
        color: #333;
        position: absolute;
        top: 20px;
        right: 20px;
    }
    .result {
        color: #28A745;
        font-size: 18px;
        margin: 10px 0;
        padding: 10px;
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 1px 5px rgba(0, 0, 0, 0.1);
    }
    .feedback {
        margin-top: 10px;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <div class="header">
        <div class="author">by Ghazi Lassoued 🇹🇳</div>
        <div class="title">⚽ Football Player Chatbot</div>
        <div class="subtitle">Ask about Tunisian football players!</div>
    </div>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    This chatbot is designed to answer questions specifically about football players from Tunisia who have Tunisian nationality.
    You can ask about their careers, achievements, and contributions to the national team.
    """, unsafe_allow_html=True
)

st.markdown("**Ask me about your favorite Tunisian football players!**")
st.markdown("**Example questions:** 'Tell me about Wahbi Khazri's career', 'What are some achievements of Youssef Msakni?', 'Describe the history of the Tunisian national team.'")

# User input for the chatbot
user_input = st.text_input("Type your question about Tunisian football players:")

if st.button('🔍 Get Answer'):
    if user_input:
        with st.spinner('Fetching answer...'):
            responses = get_most_relevant_sentences(user_input)

        st.subheader('📊 Responses:')
        for i, response in enumerate(responses, start=1):
            st.markdown(f'<p class="result">{i}. {response}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="result">Please enter a question!</p>', unsafe_allow_html=True)

    # Feedback mechanism
    feedback = st.radio("Was this answer helpful?", ('Select an option', 'Yes', 'No'))
    if feedback != 'Select an option':
        st.success(f'Thank you for your feedback: {feedback}!')
