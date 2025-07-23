import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import requests
import io

# --- Configuration ---
# IMPORTANT: For deployment, use Streamlit Secrets instead of hardcoding your API key.
# API_KEY = st.secrets["GEMINI_API_KEY"]
# For local testing, replace with your actual API key, but be cautious:
API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"  # Replace with your actual API key or use st.secrets

genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')
embedding_model = 'models/embedding-001'  # Gemini's embedding model for text embeddings

# --- Data Loading and Embedding Generation ---
# Raw GitHub URL for the CSV file
GITHUB_CSV_URL = "https://raw.githubusercontent.com/NirmalGaud1/banking_chatbot_new/main/banking_chatbot_dataset.csv"

@st.cache_data
def load_and_process_data(url):
    """
    Loads the CSV dataset from a given URL, generates embeddings for questions,
    and returns the cleaned DataFrame and question vectors.
    """
    try:
        # Fetch the CSV content from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        df = pd.read_csv(io.StringIO(response.text))  # Read string content as a file

        # Ensure 'Question' and 'Answer' columns exist
        if 'Question' not in df.columns or 'Answer' not in df.columns:
            st.error("Error: The CSV file must contain 'Question' and 'Answer' columns.")
            return None, None

        st.info("Generating embeddings for banking questions... This might take a moment.")
        question_embeddings = []
        for index, row in df.iterrows():
            question = row['Question']
            try:
                # Generate embedding for each question
                embedding = genai.embed_content(model=embedding_model, content=question)['embedding']
                question_embeddings.append(embedding)
            except Exception as e:
                st.warning(f"Warning: Could not generate embedding for question: '{question}'. Error: {e}")
                question_embeddings.append(None)  # Append None to maintain index alignment

        # Filter out questions that failed embedding generation
        df['embedding'] = question_embeddings
        df_cleaned = df.dropna(subset=['embedding']).reset_index(drop=True)

        if df_cleaned.empty:
            st.error("Error: No valid questions found in the dataset after embedding generation. "
                     "Please check your CSV content and ensure your API key is valid.")
            return None, None

        st.success(f"Successfully loaded {len(df_cleaned)} questions and generated embeddings.")
        # Convert list of embeddings to a NumPy array for efficient similarity calculation
        return df_cleaned, np.array(df_cleaned['embedding'].tolist())

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch CSV from GitHub. Please check the URL and your internet connection. Error: {e}")
        return None, None
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty. Please ensure it contains data.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading or processing data: {e}")
        return None, None

# Load and process the data when the app starts
df_data, question_vectors = load_and_process_data(GITHUB_CSV_URL)

# --- RAG Functionality ---
def retrieve_context(query, df, question_vectors, top_k=3):
    """
    Retrieves the most similar contexts (questions and answers) from the dataset
    based on the user's query.
    """
    if df is None or question_vectors is None:
        return []

    try:
        # Generate embedding for the user's query
        query_embedding = genai.embed_content(model=embedding_model, content=query)['embedding']
        query_vector = np.array(query_embedding).reshape(1, -1)

        # Calculate cosine similarity between the query and all question embeddings
        similarities = cosine_similarity(query_vector, question_vectors)[0]

        # Get the indices of the top_k most similar questions
        top_k_indices = similarities.argsort()[-top_k:][::-1]

        # Retrieve relevant contexts (questions and answers)
        contexts = []
        for i in top_k_indices:
            row = df.iloc[i]
            contexts.append({
                "question": row['Question'],
                "answer": row['Answer'],
                "similarity": similarities[i]  # Include similarity score for transparency
            })
        return contexts
    except Exception as e:
        st.error(f"Error during context retrieval: {e}")
        return []

def generate_response(user_query, contexts):
    """
    Generates a response using the Gemini model, incorporating retrieved contexts.
    """
    if not contexts:
        return "I'm sorry, I couldn't find relevant information for your query in my knowledge base. Please try rephrasing your question or ask a different one."

    # Construct the prompt for the LLM using the retrieved contexts
    context_text = "\n".join([f"Question: {c['question']}\nAnswer: {c['answer']}" for c in contexts])

    prompt = f"""
    You are a helpful and informative banking chatbot. Use the following provided context to answer the user's question.
    If the context does not contain the answer, politely state that you don't have enough information on that specific topic
    and suggest they contact banking support for more detailed assistance.
    Do not make up answers.

    Context from banking knowledge base:
    {context_text}

    User Question: {user_query}

    Answer:
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response from Gemini: {e}")
        return "I'm sorry, I encountered an error while trying to generate a response. Please try again later."

# --- Streamlit UI ---
st.set_page_config(page_title="Banking Chatbot (RAG)", layout="centered")

st.title("🏦 Banking Chatbot (RAG)")
st.markdown("Ask me anything about banking based on the provided dataset!")

# Check if data loading was successful
if df_data is None or question_vectors is None:
    st.warning("Chatbot is not ready. Please ensure the GitHub CSV URL is correct and accessible, and that your API key is valid.")
else:
    user_input = st.text_input("Your question:", key="user_input", placeholder="e.g., How do I open a new bank account?")

    if st.button("Ask", use_container_width=True):
        if user_input:
            with st.spinner("Thinking and retrieving information..."):
                # 1. Retrieve context
                retrieved_contexts = retrieve_context(user_input, df_data, question_vectors, top_k=3)  # Get top 3 for richer context

                # Display retrieved contexts for debugging/transparency (optional, can be removed in production)
                st.subheader("🔍 Retrieved Context (for transparency):")
                if retrieved_contexts:
                    for i, ctx in enumerate(retrieved_contexts):
                        st.markdown(f"**Match {i+1} (Similarity: `{ctx['similarity']:.2f}`):**")
                        st.markdown(f"**Q:** `{ctx['question']}`")
                        st.markdown(f"**A:** `{ctx['answer']}`")
                else:
                    st.info("No highly relevant context found in the dataset.")

                # 2. Generate response using the retrieved context
                response = generate_response(user_input, retrieved_contexts)
                st.subheader("💬 Chatbot's Answer:")
                st.write(response)
        else:
            st.warning("Please enter a question to get an answer.")

st.markdown("---")
st.markdown("Powered by Google Gemini and Streamlit. [Learn more about RAG](https://www.google.com/search?q=Retrieval+Augmented+Generation)")
