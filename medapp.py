import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import json
from typing import List, Dict
import os
import cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import time
from requests.exceptions import RequestException
from openai import OpenAI

# Set up OpenAI client
key = st.text_input("Enter your key:")
client = OpenAI(api_key=key)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to load BERT model with retry logic
def load_bert_model(model_name="distilbert-base-uncased", max_retries=3):
    for attempt in range(max_retries):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            return tokenizer, model
        except RequestException as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Failed to download {model_name} after {max_retries} attempts.")
                print("Falling back to a local model or a smaller model...")
                raise e

# Load BERT model and tokenizer for embeddings
try:
    bert_tokenizer, bert_model = load_bert_model()
except Exception as e:
    st.error(f"Failed to load BERT model: {str(e)}")
    st.error("Please check your internet connection and try again later.")
    st.stop()

# Function to generate embeddings using BERT
def generate_embeddings(texts: List[str]) -> np.ndarray:
    embeddings = []
    for text in texts:
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    return np.array(embeddings)

# Load video transcripts and metadata
def load_video_data(topic: str) -> List[Dict]:
    file_path = os.path.join('data', f'{topic.lower()}_videos.json')
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Retrieve relevant passages using cosine similarity
def retrieve_passages(query: str, embeddings: np.ndarray, video_data: List[Dict], top_k: int = 5) -> List[Dict]:
    query_embedding = generate_embeddings([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    retrieved_passages = []
    for idx in top_indices:
        passage = video_data[idx]
        retrieved_passages.append({
            'text': passage['text'],
            'video_title': passage['video_title'],
            'timestamp': passage['timestamp'],
            'video_path': passage['video_path']
        })
    
    return retrieved_passages

# Generate answer using OpenAI's GPT-4
def generate_answer(query: str, context: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers medical questions based on the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I couldn't generate an answer at this time."

# Preprocess video data and create embeddings
def preprocess_data(video_data: List[Dict]) -> (np.ndarray, List[Dict]):
    texts = [item['text'] for item in video_data]
    embeddings = generate_embeddings(texts)
    return embeddings, video_data

# Extract frame from video at specific timestamp
def extract_frame(video_path: str, timestamp: float) -> Image.Image:
    if not os.path.exists(video_path):
        st.error(f"Video file not found: {video_path}")
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Failed to open video file: {video_path}")
            return None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        st.info(f"Video properties: FPS={fps}, Total Frames={total_frames}, Duration={duration:.2f}s")

        if timestamp > duration:
            st.warning(f"Timestamp {timestamp}s exceeds video duration {duration:.2f}s. Using last frame.")
            timestamp = duration

        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            st.warning(f"Could not extract frame at timestamp {timestamp}s (frame {frame_number}) from {video_path}")
            return None
    except cv2.error as e:
        st.error(f"OpenCV error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

# Main Streamlit app
def main():
    st.title("Step 1 buddy")

    topics = ["immunology"]
    selected_topic = st.selectbox("Select a topic", topics)

    video_data = load_video_data(selected_topic)

    embeddings, processed_video_data = preprocess_data(video_data)

    user_query = st.text_input("Enter your question:")

    if user_query:
        relevant_passages = retrieve_passages(user_query, embeddings, processed_video_data)

        context = " ".join([p["text"] for p in relevant_passages])
        
        # Generate answer using GPT-4
        answer = generate_answer(user_query, context)

        st.subheader("Generated Answer:")
        st.write(answer)

        st.subheader("Relevant Passages:")
        for passage in relevant_passages:
            st.write(f"Video: {passage['video_title']}")
            st.write(f"Timestamp: {passage['timestamp']}")
            st.write(f"Relevant text: {passage['text']}")
            
            frame = extract_frame(passage['video_path'], passage['timestamp'])
            if frame:
                st.image(frame, caption=f"Frame at {passage['timestamp']} seconds")
            else:
                st.write("Failed to extract frame from video.")
            
            st.write("---")

if __name__ == "__main__":
    main()