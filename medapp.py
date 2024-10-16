import streamlit as st
import streamlit_analytics
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import json
from typing import List, Dict
import os
from PIL import Image
import faiss
from openai import OpenAI
import pickle
import cv2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Wrap the entire app in streamlit_analytics.track()
with streamlit_analytics.track():
    # Set up OpenAI client
    st.sidebar.title("OpenAI API Key")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
    client = OpenAI(api_key=key)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SentenceTransformer model
    @st.cache_resource
    def load_sentence_transformer():
        return SentenceTransformer('all-MiniLM-L6-v2').to(device)

    model = load_sentence_transformer()

    # Load and preprocess video data
    @st.cache_data
    def load_and_preprocess_data(topic: str):
        file_path = os.path.join('data', f'{topic.lower()}_videos.json')
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        
        # Check if preprocessed embeddings exist
        embeddings_file = f"embeddings_{topic.lower()}.pkl"
        print(f"creating {embeddings_file}")
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
        else:
            embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
            embeddings = embeddings.cpu().numpy()
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)
        
        # Create FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        return data, index, embeddings

    # Retrieve relevant passages using FAISS
    def retrieve_passages(query: str, index, embeddings: np.ndarray, video_data: List[Dict], top_k: int = 5) -> List[Dict]:
        query_embedding = model.encode([query], convert_to_tensor=True, show_progress_bar=False).cpu().numpy()
        D, I = index.search(query_embedding, top_k)
        
        retrieved_passages = []
        for idx in I[0]:
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
                    {"role": "system", "content": "You are a helpful assistant that answers medical questions based on the provided context. Always ground your answers in the given context and be concise."},
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

    # Extract frame from video at specific timestamp
    @st.cache_data
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
        st.title("Step 1 Buddy")

        # Add a new tab for disclosures
        tab1, tab2 = st.tabs(["Main", "Disclosures"])

        with tab1:
            topics = ["immunology", "gastroenterology", "cell biology"]
            selected_topic = st.selectbox("Select a topic", topics, key="topic_selectbox")

            video_data, index, embeddings = load_and_preprocess_data(selected_topic)
            user_query = st.text_input("Enter your question:", key="user_query_input")

            if user_query:
                with st.spinner("Searching for relevant information..."):
                    relevant_passages = retrieve_passages(user_query, index, embeddings, video_data)

                context = " ".join([p["text"] for p in relevant_passages])
                
                with st.spinner("Generating answer..."):
                    answer = generate_answer(user_query, context)

                st.subheader("Generated Answer:")
                st.write(answer)

                with st.expander("View Relevant Passages"):
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

        with tab2:
            st.header("Disclosures")
            with open("disclosures.txt", "r") as f:
                disclosures_content = f.read()
            st.markdown(disclosures_content)

    if __name__ == "__main__":
        main()
