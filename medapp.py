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
import firebase_admin
from firebase_admin import credentials, firestore
import base64
import matplotlib.pyplot as plt
from knowledge_graph import generate_knowledge_graph
from mindmap import get_mindmap_data
import logging
from streamlit_agraph import agraph, Node, Edge, Config

# Load environment variables
load_dotenv()

# Set up OpenAI client
key = os.getenv("OPENAI_API_KEY")
if not key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
client = OpenAI(api_key=key)

# Initialize Firebase
firebase_key = os.getenv('FIREBASE_KEY')
if firebase_key:
    try:
        # Decode the base64 encoded key
        decoded_key = base64.b64decode(firebase_key).decode('utf-8')
        firebase_key_dict = json.loads(decoded_key)
        
        # Ensure the "type" field is present
        if "type" not in firebase_key_dict:
            firebase_key_dict["type"] = "service_account"
        
        with open('firebase-key.json', 'w') as f:
            json.dump(firebase_key_dict, f)
        cred = credentials.Certificate('firebase-key.json')
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase initialized successfully")
    except Exception as e:
        print(f"Error initializing Firebase: {str(e)}")
else:
    print("FIREBASE_KEY environment variable not found")

# Set Streamlit page config
st.set_page_config(page_title="Step 1 Buddy", page_icon="⚕️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true});</script>
""", unsafe_allow_html=True)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = None
if 'graph_analysis' not in st.session_state:
    st.session_state.graph_analysis = None
if 'query_overview' not in st.session_state:
    st.session_state.query_overview = None

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
    streamlit_analytics.start_tracking(firestore_key_file="firebase-key.json", firestore_collection_name="counts")
    
    st.title("Step 1 Buddy")

    # Add new tabs for disclosures and mindmap
    tab1, tab2, tab3 = st.tabs(["Main", "Mindmap", "Disclosures"])

    topics = ["immunology", "gastroenterology", "cell biology"]
    selected_topic = st.sidebar.selectbox("Select a topic", topics, key="topic_selectbox")
    video_data, index, embeddings = load_and_preprocess_data(selected_topic)

    with tab1:
        main_tab_content(video_data, index, embeddings)

    with tab2:
        mindmap_tab_content(video_data)

    with tab3:
        disclosures_tab_content()

    streamlit_analytics.stop_tracking(firestore_key_file="firebase-key.json", firestore_collection_name="counts")

def main_tab_content(video_data, index, embeddings):
    # Content for the main tab
    user_query = st.text_input("Enter your question:", key="user_query_input")
    submit_button = st.button("Submit", key="submit_button")

    if submit_button and user_query:
        with st.spinner("Searching for relevant information..."):
            relevant_passages = retrieve_passages(user_query, index, embeddings, video_data)

        context = " ".join([p["text"] for p in relevant_passages])
        
        with st.spinner("Generating answer..."):
            answer = generate_answer(user_query, context)

        st.subheader("Generated Answer:")
        st.write(answer)

        # Store values in session state
        st.session_state.user_query = user_query
        st.session_state.answer = answer
        st.session_state.relevant_passages = relevant_passages

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

        # Add a message about the Mindmap
        st.info("To create a Mindmap for this query, please go to the Mindmap tab.")

    # Add the feedback button at the end of the main tab
    st.markdown("---")
    st.markdown(
        """
        <div style="display: flex; justify-content: center; margin-top: 30px;">
            <a href="https://forms.gle/ht6MH14t8kFqrrni6" target="_blank">
                <button style="
                    font-size: 18px;
                    padding: 12px 24px;
                    background-color: #FFB347;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    transition: all 0.3s ease;
                ">
                    ⚕️ Leave Feedback if you liked it! ⚕️
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

def mindmap_tab_content(video_data):
    st.header("Interactive Mindmap")
    
    if 'user_query' in st.session_state and 'answer' in st.session_state and 'relevant_passages' in st.session_state:
        if st.button("Generate Mindmap"):
            with st.spinner("Generating Mindmap..."):
                try:
                    # Generate Mindmap
                    mindmap_structure, mindmap_analysis = get_mindmap_data(
                        st.session_state.user_query,
                        st.session_state.relevant_passages,
                        st.session_state.answer,
                        video_data
                    )
                    st.session_state.mindmap_structure = mindmap_structure
                    st.session_state.mindmap_analysis = mindmap_analysis

                    st.success("Mindmap generated successfully!")
                    logging.info("Mindmap generated successfully")
                except Exception as e:
                    st.error(f"Error generating mindmap: {str(e)}")
                    logging.error(f"Error generating mindmap: {str(e)}", exc_info=True)

    if st.session_state.get('mindmap_structure'):
        mermaid_code = convert_to_mermaid(st.session_state.mindmap_structure)
        st.markdown(f"""
        ```mermaid
        {mermaid_code}
        ```
        """)

        st.subheader("Mindmap Analysis")
        if st.session_state.mindmap_analysis:
            st.markdown(st.session_state.mindmap_analysis)
        else:
            st.info("No mindmap analysis available yet.")
    else:
        st.info("Generate a mindmap by submitting a query in the Main tab and then clicking the 'Generate Mindmap' button above.")

def convert_to_mermaid(mindmap_structure):
    lines = mindmap_structure.split('\n')
    mermaid_lines = ["mindmap"]
    current_level = 0
    
    for line in lines:
        level = line.count('#')
        title = line.strip('#').strip()
        
        if not title:
            continue
        
        if level > current_level:
            mermaid_lines.append("  " * (level - 1) + f"{{")
        elif level < current_level:
            mermaid_lines.append("  " * (level - 1) + f"}}")
        
        mermaid_lines.append("  " * level + title)
        current_level = level
    
    while current_level > 0:
        mermaid_lines.append("  " * (current_level - 1) + f"}}")
        current_level -= 1
    
    return "\n".join(mermaid_lines)

def disclosures_tab_content():
    st.header("Disclosures")
    with open("disclosures.txt", "r") as f:
        disclosures_content = f.read()
    st.markdown(disclosures_content)

if __name__ == "__main__":
    main()
