import streamlit as st
import streamlit_analytics
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import json
from typing import List, Dict, Tuple
import os
from PIL import Image
import faiss
from openai import OpenAI
import pickle
import cv2
from dotenv import load_dotenv
from streamlit_agraph import agraph, Node, Edge, Config

# Load environment variables
load_dotenv()

# Set up OpenAI client
key = os.getenv("OPENAI_API_KEY")
if not key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
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
def generate_answer(query: str, context: str) -> Tuple[str, List[str]]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers medical questions based on the provided context. Always ground your answers in the given context and be concise. After your answer, provide a list of 3-5 related topics separated by commas."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
            ],
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
        
        # Split the answer and related topics
        parts = answer.split("\n\nRelated topics:")
        main_answer = parts[0]
        related_topics = parts[1].strip().split(", ") if len(parts) > 1 else []
        
        return main_answer, related_topics
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I couldn't generate an answer at this time.", []

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

# Add a new function to build the mindmap data
def build_mindmap(query: str, answer: str, related_topics: List[str]) -> Tuple[List[Node], List[Edge]]:
    nodes = [Node(id="query", label=query, size=25)]
    edges = []
    
    # Add answer node
    answer_node = Node(id="answer", label=answer[:50] + "...", size=20)
    nodes.append(answer_node)
    edges.append(Edge(source="query", target="answer"))
    
    # Add related topics
    for i, topic in enumerate(related_topics):
        topic_node = Node(id=f"topic_{i}", label=topic, size=15)
        nodes.append(topic_node)
        edges.append(Edge(source="answer", target=f"topic_{i}"))
    
    return nodes, edges

# Main Streamlit app
def main():
    streamlit_analytics.start_tracking()
    
    st.title("Step 1 Buddy")

    # Initialize session state for graph data
    if 'graph_data' not in st.session_state:
        st.session_state.graph_data = None

    # Add a new tab for disclosures
    tab1, tab2 = st.tabs(["Main", "Disclosures"])

    with tab1:
        topics = ["immunology", "gastroenterology", "cell biology"]
        selected_topic = st.selectbox("Select a topic", topics, key="topic_selectbox")

        video_data, index, embeddings = load_and_preprocess_data(selected_topic)
        user_query = st.text_input("Enter your question:", key="user_query_input")
        submit_button = st.button("Submit", key="submit_button")

        if submit_button and user_query:
            with st.spinner("Searching for relevant information..."):
                relevant_passages = retrieve_passages(user_query, index, embeddings, video_data)

            context = " ".join([p["text"] for p in relevant_passages])
            
            with st.spinner("Generating answer..."):
                answer, related_topics = generate_answer(user_query, context)

            st.subheader("Generated Answer:")
            st.write(answer)

            # Update session state with new graph data
            st.session_state.graph_data = build_mindmap(user_query, answer, related_topics)

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

        # Add the mindmap visualization within an expander
        with st.expander("View Interactive Mindmap"):
            st.subheader("Interactive Mindmap")
            if st.session_state.graph_data:
                nodes, edges = st.session_state.graph_data
                config = Config(width=750, 
                                height=500, 
                                directed=True, 
                                physics=True, 
                                hierarchical=False,
                                nodeHighlightBehavior=True, 
                                highlightColor="#F7A7A6",
                                collapsible=True)
                agraph(nodes=nodes, 
                       edges=edges, 
                       config=config)

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

    with tab2:
        st.header("Disclosures")
        with open("disclosures.txt", "r") as f:
            disclosures_content = f.read()
        st.markdown(disclosures_content)

    streamlit_analytics.stop_tracking(save_to_json=os.path.join("tracking", "tracking_data.json"))

if __name__ == "__main__":
    main()
