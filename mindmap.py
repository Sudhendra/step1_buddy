import streamlit as st
from openai import OpenAI
import os
import time

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_mindmap(query: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical expert specializing in USMLE Step 1 content. Create a detailed mindmap in markdown format for the given query, focusing on USMLE Step 1 concepts and their interconnections."},
                {"role": "user", "content": f"Create a detailed mindmap for the following USMLE Step 1 related query: {query}"}
            ],
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating mindmap: {str(e)}")
        return "Sorry, I couldn't generate a mindmap at this time."

def display_mindmap(mindmap_content: str):
    st.code(mindmap_content, language="markdown")

def mindmap_tab_content():
    st.header("Mindmap")
    
    if 'user_query' in st.session_state and st.session_state.user_query:
        query = st.session_state.user_query
        st.write(f"Generating mindmap for: {query}")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Simulate progress while generating the mindmap
        for i in range(100):
            # Update progress bar
            progress_bar.progress(i + 1)
            
            if i == 0:
                # Start generating the mindmap
                mindmap = generate_mindmap(query)
            
            # Add a small delay to make the progress bar visible
            time.sleep(0.05)
        
        # Display the generated mindmap
        display_mindmap(mindmap)
        
        # Remove the progress bar
        progress_bar.empty()
    else:
        st.write("Please enter a query in the Main tab to generate a mindmap.")
