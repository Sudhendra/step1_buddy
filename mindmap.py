import streamlit as st
from openai import OpenAI
import os
import time
from streamlit_markmap import markmap

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_mindmap(query: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical expert specializing in USMLE Step 1 content. Create a detailed mindmap in markdown format for the given query, focusing on USMLE Step 1 concepts and their interconnections. Use only '#', '##', '###', etc. for hierarchy."},
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
        return "# Error\n## Sorry, I couldn't generate a mindmap at this time."

def generate_analysis(mindmap_content: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical expert specializing in USMLE Step 1 content. Provide a critical analysis report based on the given mindmap, expanding upon the topics to help USMLE Step 1 students understand this topic immediately."},
                {"role": "user", "content": f"Based on the following mindmap, provide a critical analysis report for USMLE Step 1 students:\n\n{mindmap_content}"}
            ],
            max_tokens=1500,
            n=1,
            stop=None,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating analysis: {str(e)}")
        return "Sorry, I couldn't generate an analysis at this time."

def display_mindmap(mindmap_content: str):
    # Display the raw markdown content
    st.subheader("Markdown Content:")
    st.code(mindmap_content, language="markdown")
    
    # Display the mindmap using markmap
    st.subheader("Interactive Mindmap:")
    markmap(mindmap_content)

def mindmap_tab_content():
    st.header("Mindmap")
    
    if 'user_query' in st.session_state and st.session_state.user_query:
        query = st.session_state.user_query
        st.write(f"Generating mindmap for: {query}")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Generate the mindmap
        mindmap = generate_mindmap(query)
        
        # Simulate progress while generating the mindmap
        for i in range(100):
            # Update progress bar
            progress_bar.progress(i + 1)
            time.sleep(0.05)
        
        # Display the generated mindmap
        display_mindmap(mindmap)
        
        # Generate and display the analysis
        st.subheader("Analysis")
        with st.spinner("Generating analysis..."):
            analysis = generate_analysis(mindmap)
        st.markdown(analysis)
        
        # Remove the progress bar
        progress_bar.empty()
    else:
        st.write("Please enter a query in the Main tab to generate a mindmap.")
