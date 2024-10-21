import json
from openai import OpenAI
import os
import base64

def generate_mindmap(query, relevant_passages, answer, all_data):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Prepare the context for OpenAI
    context = "\n".join([f"- {p['text']}" for p in relevant_passages])
    
    # Generate mindmap content using OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Create a detailed mindmap structure based on the given query. Use Markdown format with # for main topics, ## for subtopics, and ### for further details. Ensure comprehensive coverage of the USMLE Step 1 syllabus. Start with the main topic using a single #."},
            {"role": "user", "content": f"Query: {query}\nContext: {context}\nAnswer: {answer}\nCreate a detailed mindmap structure in Markdown format:"}
        ]
    )
    
    mindmap_structure = response.choices[0].message.content.strip()

    # Generate analysis and summary
    analysis_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Analyze the mindmap and provide a detailed analysis of how the query relates to other topics in the USMLE Step 1 syllabus."},
            {"role": "user", "content": f"Query: {query}\nMindmap structure:\n{mindmap_structure}\nProvide a detailed analysis:"}
        ]
    )
    
    analysis = analysis_response.choices[0].message.content.strip()
    
    return mindmap_structure, analysis

def get_mindmap_data(query, relevant_passages, answer, all_data):
    mindmap_structure, analysis = generate_mindmap(query, relevant_passages, answer, all_data)
    return mindmap_structure, analysis
