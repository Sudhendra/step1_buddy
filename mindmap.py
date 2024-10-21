import json
from openai import OpenAI
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import base64

def generate_mindmap(query, relevant_passages, answer, all_data):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Prepare the context for OpenAI
    context = "\n".join([f"- {p['text']}" for p in relevant_passages])
    
    # Generate mindmap content using OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Create a detailed mindmap structure based on the given query. Use Markdown format with # for main topics, ## for subtopics, and ### for further details. Ensure comprehensive coverage of the USMLE Step 1 syllabus. Start with the main topic using a single #. Focus on creating a balanced structure with meaningful connections between topics."},
            {"role": "user", "content": f"Query: {query}\nContext: {context}\nAnswer: {answer}\nCreate a detailed mindmap structure in Markdown format, ensuring a balanced and interconnected structure:"}
        ]
    )
    
    mindmap_structure = response.choices[0].message.content.strip()

    # Generate analysis and summary
    analysis_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Analyze the mindmap and provide a detailed analysis of how the query relates to other topics in the USMLE Step 1 syllabus. Focus on the interconnections between different branches of the mindmap."},
            {"role": "user", "content": f"Query: {query}\nMindmap structure:\n{mindmap_structure}\nProvide a detailed analysis, highlighting the connections between topics:"}
        ]
    )
    
    analysis = analysis_response.choices[0].message.content.strip()
    
    return mindmap_structure, analysis

def create_mindmap(mindmap_structure):
    G = nx.Graph()
    lines = mindmap_structure.split('\n')
    parent_stack = []
    
    for line in lines:
        level = line.count('#')
        title = line.strip('#').strip()
        
        if not title:
            continue
        
        while len(parent_stack) >= level:
            parent_stack.pop()
        
        if parent_stack:
            G.add_edge(parent_stack[-1], title)
        
        parent_stack.append(title)
        G.add_node(title)
    
    pos = nx.spring_layout(G, k=0.9, iterations=50)
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=8, font_weight='bold')
    
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Mindmap", fontsize=16)
    plt.axis('off')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    
    return base64.b64encode(img.getvalue()).decode()

def get_mindmap_data(query, relevant_passages, answer, all_data):
    mindmap_structure, analysis = generate_mindmap(query, relevant_passages, answer, all_data)
    mindmap_image = create_mindmap(mindmap_structure)
    return mindmap_image, analysis
