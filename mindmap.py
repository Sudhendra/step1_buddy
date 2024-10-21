import json
from openai import OpenAI
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    G = nx.DiGraph()
    lines = mindmap_structure.split('\n')
    parent_stack = []
    node_levels = {}
    max_level = 0

    for line in lines:
        level = line.count('#')
        title = line.strip('#').strip()
        
        if not title:
            continue
        
        while parent_stack and len(parent_stack) >= level:
            parent_stack.pop()
        
        if parent_stack:
            G.add_edge(parent_stack[-1], title)
        else:
            G.add_node(title)
        
        parent_stack.append(title)
        node_levels[title] = level
        max_level = max(max_level, level)

    if not G.nodes():
        G.add_node("No valid mindmap structure")
        node_levels["No valid mindmap structure"] = 1
        max_level = 1

    pos = nx.spring_layout(G, k=1, iterations=50)

    plt.figure(figsize=(16, 10))
    ax = plt.gca()

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)

    # Draw nodes
    for node, (x, y) in pos.items():
        level = node_levels[node]
        color = plt.cm.YlOrRd(1 - (level - 1) / max_level)
        size = 3000 * (max_level - level + 1) / max_level
        rect = patches.Rectangle((x - size/10000, y - size/10000), size/5000, size/5000, 
                                 fill=True, facecolor=color, edgecolor='gray', 
                                 linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        plt.text(x, y, node, ha='center', va='center', wrap=True, 
                 fontsize=10, fontweight='bold', color='black')

    plt.title("Mindmap", fontsize=20, fontweight='bold')
    plt.axis('off')

    # Adjust the plot to fit all nodes
    plt.tight_layout()
    ax.margins(0.2)

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    plt.close()

    return base64.b64encode(img.getvalue()).decode()

def get_mindmap_data(query, relevant_passages, answer, all_data):
    mindmap_structure, analysis = generate_mindmap(query, relevant_passages, answer, all_data)
    mindmap_image = create_mindmap(mindmap_structure)
    return mindmap_image, analysis
