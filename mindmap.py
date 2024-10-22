import json
from openai import OpenAI
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64

def extract_key_terms(relevant_passages):
    # Extract key terms from the relevant passages
    context = " ".join([p['text'] for p in relevant_passages])
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Extract key medical terms and concepts from the provided text, focusing on USMLE Step 1 topics."
            },
            {
                "role": "user",
                "content": f"Text: {context}\n\nList the key terms as a comma-separated list:"
            }
        ],
        max_tokens=150,
        temperature=0.5,
    )
    key_terms = response.choices[0].message.content.strip().split(',')
    return [term.strip() for term in key_terms if term.strip()]

def generate_mindmap(query, relevant_passages, answer, all_data):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Extract key terms from relevant passages
    key_terms = extract_key_terms(relevant_passages)

    # Prepare the context for OpenAI
    key_terms_text = "\n".join([f"- {term}" for term in key_terms])

    # Generate mindmap content using OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": (
                    "Create a detailed and accurate mindmap structure based on the given query and key terms. "
                    "Use Markdown format with # for main topics, ## for subtopics, and ### for further details. "
                    "Ensure the content is grounded in the provided key terms and aligns with the USMLE Step 1 syllabus."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\nKey Terms:\n{key_terms_text}\n\n"
                    f"Answer: {answer}\n\n"
                    "Create a detailed mindmap structure in Markdown format, ensuring it is comprehensive, informative, and grounded in the key terms:"
                )
            }
        ],
        max_tokens=800,
        temperature=0.7,
    )

    mindmap_structure = response.choices[0].message.content.strip()

    # Generate analysis and summary
    analysis_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": (
                    "Provide a detailed analysis of how the query relates to the key terms and other topics in the USMLE Step 1 syllabus. "
                    "Focus on the interconnections and ensure the analysis is accurate and informative."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\nMindmap Structure:\n{mindmap_structure}\n\n"
                    "Provide a detailed analysis highlighting the connections between topics, ensuring minimal fluff and maximum informational value:"
                )
            }
        ],
        max_tokens=800,
        temperature=0.7,
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

    # Use a hierarchical layout instead of spring layout
    pos = nx.spring_layout(G, k=2, iterations=50)  # Increase k for more spacing

    plt.figure(figsize=(20, 12))  # Increase figure size
    ax = plt.gca()

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', arrows=True, arrowsize=15, width=1, alpha=0.6)

    # Draw nodes
    for node, (x, y) in pos.items():
        level = node_levels[node]
        color = plt.cm.YlOrRd(1 - (level - 1) / max_level)
        size = 2000 * (max_level - level + 1) / max_level  # Reduce node size
        rect = patches.Rectangle((x - size/20000, y - size/20000), size/10000, size/10000, 
                                 fill=True, facecolor=color, edgecolor='gray', 
                                 linewidth=1, alpha=0.7)
        ax.add_patch(rect)
        plt.text(x, y, node, ha='center', va='center', wrap=True, 
                 fontsize=8, fontweight='bold', color='black')  # Reduce font size

    plt.title("Mindmap", fontsize=24, fontweight='bold', pad=20)
    plt.axis('off')

    # Adjust the plot to fit all nodes
    plt.tight_layout()
    ax.margins(0.3)  # Increase margins

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    img.seek(0)
    plt.close()

    return base64.b64encode(img.getvalue()).decode()

def get_mindmap_data(query, relevant_passages, answer, all_data):
    mindmap_structure, analysis = generate_mindmap(query, relevant_passages, answer, all_data)
    mindmap_image = create_mindmap(mindmap_structure)
    return mindmap_image, analysis
