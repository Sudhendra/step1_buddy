import json
from openai import OpenAI
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64
from pyvis.network import Network

def extract_key_terms(relevant_passages):
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

def generate_mindmap(query, relevant_passages, answer):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    key_terms = extract_key_terms(relevant_passages)
    key_terms_text = "\n".join([f"- {term}" for term in key_terms])

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Create a detailed and accurate mindmap structure based on the given query and key terms. Use Markdown format with # for main topics, ## for subtopics, and ### for further details."
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nKey Terms:\n{key_terms_text}\n\nAnswer: {answer}\n\nCreate a detailed mindmap structure in Markdown format:"
            }
        ],
        max_tokens=800,
        temperature=0.7,
    )

    mindmap_structure = response.choices[0].message.content.strip()
    return mindmap_structure

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

    net = Network(height="600px", width="100%", bgcolor="#FFFFFF", font_color="black")
    net.from_nx(G)

    for node in net.nodes:
        level = node_levels[node['id']]
        size = 30 * (max_level - level + 1)
        color = f"rgb({255 - 30*level}, {100 + 30*level}, 255)"
        node.update({'size': size, 'color': color})

    net.set_options("""
    var options = {
      "nodes": {
        "shape": "dot",
        "font": {
          "size": 16,
          "face": "Tahoma"
        }
      },
      "edges": {
        "color": {
          "color": "#888888",
          "highlight": "#000000"
        },
        "smooth": {
          "type": "cubicBezier",
          "forceDirection": "horizontal"
        }
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "levelSeparation": 150,
          "nodeSpacing": 100,
          "treeSpacing": 200,
          "direction": "LR"
        }
      },
      "physics": {
        "hierarchicalRepulsion": {
          "centralGravity": 0.0,
          "springLength": 100,
          "springConstant": 0.01,
          "nodeDistance": 120,
          "damping": 0.09
        },
        "minVelocity": 0.75,
        "solver": "hierarchicalRepulsion"
      }
    }
    """)

    html_file = "mindmap.html"
    net.save_graph(html_file)
    
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    return content

def get_mindmap_data(query, relevant_passages, answer):
    mindmap_structure = generate_mindmap(query, relevant_passages, answer)
    return mindmap_structure
