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
    # Extract key terms from the relevant passages
    context = " ".join([p['text'] for p in relevant_passages])
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
        model="gpt-4o-mini",
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
        model="gpt-4o-mini",
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

def get_mindmap_data(query, relevant_passages, answer, all_data):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Extract key terms from relevant passages
    key_terms = extract_key_terms(relevant_passages)

    # Prepare the context for OpenAI
    key_terms_text = "\n".join([f"- {term}" for term in key_terms])

    # Generate mindmap content using OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create a detailed and accurate mindmap structure based on the given query and key terms. "
                        "The output should be a JSON array of objects, where each object represents a node in the mindmap. "
                        "Each node should have the following properties: "
                        "id (integer), label (string), group (integer), and optionally parent (integer). "
                        "The main topic (query) should have id 1 and no parent. "
                        "Subtopics should have increasing ids and reference their parent's id. "
                        "Use group 1 for the main topic, group 2 for primary subtopics, and group 3 for secondary subtopics. "
                        "Ensure the content is grounded in the provided key terms and aligns with the USMLE Step 1 syllabus."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\nKey Terms:\n{key_terms_text}\n\n"
                        f"Answer: {answer}\n\n"
                        "Create a detailed mindmap structure in the specified JSON format, ensuring it is comprehensive, informative, and grounded in the key terms:"
                    )
                }
            ],
            max_tokens=1000,
            temperature=0.2,
        )

        mindmap_data = json.loads(response.choices[0].message.content.strip())
        
        # Validate the structure of mindmap_data
        if not isinstance(mindmap_data, list):
            raise ValueError("Mindmap data is not a list")
        
        for node in mindmap_data:
            if not all(key in node for key in ['id', 'label', 'group']):
                raise ValueError("Mindmap node is missing required keys")

    except json.JSONDecodeError:
        print("Error: Invalid JSON response from OpenAI")
        mindmap_data = generate_fallback_mindmap(query, key_terms)
    except Exception as e:
        print(f"Error generating mindmap: {str(e)}")
        mindmap_data = generate_fallback_mindmap(query, key_terms)

    return mindmap_data

def generate_fallback_mindmap(query, key_terms):
    fallback_data = [
        {"id": 1, "label": query, "group": 1}
    ]
    for i, term in enumerate(key_terms[:5], start=2):
        fallback_data.append({"id": i, "label": term, "group": 2, "parent": 1})
    return fallback_data
