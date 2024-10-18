import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict
import os
from collections import defaultdict
from openai import OpenAI

def generate_knowledge_graph(query: str, relevant_passages: List[Dict], answer: str, all_data: List[Dict]) -> go.Figure:
    G = nx.Graph()
    G.add_node(query, color='lightblue', size=3000)
    
    # Use OpenAI to extract key concepts from the query and answer
    key_concepts = extract_key_concepts(query, answer)
    
    # Add key concepts as nodes
    for concept in key_concepts:
        G.add_node(concept, color='lightgreen', size=2000)
        G.add_edge(query, concept)
    
    # Create a dictionary to store related topics and their connections
    related_topics = defaultdict(set)
    
    # Process all data to find related topics
    for item in all_data:
        for concept in key_concepts:
            if concept.lower() in item['text'].lower():
                related_topics[concept].add(item['video_title'])
    
    # Add related topics as nodes and connect them
    for concept, topics in related_topics.items():
        for topic in topics:
            G.add_node(topic, color='lightyellow', size=1500)
            G.add_edge(concept, topic)
    
    # Use OpenAI to generate meaningful connections
    connections = generate_connections(query, key_concepts, list(related_topics.keys()))
    
    # Add connections to the graph
    for connection in connections:
        G.add_edge(connection['source'], connection['target'], label=connection['relationship'])
    
    # Prepare data for Plotly
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)  # None to break the line
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)  # None to break the line

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_color = []
    node_size = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(G.nodes[node]['color'])
        node_size.append(G.nodes[node]['size'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=node_color,
            line_width=2
        ),
        hoverinfo='text'
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Knowledge Graph for: {query}',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    return fig

def extract_key_concepts(query: str, answer: str) -> List[str]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract 5-7 key concepts from the given query and answer."},
            {"role": "user", "content": f"Query: {query}\nAnswer: {answer}"}
        ]
    )
    concepts = response.choices[0].message.content.strip().split('\n')
    return [concept.strip() for concept in concepts if concept.strip()]

def generate_connections(query: str, key_concepts: List[str], related_topics: List[str]) -> List[Dict]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate meaningful connections between the query, key concepts, and related topics."},
            {"role": "user", "content": f"Query: {query}\nKey Concepts: {', '.join(key_concepts)}\nRelated Topics: {', '.join(related_topics)}"}
        ]
    )
    connections_text = response.choices[0].message.content.strip().split('\n')
    connections = []
    for connection in connections_text:
        parts = connection.split(' - ')
        if len(parts) == 3:
            connections.append({
                'source': parts[0].strip(),
                'target': parts[2].strip(),
                'relationship': parts[1].strip()
            })
    return connections
