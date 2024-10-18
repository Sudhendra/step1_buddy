import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict
import openai
import os
from collections import defaultdict

def generate_knowledge_graph(query: str, relevant_passages: List[Dict], answer: str, all_data: List[Dict]) -> plt.Figure:
    G = nx.Graph()
    G.add_node(query, color='lightblue', size=2000)
    
    # Use OpenAI to extract key concepts from the query and answer
    key_concepts = extract_key_concepts(query, answer)
    
    # Add key concepts as nodes
    for concept in key_concepts:
        G.add_node(concept, color='lightgreen', size=1500)
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
            G.add_node(topic, color='lightyellow', size=1000)
            G.add_edge(concept, topic)
    
    # Use OpenAI to generate meaningful connections
    connections = generate_connections(query, key_concepts, list(related_topics.keys()))
    
    # Add connections to the graph
    for connection in connections:
        G.add_edge(connection['source'], connection['target'], label=connection['relationship'])
    
    # Create the plot
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    plt.figure(figsize=(16, 12))
    nx.draw(G, pos, with_labels=True, node_color=[node[1]['color'] for node in G.nodes(data=True)],
            node_size=[node[1]['size'] for node in G.nodes(data=True)], font_size=8)
    
    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    
    plt.title(f"Knowledge Graph for: {query}")
    return plt.gcf()

def extract_key_concepts(query: str, answer: str) -> List[str]:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract 5-7 key concepts from the given query and answer."},
            {"role": "user", "content": f"Query: {query}\nAnswer: {answer}"}
        ]
    )
    concepts = response.choices[0].message['content'].strip().split('\n')
    return [concept.strip() for concept in concepts if concept.strip()]

def generate_connections(query: str, key_concepts: List[str], related_topics: List[str]) -> List[Dict]:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate meaningful connections between the query, key concepts, and related topics."},
            {"role": "user", "content": f"Query: {query}\nKey Concepts: {', '.join(key_concepts)}\nRelated Topics: {', '.join(related_topics)}"}
        ]
    )
    connections_text = response.choices[0].message['content'].strip().split('\n')
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
