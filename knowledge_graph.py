import networkx as nx
from pyvis.network import Network
from typing import List, Dict
import os
from collections import defaultdict
from openai import OpenAI

def generate_knowledge_graph(query: str, relevant_passages: List[Dict], answer: str, all_data: List[Dict]) -> str:
    G = nx.Graph()
    G.add_node(query, color='#ADD8E6', size=30, title=query)
    
    # Use OpenAI to extract key concepts from the query and answer
    key_concepts = extract_key_concepts(query, answer)
    
    # Add key concepts as nodes
    for concept in key_concepts:
        G.add_node(concept, color='#90EE90', size=25, title=concept)
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
            G.add_node(topic, color='#FFFFE0', size=20, title=topic)
            G.add_edge(concept, topic)
    
    # Use OpenAI to generate meaningful connections
    connections = generate_connections(query, key_concepts, list(related_topics.keys()))
    
    # Add connections to the graph
    for connection in connections:
        G.add_edge(connection['source'], connection['target'], title=connection['relationship'])

    # Create Pyvis network
    net = Network(height="600px", width="100%", bgcolor="#FFFFFF", font_color="black")
    net.from_nx(G)
    
    # Customize the graph appearance
    net.toggle_physics(True)
    net.show_buttons(filter_=['physics'])
    
    # Save the graph as an HTML file
    html_file = "knowledge_graph.html"
    net.save_graph(html_file)
    
    return html_file

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
