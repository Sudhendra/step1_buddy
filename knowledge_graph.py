import networkx as nx
from pyvis.network import Network
from typing import List, Dict
import os
from collections import defaultdict
from openai import OpenAI
import itertools

def generate_knowledge_graph(query: str, relevant_passages: List[Dict], answer: str, all_data: List[Dict]) -> str:
    G = nx.Graph()
    G.add_node(query, color='#ADD8E6', size=30, title=query, group='query')
    
    # Extract key concepts from the query and answer
    key_concepts = extract_key_concepts(query, answer, relevant_passages)
    
    # Add key concepts as nodes
    for concept in key_concepts:
        G.add_node(concept, color='#90EE90', size=25, title=concept, group='key_concept')
        G.add_edge(query, concept)
        
        # Explain the relationship between query and concept
        explanation = explain_relationship(query, concept, relevant_passages)
        G.edges[query, concept]['title'] = explanation
    
    # Process all data to find related topics across all subjects
    related_topics = defaultdict(set)
    for item in all_data:
        for concept in key_concepts:
            if concept.lower() in item['text'].lower():
                related_topics[concept].add((item['video_title'], item['text']))
    
    # Add related topics as nodes and connect them
    for concept, topics in related_topics.items():
        for topic, text in topics:
            topic_node = f"{topic}: {text[:50]}..."
            G.add_node(topic_node, color='#FFFFE0', size=20, title=text, group=topic.split()[0])
            G.add_edge(concept, topic_node)
            
            # Explain the relationship between concept and topic
            explanation = explain_relationship(concept, topic, [{'text': text}])
            G.edges[concept, topic_node]['title'] = explanation
    
    # Create Pyvis network
    net = Network(height="800px", width="100%", bgcolor="#FFFFFF", font_color="black")
    net.from_nx(G)
    
    # Customize the graph appearance
    net.toggle_physics(True)
    net.show_buttons(filter_=['physics'])
    
    # Set node colors based on groups
    for node in net.nodes:
        if 'group' in node:
            if node['group'] == 'query':
                node['color'] = '#ADD8E6'
            elif node['group'] == 'key_concept':
                node['color'] = '#90EE90'
            elif 'immunology' in node['group'].lower():
                node['color'] = '#FFB3BA'
            elif 'gastroenterology' in node['group'].lower():
                node['color'] = '#BAFFC9'
            elif 'cell biology' in node['group'].lower():
                node['color'] = '#BAE1FF'
            else:
                node['color'] = '#FFFFE0'
    
    # Save the graph as an HTML file
    html_file = "knowledge_graph.html"
    net.save_graph(html_file)
    
    return html_file

def extract_key_concepts(query: str, answer: str, relevant_passages: List[Dict]) -> List[str]:
    context = " ".join([p['text'] for p in relevant_passages])
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract 5-10 key concepts from the given query, answer, and context. Focus on medical terms and concepts."},
            {"role": "user", "content": f"Query: {query}\nAnswer: {answer}\nContext: {context}"}
        ]
    )
    concepts = response.choices[0].message.content.strip().split('\n')
    return [concept.strip() for concept in concepts if concept.strip()]

def generate_connections(query: str, key_concepts: List[str], related_topics: List[str]) -> List[Dict]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate meaningful connections between the query, key concepts, and related topics. Focus on interdisciplinary connections and how concepts relate across different medical fields."},
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

def explain_relationship(source: str, target: str, relevant_passages: List[Dict]) -> str:
    context = " ".join([p['text'] for p in relevant_passages])
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Explain the relationship between two medical concepts based on the given context. Be concise and specific."},
            {"role": "user", "content": f"Source: {source}\nTarget: {target}\nContext: {context}\n\nExplain the relationship:"}
        ],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()
