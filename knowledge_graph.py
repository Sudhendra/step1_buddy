import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict

def generate_knowledge_graph(query: str, relevant_passages: List[Dict], answer: str) -> plt.Figure:
    G = nx.Graph()
    G.add_node(query, color='lightblue', size=2000)
    
    for i, passage in enumerate(relevant_passages):
        passage_node = f"Passage {i+1}"
        G.add_node(passage_node, color='lightgreen', size=1500)
        G.add_edge(query, passage_node)
        
        keywords = extract_keywords(passage['text'])
        for keyword in keywords:
            G.add_node(keyword, color='lightyellow', size=1000)
            G.add_edge(passage_node, keyword)
    
    answer_keywords = extract_keywords(answer)
    for keyword in answer_keywords:
        G.add_node(keyword, color='lightpink', size=1000)
        G.add_edge(query, keyword)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color=[node[1]['color'] for node in G.nodes(data=True)],
            node_size=[node[1]['size'] for node in G.nodes(data=True)])
    
    return plt.gcf()

def extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    # Implement keyword extraction logic here
    # For simplicity, we'll just split the text and take the first few words
    words = text.split()
    return words[:num_keywords]
