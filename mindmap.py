import json
from openai import OpenAI
import os
import base64

def generate_mindmap(query, relevant_passages, answer, all_data):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Prepare the context for OpenAI
    context = "\n".join([f"- {p['text']}" for p in relevant_passages])
    
    # Generate mindmap content using OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Create a detailed mindmap structure based on the given query. Use Markdown format with # for main topics, ## for subtopics, and ### for further details. Ensure comprehensive coverage of the USMLE Step 1 syllabus."},
            {"role": "user", "content": f"Query: {query}\nContext: {context}\nAnswer: {answer}\nCreate a detailed mindmap structure in Markdown format:"}
        ]
    )
    
    mindmap_structure = response.choices[0].message.content.strip()

    # Generate analysis and summary
    analysis_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Analyze the mindmap and provide a detailed analysis of how the query relates to other topics in the USMLE Step 1 syllabus."},
            {"role": "user", "content": f"Query: {query}\nMindmap structure:\n{mindmap_structure}\nProvide a detailed analysis:"}
        ]
    )
    
    analysis = analysis_response.choices[0].message.content.strip()
    
    # Create HTML content for the mindmap
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.jsdelivr.net/npm/markmap-autoloader"></script>
    </head>
    <body>
        <div id="mindmap"></div>
        <script>
            const mindmapContent = `{mindmap_structure}`;
            markmap.autoLoader.renderString(mindmapContent, document.getElementById('mindmap'));
        </script>
    </body>
    </html>
    """

    # Save the HTML content to a file
    html_file = "mindmap.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_file, analysis, mindmap_structure

def get_mindmap_data(query, relevant_passages, answer, all_data):
    html_file, analysis, mindmap_structure = generate_mindmap(query, relevant_passages, answer, all_data)
    return html_file, analysis, mindmap_structure

