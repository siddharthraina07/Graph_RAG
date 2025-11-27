#CONTAINS ALL FUNCTIONALITIES plus new render graph

from flask import Flask, render_template, request, jsonify
from neo4j import GraphDatabase
from langchain.prompts import PromptTemplate
from pyvis.network import Network
import random
import ssl
import json
from datetime import datetime
#from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
#from langchain.chains import GraphCypherQAChain
from flask_cors import CORS
import textwrap
import os
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_community.vectorstores import Neo4jVector
from langchain.schema import HumanMessage  # Import the appropriate schema
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List, Tuple

app = Flask(__name__)
CORS(app)




# ssl_context = ssl.create_default_context()
# ssl_context.check_hostname = False
# ssl_context.verify_mode = ssl.CERT_NONE

# Custom Embeddings Class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

embeddings = SentenceTransformerEmbeddings()

graph = Neo4jGraph(
    url=uri,
    username=user,
    password=password
)

vector_index_chunk = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="CHUNK",
    text_node_properties=["content"],
    embedding_node_property="embedding_new",
    url=uri,
    username=user,
    password=password
)

driver = GraphDatabase.driver(
    uri,
    auth=(user, password)
    # trust="TRUST_ALL_CERTIFICATES",
    # connection_timeout=30,
    # max_connection_lifetime=3600
)

def remove_lucene_chars(text: str) -> str:
    chars = ['+', '-', '&', '|', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\','/']
    for char in chars:
        text = text.replace(char, ' ')
    return text

# Function to retrieve data and create the graph
def create_graph(query):
    # Create the Pyvis network object
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    physics_options = {
        "physics": {
            "enabled": True,
            "barnesHut": {
                "theta": 0.5,
                "gravitationalConstant": -10000,
                "centralGravity": 0.3,
                "springLength": 600,
                "springConstant": 0.1,
                "damping": 0.09,
                "avoidOverlap": 1
            },
        }
    }

    net.set_options(json.dumps(physics_options))
    # Custom color scheme based on node type
    color_map = {
    # Case Status Colors - Using distinct pastels
    'CASE': '#ADD8E6',      # Light Blue
    'CIVIL': '#FFB6C1',     # Light Pink
    'CRIMINAL': '#DDA0DD',  # Plum
    'PENDING': '#FFD700',   # Gold
    'DISPOSED': '#98FB98',  # Pale Green
    
    # Legal Entity Colors - Using distinct bold colors
    'ANALYSIS': '#1f77b4',       # Steel Blue
    'ARGUMENTS': '#ff7f0e',      # Dark Orange
    'COURT': '#d62728',          # Brick Red
    'DATE': '#9467bd',           # Medium Purple
    'DECISION': '#8c564b',       # Brown
    'DOCUMENT': '#e377c2',  # Orchid
    'FACT': '#FFE135',          # Banana Yellow
    'GPE': '#2E8B57',           # Sea Green
    'GROUND': '#17becf',        # Deep Sky Blue
    'JUDGE': '#FF6B6B',          # Coral Red
    'LAWYER': '#32CD32',         # Lime Green
    'ORDER': '#9370DB',          # Medium Purple
    'PARTICULAR': '#FFA500',     # Pure Orange
    'PETITIONER': '#BDB76B',     # Dark Khaki
    'PRAYER': '#CD853F',         # Peru Brown
    'PRE_RELIED': '#87CEEB',     # Sky Blue
    'PRECEDENT': '#FF69B4',      # Hot Pink
    'PROVISION': '#A0522D',      # Sienna
    'RESPONDENT': '#4682B4',     # Steel Blue
    'RLC': '#FF8C00',           # Dark Orange
    'STATUTE': '#20B2AA',        # Light Sea Green
    'SUBJECT': '#DAA520',        # Golden Rod
    'WITNESS': '#6495ED'         # Cornflower Blue
    }

    # Helper function to get truncated label
    def get_truncated_label(n,n_type):
        if 'name' in n: return n['name'][:20] + "..." if len(n['name']) > 20 else n['name']
        if 'purpose' in n: return n['purpose'][:20] + "..." if len(n['purpose']) > 20 else n['purpose']
        if 'Case_Number' in n: return n['Case_Number'][:20] + "..." if len(n['Case_Number']) > 20 else n['Case_Number']
        if 'Case_Type' in n: return n['Case_Type'][:20] + "..." if len(n['Case_Type']) > 20 else n['Case_Type']
        if 'CNR_No' in n: return n['CNR_No'][:20] + "..." if len(n['CNR_No']) > 20 else n['CNR_No']
        if 'Name_of_the_State' in n: return n['Name_of_the_State'][:20] + "..." if len(n['Name_of_the_State']) > 20 else n['Name_of_the_State']

        return n_type  # Default label if none of the fields are found

    # Helper function to format properties for display in title
    def format_properties(n):
        return "\n".join([f"{key}: {value}" for key, value in n.items()])

    # Retrieve nodes and edges from Neo4j
    with driver.session() as session:
        result = session.run(query)
        # Process each node and relationship
        for record in result:
            print(record)
            for node_label in ['n', 'm']:  # For both start and end nodes in relationships
                node = record[node_label]
                print(node_label)
                print(node)
                n_id = node.id
                n_type = list(node.labels)[0]  # Assuming single label per node for simplicity
                n_properties = node.items()
                #print("n_type ",n_type)
                #print("n_props ",n_properties)
                
                # Determine the display label and color based on type and properties
                n_label_display = get_truncated_label(dict(n_properties),n_type)
                n_color = color_map.get(n_type, '#D3D3D3')  # Default color is light gray
                net.add_node(n_id, 
                             label=n_label_display, 
                             title=f"Type: {n_type}\nProperties:\n{format_properties(dict(n_properties))}", 
                             color=n_color)
            # Add the edge between nodes
            rel = record['r']
            net.add_edge(record['n'].id, record['m'].id, title=f"{rel.type}", color="#888888")
    
    timestamp = datetime.now().strftime("%H%M%S")
    output_path = f'saved_graphs/knowledge_graph_{timestamp}.html'
    net.save_graph(output_path)
    return net

@app.route('/')
def index():
    queries = {
        'case_numbers': "MATCH (n:CASE) RETURN DISTINCT n.Case_Number AS name",
        'provisions': "MATCH (n:PROVISION) RETURN DISTINCT n.name AS name",
        'subjects': "MATCH (n:SUBJECT) RETURN DISTINCT n.name AS name",
        'judges': "MATCH (n:JUDGE) RETURN DISTINCT n.name AS name"
    }
    
    results = {}
    with driver.session() as session:
        for key, query in queries.items():
            result = session.run(query)
            results[key] = [record["name"] for record in result]
        print(results)
    return render_template('final_index.html', **results)

@app.route('/overall_graph', methods=['GET'])
def overall_graph():
    query = """
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    LIMIT 1000
    """
    graph_html = create_graph(query)
    return graph_html.generate_html()

@app.route('/search_graph', methods=['POST'])
def search_graph():
    case_numbers = request.form.getlist('case_numbers')
    provisions = request.form.getlist('provisions')
    subjects = request.form.getlist('subjects')
    judges = request.form.getlist('judges')

    conditions = []
    if case_numbers:
        conditions.append(f"n:CASE AND n.Case_Number IN {case_numbers}")
    if provisions:
        conditions.append(f"n:PROVISION AND n.name IN {provisions}")
    if subjects:
        conditions.append(f"n:SUBJECT AND n.name IN {subjects}")
    if judges:
        conditions.append(f"n:JUDGE AND n.name IN {judges}")
    
    where_clause = " OR ".join(conditions)
    
    query = f"""
    MATCH (n)-[r]-(m)
    WHERE {where_clause}
    RETURN n, r, m
    LIMIT 1000
    """
    
    graph_html = create_graph(query)
    return graph_html.generate_html()

@app.route('/search_text', methods=['POST'])
def search_text():
    text = request.form.get('text')
    print(text)
    query = f"""
    MATCH (n)-[r]-(m)
    WHERE 
    NOT 'CHUNK' IN labels(n) AND
    NOT 'CHUNK' IN labels(m) AND
    (
        any(prop IN keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower('{text}')) OR
        any(prop IN keys(m) WHERE toLower(toString(m[prop])) CONTAINS toLower('{text}'))
    )
    RETURN n, r, m
    LIMIT 1000
    """
    graph_html = create_graph(query)
    return graph_html.generate_html()

@app.route('/add_node', methods=['POST'])
def add_node():
    node1_type = request.form.get('node1_type')
    node1_name = request.form.get('node1_name')
    relation_name = request.form.get('relation_name')
    node2_type = request.form.get('node2_type')
    node2_name = request.form.get('node2_name')
    if node1_type=="CASE" and node2_type!="CASE":   
        query = f"""
    MERGE (n1:{node1_type} {{Case_Number: '{node1_name}'}})
    MERGE (n2:{node2_type} {{name: '{node2_name}'}})
    MERGE (n1)-[r:{relation_name}]->(n2)
    """
    if node2_type=="CASE" and node1_type!="CASE":
        query = f"""
    MERGE (n1:{node1_type} {{name: '{node1_name}'}})
    MERGE (n2:{node2_type} {{Case_Number: '{node2_name}'}})
    MERGE (n1)-[r:{relation_name}]->(n2)
    """
    if node1_type=="CASE" and node2_type=="CASE": 
        query = f"""
        MERGE (n1:{node1_type} {{Case_Number: '{node1_name}'}})
        MERGE (n2:{node2_type} {{Case_Number: '{node2_name}'}})
        MERGE (n1)-[r:{relation_name}]->(n2)
        """
    else:
        query = f"""
        MERGE (n1:{node1_type} {{name: '{node1_name}'}})
        MERGE (n2:{node2_type} {{name: '{node2_name}'}})
        MERGE (n1)-[r:{relation_name}]->(n2)
        """
    with driver.session() as session:
        session.run(query)
    
    return jsonify({'status': 'Node and relationship added'})

@app.route('/delete_node', methods=['POST'])
def delete_node():
    node_type = request.form.get('node_type')
    node_name = request.form.get('node_name')
    if node_type=="CASE":
        query = f"""
        MATCH (n:{node_type} {{Case_Number: '{node_name}'}})
        DETACH DELETE n
        """
    else:
        query = f"""
        MATCH (n:{node_type} {{name: '{node_name}'}})
        DETACH DELETE n
        """
    with driver.session() as session:
        session.run(query)
    
    return jsonify({'status': 'Node deleted'})

@app.route('/graph_qa', methods=['POST'])
def graph_qa():
    query = request.form.get('query')
    query=remove_lucene_chars(query)
    

    # graph=Neo4jGraph(
    #     url=uri,
    #     username=user,
    #     password=password
    # )
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    #model_name = "llama3.1"
    
    #llm = Ollama(temperature=0,base_url="http://164.52.205.203:11434", model="llama3.1")
    
    #llm = ChatOpenAI(model_name="gpt-4o-mini")
    schema=graph.schema

    # Cypher prompt template
    cypher_template="""You are a Neo4j Graph Database Expert. Given an input question, generate a Cypher query on a predefined full text index "entity" containing every nodes content.\n\nHere is the schema information\n{schema}.\n
    Do not include any text except the generated Cypher statement and always generate the cypher.
    
    EXAMPLE PROCEDURE:
    User input: Find the lawyer in crrfc 7 
    Cypher query:CALL db.index.fulltext.queryNodes('entity', "crrfc 7", {{limit: 10}})
    YIELD node, score
    WITH node
    MATCH (node)-[]-(lawyer:LAWYER)
    RETURN node as nod, lawyer as lawyer LIMIT 50

    User Input: Who is karuna gelhot
    Cypher query:CALL db.index.fulltext.queryNodes('entity', "karuna gelhot", {{limit: 10}})
    YIELD node, score
    RETURN node as nod LIMIT 50
    
                        ACTUAL PROCEDURE:
                        User input: {query}\nCypher query: """

    # Create the prompt template
    cypher_prompt = PromptTemplate(
                    template=cypher_template,input_variables=["schema","query"]
                    )
    
    QA_GENERATION_TEMPLATE = """you are a RAG CHATBOT. Answer the Question which is the provided context:

    Context: {context}

    Question: {question}

    Dont make it seem like you answering by looking at context. 
    Dont provide any other information and stick the question.
    """
    
    # QA_GENERATION_TEMPLATE="""
    # Task: answer the question you are given based on the list of dictionary provided as context. Make sense of that context by unpacking the dictionary and answer.
    # INSTRUCTIONS: 
    # You are an assistant that helps to form nice and human understandable answers.
    # When the provided information contains multiple elements, structure your answer as a bulleted or numbered list to enhance clarity and readability.
    # The provided information is authoritative; do not doubt it or try to use your internal knowledge to correct it.
    # Make the answer sound like a response to the question without mentioning that you based the result on the given information.

    # Here's the information: {context}    
    # Question: {question}
    # Answer: 
    #"""
    qa_prompt = PromptTemplate(template=QA_GENERATION_TEMPLATE,input_variables=["context","question"])

    chain = GraphCypherQAChain.from_llm(
    cypher_llm=llm,
    qa_llm=llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
    qa_prompt=qa_prompt,
    top_k=20,
    return_intermediate_steps=True,
    validate_cypher=True,
    allow_dangerous_requests=True
    )

    max_retries = 3  # Define a max retries limit to avoid infinite loop
    attempt = 0
    context=[]
    
    while attempt < max_retries:
        try:
            response = chain.invoke({"query": query, "schema": graph.schema})
            steps = response.get("intermediate_steps", 'No steps found')
            context = steps[1]["context"]
            chain_query=steps[0]["query"]
            print("cypher query context",context)
            print("cypher query chain_query",chain_query)
            print("cypher query response",response)

            # If context is non-empty, break the loop
            if chain_query and context:
                response = response.get("result", 'No result found')
                return jsonify({"response": response})
        except Exception as e:
                print(f"Error encountered: {str(e)}")

        attempt += 1
        print(f"Retrying... Attempt {attempt}")

    if context == []:
        context= [el.page_content for el in vector_index_chunk.similarity_search(query, k=5)]
        print("vector search context")
        prompt = qa_prompt.format(context=context, question=query)
        response = llm.invoke(prompt)  # Or use `llm(prompt)` if it's not a chat model
        response=response.content
        #respone=response.content
        #response2=llm(prompt)
        print("response")
        print(response)
        #print("response2 ",response2)

  

    return jsonify({"response": str(response)})

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
