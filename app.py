import os
import requests
import json
import openai
import re
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
# Set the API key and endpoint for the Azure OpenAI service

openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set the deployment ID for the Davinci03 model
deployment_id = 'aouy-txt-davinci-003' # This will correspond to the custom name you chose for your deployment when you deployed the model
TEXT_SEARCH_DOC_EMBEDDING_ENGINE = 'aouy-text-search-davinci-doc-001'
TEXT_SEARCH_QUERY_EMBEDDING_ENGINE = 'aouy-text-search-davinci-query-001'
TEXT_DAVINCI_001 = "aouy-txt-davinci003"

#Defining helper functions

#Splits text after sentences ending in a period. Combines n sentences per chunk.
def splitter(n, s):
    pieces = s.split(". ")
    list_out = [" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n)]
    return list_out

# Perform light data cleaning (removing redudant whitespace and cleaning up punctuation)
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

def search_docs(df, user_query, top_n=3, to_print=True):
    embedding = get_embedding(
        user_query,
        engine=TEXT_SEARCH_QUERY_EMBEDDING_ENGINE
    )
    df["similarities"] = df.curie_search.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    return res

# Set up Flask app
from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_query', methods=['POST'])
def process_query():

    index_name = "azureblob-index"
    # Create a client

    # Get user query from form
    query = request.form['query']

    prompt_keyword = 'Extract the Keyword given the text provided.\n\Text:\n'+" ".join([query])+ '\n\nKeyword:\n'

    # Extract keyword using OpenAI Completion API
    keyword_response = openai.Completion.create(
        engine=deployment_id,
        prompt=prompt_keyword,
        max_tokens=1024,
        temperature=0,
    )

    keyword = keyword_response['choices'][0]['text']


    index_name = "azureblob-index"
    # Create a client
    credential = DefaultAzureCredential()
    SEARCH_ENDPOINT="https://srch-imlginsyxfgjo.search.windows.net"
    
    client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=index_name, credential=credential)
    results = client.search(search_text=keyword)

    search_df = pd.DataFrame(results)

    document=search_df._get_value(0,'article')
    # (STEP 1) Chunking the document by creating a chunk every 10th sentence and creating a pandas DF
    document_chunks = splitter(10, normalize_text(document))
    df = pd.DataFrame(document_chunks, columns = ["chunks"])

    #(STEP 2): Create an embedding vector for each chunk that will capture the semantic meaning and overall topic of that chunk
    df['curie_search'] = df["chunks"].apply(lambda x : get_embedding(x, engine = TEXT_SEARCH_DOC_EMBEDDING_ENGINE))

    # (STEP 3) upon receiving a query for the specific document, embed the query in the same vector space as the context chunks
    document_specific_query = query
    res = search_docs(df, document_specific_query, top_n=2)

    result_n = [res["chunks"][i] for i in res.index]
    result_add =""
    for i in result_n:
        result_add = result_add + '\nText:\n'+ " ".join([normalize_text(result_n[0])]) +'\n'

    prompt_i = 'Summarize the content given the text provided.'+" ".join([result_add])+'\nSummary:\n'

    # Use Azure OpenAI service to process query

    response = openai.Completion.create(
        engine= deployment_id,
        prompt = prompt_i,
        temperature = 0,
        max_tokens = 1000,
        top_p = 1.0,
        best_of = 1
    )

    # Filter response to show only relevant output
    output = response['choices'][0]['text']
    source_file = search_df._get_value(0,'metadata_storage_path')
    source_id = search_df._get_value(0,'id')
    # Render output page with back button
    return render_template('output.html', output=output, source_file=source_file, source_id=source_id)

@app.route('/back', methods=['POST'])
def back():
    # Go back to input form
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
