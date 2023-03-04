# Document Summarization using Azure Cognitive Search and OpenAI

This code demonstrates a Flask-based web application for document summarization. It takes in a user query and searches for relevant documents using Azure Cognitive Search, and then summarizes the content using OpenAI. 

## Requirements

The following packages are required to run this application:
- os
- requests
- json
- openai
- re
- pandas
- azure.search.documents
- azure.identity
- dotenv
- Flask

## Configuration

The following environment variables must be set in a .env file or in the environment:
- OPENAI_API_TYPE
- OPENAI_API_VERSION
- OPENAI_API_BASE
- OPENAI_API_KEY


In addition, the `TEXT_SEARCH_DOC_EMBEDDING_ENGINE`, `TEXT_SEARCH_QUERY_EMBEDDING_ENGINE`, and `TEXT_DAVINCI_001` variables should be set according to the specific engine IDs in your OpenAI account.

The Azure Cognitive Search endpoint and index name should be set in the `process_query` function.

## Usage

To run the Flask app, execute the following command:
'python app.py'

Then, navigate to `http://localhost:PORT/` in your web browser to access the application.

## Functionality

The application presents the user with a search bar where they can input their query. After submitting the query, the application searches for documents in the Azure Cognitive Search index that contain the keyword extracted from the query using OpenAI. The application then summarizes the top two chunks of text from the document that are most similar to the user's query, using OpenAI's text summarization capabilities.

The summarized text is presented to the user, along with a link to the original document and its unique ID. The user can click the "Back" button to return to the search bar and enter a new query.

# Code Description

The code imports the necessary packages, sets the environment variables, and defines several helper functions. The `splitter` function splits text into chunks of n sentences, while `normalize_text` performs light data cleaning on text.

The `search_docs` function takes a DataFrame of chunks of text and a user query, and returns the top n chunks that are most similar to the user's query using cosine similarity.

The Flask app is set up with two routes: the main page and the processing page. The main page presents the user with a search bar, while the processing page processes the user's query and displays the summarized text.

The `process_query` function extracts the keyword from the user's query using OpenAI, searches for documents containing the keyword using Azure Cognitive Search, and summarizes the content of the top two chunks most similar to the user's query using OpenAI.

The `back` function returns the user to the main page.

## Conclusion

This code demonstrates how to use Azure Cognitive Search and OpenAI to summarize documents based on user queries. With some modifications, it can be adapted to different use cases and datasets.

# Installation

1. Clone the repository:
`git clone https://github.com/lordaouy/aoai-flask-webapp.git`

2. Change into the directory:
`cd repository`

3. Install the required packages:
`pip install -r requirements.txt`


4.  Set up the environment variables:

Create a `.env` file in the root directory of the repository with the following contents:

- OPENAI_API_TYPE=your_api_type
- OPENAI_API_VERSION=your_api_version
- OPENAI_API_BASE=your_api_base
- OPENAI_API_KEY=your_api_key


Replace `your_api_type`, `your_api_version`, `your_api_base`, and `your_api_key` with your own values.

5.  Set up Azure Cognitive Search:
Set up an Azure Cognitive Search service and create an index with the necessary fields for your data. Set the endpoint and index name in the `process_query` function.

6.  Set up OpenAI:

Create an OpenAI account and set up the necessary engines. Set the engine IDs in the `process_query` function.

7.  Run the application:
`python app.py`

8.  Navigate to `http://localhost:PORT/` in your web browser to access the application.

All of this use gpt3 to generate

