import os
import openai
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Initialize OpenAI client using the new method
from openai import OpenAI
client = OpenAI(
    api_key='replace with your actual key',  
)

# API keys and environment variables
OPENAI_API_KEY = 'replace with your actual key'
PINECONE_API_KEY = 'replace with your actual key'
PINECONE_ENVIRONMENT = 'us-east-1'
PINECONE_INDEX_NAME = 'chat-history'
MAX_TOKENS = 225

# Initialize OpenAI client
client.api_key = OPENAI_API_KEY

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists
existing_indexes = pinecone_client.list_indexes()
if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"Creating index '{PINECONE_INDEX_NAME}'...")
    pinecone_client.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # Adjust dimension based on model
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
    )
else:
    print(f"Index '{PINECONE_INDEX_NAME}' already exists, connecting to the index.")

# Connect to the index
index = pinecone_client.Index(PINECONE_INDEX_NAME)

# Cell 3: Define the conversation history
history = [
    "1: User: Hi there! How are you doing today? | Bot: Hello! I'm doing great, thank you! How can I assist you today?",
    # Additional conversation data...
]

# Function to get embeddings using the new OpenAI API
def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

# Cell 4: Define the function to add embeddings to Pinecone
def add_embeddings_to_pinecone(history, index_name='chat-history'):
    """
    This function:
    1. Encodes each message in the history using OpenAI.
    2. Upserts the encoded messages into the Pinecone index.
    """
    # Connect to the index
    index = pinecone_client.Index(index_name)

    # Loop to encode the history and upsert into Pinecone
    for idx, message in enumerate(history):
        embedding = get_embedding(message)
        index.upsert([(str(idx), embedding)])

# Cell 5: Define the RAG mechanism to retrieve relevant history
def retrieve_relevant_history(query, index_name='chat-history'):
    """
    This function:
    1. Encodes the query using OpenAI's embedding model.
    2. Queries Pinecone to retrieve the most relevant historical messages.
    """
    index = pinecone_client.Index(index_name)
    query_embedding = get_embedding(query)
    result = index.query(queries=[query_embedding], top_k=5)  # Retrieves top 5 relevant messages
    return [match['id'] for match in result['matches']]

# Cell 6: Function to prepare the prompt
def prepare_prompt(test_prompt, history, index_name='chat-history'):
    """
    This function:
    1. Retrieves relevant history messages using the RAG mechanism.
    2. Combines the retrieved messages with the test prompt.
    3. Ensures the combined prompt does not exceed the token limit.
    """
    relevant_message_ids = retrieve_relevant_history(test_prompt, index_name)
    relevant_history = [history[int(msg_id)] for msg_id in relevant_message_ids]
    
    combined_prompt = test_prompt + "\n\n" + "\n".join(relevant_history)
    if len(combined_prompt.split()) > MAX_TOKENS:
        combined_prompt = " ".join(combined_prompt.split()[:MAX_TOKENS])
    return combined_prompt, relevant_history

# Cell 7: Function to test the final prompt
def test_final_prompt():
    """
    This function:
    1. Defines the final test prompt.
    2. Prepares the prompt using the prepare_prompt function.
    3. Calls OpenAI to generate a response.
    4. Prints the final test prompt, context, and response.
    """
    final_test_prompt = "Do you think it will help me stay fit?"
    prepared_prompt, context_referred = prepare_prompt(final_test_prompt, history)
    
    # Call OpenAI API to generate a response
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        prompt=prepared_prompt,
        max_tokens=100
    )
    
    print(f"Final Test Prompt: {final_test_prompt}")
    print(f"Context Referred: {context_referred}")
    print(f"Final Test Prompt Response: {response.choices[0].text.strip()}")

# Call the test function to generate the Final Test Prompt Response
test_final_prompt()
