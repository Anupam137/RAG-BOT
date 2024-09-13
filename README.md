# RAG-BOT
# Chat Application with RAG + Gen AI

## Overview

This project demonstrates a chat application that utilizes Retrieval-Augmented Generation (RAG) with two separate bots: one implemented with Pinecone and the other with FAISS. The application integrates OpenAI's GPT models to handle communication between the user and the bots.

## Components

1. **Pinecone Bot**: A bot powered by Pinecone for vector-based retrieval and similarity search.
2. **FAISS Bot**: A bot utilizing FAISS for efficient similarity search in large-scale datasets.
3. **OpenAI**: GPT-4 models are used for generating responses based on the retrieved context.

## Features

- **Two Separate Bots**: Each bot is responsible for handling different aspects of the chat.
- **Contextual Responses**: Using RAG, the bots leverage external knowledge to provide more accurate and contextually relevant responses.
- **Integration with OpenAI**: GPT-4 is used for generating human-like responses based on the retrieved context.

## Setup

### Prerequisites

- Python 3.7 or later
- API keys for OpenAI, Pinecone, and FAISS (if using a cloud-based version)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/chat-app-rag-gen-ai.git
   cd chat-app-rag-gen-ai

2. Environment
python -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`

3. Set up your API Keys:
   OPENAI_API_KEY=your-openai-api-key
    PINECONE_API_KEY=your-pinecone-api-key


