# AI-Powered-Customer-Support-System

Project Overview:

The AI-Powered Customer Support System is a machine learning and NLP-based project that automatically understands and responds to customer queries. It analyzes customer messages, identifies the query intent, detects sentiment, and provides appropriate responses to assist users efficiently.
The system is designed to simulate an intelligent customer support assistant capable of identifying customer intent, detecting sentiment, retrieving relevant knowledge, and generating appropriate responses.
The project combines transformer models, vector search, and generative AI to create a complete AI support pipeline. It helps automate customer service tasks and improve response efficiency by providing quick and relevant answers to user queries.

Dataset:
original dataset - 50000
sample dataset - 25000 (selected radomly)

columns :

query – Customer message
intent – Type of customer request
sentiment – Emotional state of the customer
country
product_group
plan
customer_type
priority
amount_billed
satisfaction_score
*The query column was renamed to text for model training.*

Data Preprocessing:
--> steps :
Random sampling of 25,000 records
Selection of relevant columns
Renaming query to text
Filtering important intent categories
Filtering selected sentiment categories
Cleaning text data.

Exploratory Data Analysis (EDA):
Exploratory data analysis was performed to understand the dataset structure and patterns.

Visualizations Created --> Sentiment Distribution, Intent Distribution, Intent vs Sentiment, Customer Message Length Distribution

Intent Classification Model:

Steps-
Label encoding of intent categories
Train-test split (80% training, 20% testing)
Tokenization using HuggingFace tokenizer
Conversion to HuggingFace dataset format
Fine-tuning using the Trainer API.

Model Evaluation:
The intent classification model was evaluated using several performance metrics.

Sentiment Analysis Model:
The project also trains a second transformer model to detect the customer's emotional state.

Vector Database using FAISS:
To provide intelligent responses, the project uses FAISS (Facebook AI Similarity Search).
FAISS is used as a vector database that stores embeddings of support documents.

Retrieval-Augmented Generation (RAG):
he project implements a RAG (Retrieval-Augmented Generation) architecture.

Response Generation:
The system uses a text generation model to produce responses.
model used :  GPT -2.

AI Copilot System:
combain with multiple AI component --> Intent Classification,Sentiment Detection,Knowledge Retrieval,Response Generation

Web Application (Streamlit):
includes a Streamlit web interface that allows users to interact with the AI support assistant.

Conclusion :

This project demonstrates how modern NLP techniques and generative AI can be integrated to build an intelligent automated customer support system. By combining transformer models, vector search, and RAG architecture, the system can understand customer queries, retrieve relevant knowledge, and generate helpful responses in real time.
