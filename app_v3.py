import logging
import sys
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.core import set_global_tokenizer
from llama_index.core import Settings

# from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.ollama import Ollama

# from flask import Flask, request, jsonify
import streamlit as st
from streamlit_option_menu import option_menu

# import os
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load documents and setup embedding model
documents = SimpleDirectoryReader("Data/").load_data()


from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 1024

from llama_index.core import PromptTemplate

system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

# llm = Ollama(
#     model="mistral",
#     system_prompt=system_prompt,
#     query_wrapper_prompt=query_wrapper_prompt,
#     request_timeout=300.0,
# )

# import torch
# from llama_index.llms.huggingface import HuggingFaceLLM
# llm = HuggingFaceLLM(
#     context_window=8192,
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.7, "do_sample": False},
#     system_prompt=system_prompt,
#     query_wrapper_prompt=query_wrapper_prompt,
#     tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
#     model_name="mistralai/Mistral-7B-Instruct-v0.2",
#     device_map="auto",
#     # stopping_ids=[50278, 50279, 50277, 1, 0],
#     tokenizer_kwargs={"max_length": 4096},
#     # uncomment this if using CUDA to reduce memory usage
#     # model_kwargs={"torch_dtype": torch.float16}
# )

# Initialize LlamaCPP with CPU settings
llm = LlamaCPP(
    # model_url="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    model_path="Model\mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.1,
    max_new_tokens=2048,
    context_window=8192,
    generate_kwargs={},
    model_kwargs={},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

Settings.llm = llm
Settings.chunk_size = 1024


# chroma_client = chromadb.EphemeralClient()
# chroma_collection = chroma_client.create_collection("quickstart1")
# # set up ChromaVectorStore and load in data
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(
#     documents, llm=llm, storage_context=storage_context, embed_model=embed_model
# )


db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("all")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    llm=llm,
    storage_context=storage_context,
    embed_model=embed_model,
)

# # load from disk
# db2 = chromadb.PersistentClient(path="./chroma_db")
# chroma_collection = db2.get_or_create_collection("quickstart1")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# index = VectorStoreIndex.from_vector_store(
#     vector_store,
#     embed_model=embed_model,
# )


# index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=20)


# # Flask app setup
# app = Flask(__name__)

# @app.route("/chatbot", methods=["POST"])
# def chatbot_response():
#     try:
#         data = request.json
#         question = data["question"]
#         chatbot_response = query_engine.query(question)
#         response_text = str(chatbot_response)

#         return jsonify({
#             "response": response_text
#         })
#     except Exception as e:
#         logging.error(f"Error processing chatbot response: {e}")
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(port=5000)
# Streamlit UI
st.title("Chatbot with Streamlit")

question = st.text_input("Ask a question:")

if st.button("Submit"):
    try:
        chatbot_response = query_engine.query(question)
        response_text = str(chatbot_response)
        st.write(response_text)
    except Exception as e:
        st.error(f"Error processing chatbot response: {e}")
