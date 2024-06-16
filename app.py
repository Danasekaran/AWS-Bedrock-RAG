# import langchain_aws
# import json
import os
import boto3
import streamlit as st

# For Embedding Model
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_aws import BedrockLLM

# for Data
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

# vector embeding and vector store
from langchain_community.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator


# LLM
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
# from langchain.chains import RetrivalQA


os.environ['aws_access_key_id']='your_aws_access_key_id'
os.environ['aws_secret_access_key']='your_aws_secret_access_key'
os.environ['region_name']='your_region_name'

# bedrock client
bedrock=boto3.client("bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock,region_name='your_region_name',credentials_profile_name="default")

from langchain.chains.combine_documents import create_stuff_documents_chain

# data ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("source_data")
    documents=loader.load()
    text_split=RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap  =100)
    doc_split=text_split.split_documents(documents)
    return doc_split

def get_vector_store(splitted_docs):
    vector_store=Chroma.from_documents(splitted_docs,bedrock_embeddings,persist_directory="vector_data_persist")
    vector_store.persist()

def vector_retriever(query): 
    vectordb = Chroma(persist_directory="vector_data_persist", embedding_function=bedrock_embeddings)
    retriever = vectordb.as_retriever(search_type='similarity',search_kwargs={"k":2})
    return retriever

def get_llm():
    llm=Bedrock(model_id="amazon.titan-text-lite-v1",client=bedrock)
    return llm

#Create the retrieval chain
template = """
    You are an expert AI.
    you will Answer based on the context provided. 
    context: {context}
    input: {input}
    answer:
    """
prompt_template = PromptTemplate.from_template(template)

def get_response_from_llm(llm,vector_store,prompt_template,pass_prompt):
    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template) 
    retrieval_chain = create_retrieval_chain(vector_store, combine_docs_chain)
    response=retrieval_chain.invoke({"input":pass_prompt})
    return response["answer"]



