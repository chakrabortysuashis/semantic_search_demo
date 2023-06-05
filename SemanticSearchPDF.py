import PyPDF2
from PyPDF2 import PdfReader
import openai
import re
import tiktoken
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np
from itertools import islice
import pickle
import pandas as pd

openai.api_type = "azure"
openai.api_base = "https://openai-rq75033025.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key="9fd911d1810644a5b0bd4e9c75d9e8a0"
dict={}

decoded_chunk_data=[]
chunk_data=[]
chunk_embeddings = []
chunk_lens = []
    

embedding_model = 'iCETS-MSCoE-text-embedding-ada-002'
embedding_length = 2000
embedding_encoding = 'cl100k_base'
tokenizer = tiktoken.get_encoding(embedding_encoding)
embedding_cache_path = "data/recommendations_embeddings_cache.pkl"

def read_or_new_pickle(path, default):
    try:
        with open(path, "rb") as f:
            foo = pickle.load(f)
    except Exception:
        foo = default
        with open(path, "wb") as f:
            pickle.dump(foo, f)
    return foo

def read_pdf_data(pdf):
    pdfData=""
    pdfReader=PyPDF2.PdfReader(pdf)
    count=len(pdfReader.pages)
    for i in range(count):
        pageObj= pdfReader.pages[i].extract_text()
        pdfData=pdfData+""+pageObj
    print(pdfData)
    return pdfData


def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s


def callDavinciModel(response,query):
    print("calling davinci model")
    deployment_name='iCETS-MSCoE-text-davinci-003'
    # prompt=response+" Provide answer to the query "+query+" only from provided details in exact words provided.Response should match exactly with provided data"
    # prompt="Context: "+response+"/n query: "+query+"/n role: You are a person who read and understand the context provided to you and also provide answer to the query, based on your anlysis to the context"+" Note- If you cannot find any answer to the query in the context then return 'The query you are looking for is not provided in the document'"
    prompt=" You need to answer the question based on the context below . Respond 'Unsure about answer' if not sure about the answer \n"+"Context: " +response+"\n Question: "+query+"\n Answer:"
    print("prompt:",prompt)
    encoded_data = tokenizer.encode(prompt)
    print("length is",len(encoded_data))
    if(len(encoded_data)>3000):
        return "Query length exceeded, Please provide valid query"
    response=openai.Completion.create(engine=deployment_name,temperature=0,prompt=prompt,max_tokens=1000)
    text=response['choices'][0]['text'].replace('/n','').replace(" .","").strip()
    print("text",text)
    return text

def total_token_count(data):
    
    encoded_data = tokenizer.encode(data)
    print(len(encoded_data))
    return len(encoded_data)

def batched(iterable, n):
    print("inside batched")
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    
    while (batch := tuple(islice(it, n))):
        yield batch

def get_embeddings(text_or_tokens, model=embedding_model):
    print("insisde get_embeddings")
    return openai.Embedding.create(input=text_or_tokens, engine=model)["data"][0]["embedding"]

def chunked_tokens(text, encoding_name, chunk_length):
    print("inside chunked tokens")
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    
    yield from chunks_iterator

def get_embeddingvectors_list(text, model=embedding_model, max_tokens=embedding_length, encoding_name=embedding_encoding):
    
    decoded_string=""
    for chunk in chunked_tokens(text, encoding_name=encoding_name, chunk_length=max_tokens):
        chunk_embeddings.append(get_embeddings(chunk, model=model))
        decode = tokenizer.decode_tokens_bytes(chunk)
        for i in decode:
             decoded_string=decoded_string+str(i,'latin1')+""
           
        chunk_lens.append(len(chunk))
        print("chunk is",decoded_string)
        decoded_chunk_data.append(decoded_string)
        print("list length",len(decoded_chunk_data))
        decoded_string=""
        print(chunk_lens)
    
    return chunk_embeddings



#search through the pdf
def search_docs( user_query,pdf):
    similarity=[]

    #Read pdf data
    pdfData=read_pdf_data(pdf)

    #Normalize extracted pdf data
    pdfData= normalize_text(pdfData)

    #Get total token count for pdf
    tokenCount=total_token_count(pdfData)

    print("Total token count",tokenCount)

    #Get embedding vectors for user provided query
    embedding = get_embeddings(user_query)

    data=read_or_new_pickle("SaveEmbeddings.pkl", default=0)

     #Read dictionary from pickle
    with open("SaveEmbeddings.pkl",'rb') as fp:
        pickle_data = pickle.load(fp)
       
        if(pickle_data==0):
             #save dictionary to pickle file
            with open("SaveEmbeddings.pkl",'wb') as fp:
                
                 #Get embedding vectors list for given pdf
                chunks_embedding_vectors = get_embeddingvectors_list(pdfData)
                
                dict={ decoded_chunk_data[i]:chunk_embeddings[i] for i in range(len(chunks_embedding_vectors))}
                print("dictionary",dict)
                chunk_data=decoded_chunk_data
                
                pickle.dump(dict,fp)
                
        else:
           

            chunks_embedding_vectors=pickle_data.values()
            chunk_data=list(pickle_data.keys())
            
            

        
        
    

    print(len(chunks_embedding_vectors))

 
    #Similarity check between user query vector and vectors of pdf
    for i in chunks_embedding_vectors:
        similarity.append(cosine_similarity(i, embedding))
    print("similarity:",similarity)
    
    maxSimilarity=max(similarity)
    print("max",maxSimilarity)
    indexValue=similarity.index(maxSimilarity)
    print("index",indexValue)
    finalized_chunk=chunk_data[indexValue]
    print(finalized_chunk)
    searchResponse=callDavinciModel(finalized_chunk,user_query)
    return searchResponse
    

#res = search_docs( "I travelled in a bike and I got injured in an accident. will my insurance cover the medical costs?","SemanticSearchSample.pdf")









