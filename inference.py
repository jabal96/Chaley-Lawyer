from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
import streamlit as st
import ollama
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from summary import initialize_client
from summary import pull_embedding_model
import time
def semantic_search(query_text,model, index_name, k=5):
    # Embed the query — must match indexing pipeline (mean pooling + L2 norm
    # is handled internally by sentence-transformers' .encode())
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()

    query = {
        "size": k,
        "query": {
            "knn": {
                "embeddings": {          # <-- change to your actual vector field name
                    "vector": query_vector,
                    "k": k
                }
            }
        }
    }

    response = client.search(index=index_name, body=query)
    return response["hits"]["hits"]

def keyword_search(query_text, index_name, k=5):
    query = {
        "size": k,
        "query": {
            "match": {
                "chunks": {              # <-- your text field name
                    "query": query_text
                }
            }
        }
    }
    response = client.search(index=index_name, body=query)
    return response["hits"]["hits"]



def hybrid_search(query_text, index_name, k=5):
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()

    query = {
        "size": k,
        # "explain": True,
         "_source": {"exclude": ["embeddings"]}, 
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "knn": {
                            "embeddings": {
                                "vector": query_vector,
                                "k": k
                            }
                        }
                    },
                    {
                        "match": {
                            "chunks": {
                                "query": query_text
                                #"boost": 0.3  # weight keyword match lower than semantic
                            }
                        }
                    }
                ]
            }
        }
    }
    response = client.search(index=index_name, body=query, search_pipeline="nlp-search-pipeline")
    return response["hits"]["hits"]
def prompt_generation(hits:list,query : str)->str:
    context = ""
    for hit in hits:
        context += (f"{hit["_index"]} \n {hit["_source"].get("chunk_text")}\n")

    prompt = (" You are my personal Legal assistant.any question i ask you i want you to use the context provided to answer the question. If you think the answer to" \
                "the question asked can not be found in the provided context, say you dont know.In your response cite the document And do not Halucinate. \n CONTEXT:\n"
                     + context + "\nquestion: " + query)
    
    return prompt




def timed(label, fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    st.sidebar.write(f"⏱️ {label}: {elapsed:.2f}s")
    return result



def generate_response(prompt:str,model_name ="gemma4:31b-cloud" ): #"llama3.2:3b"):
            try:
                stream = ollama.chat(
                model=model_name,
                messages=[{"role":"user","content":prompt}],
                stream=True)
            except ollama.RequestError as e:
                 print(f" error during stream {e} ")  
                 raise
            for respond in stream:
                yield respond["message"]["content"]

response2 = ''
if __name__ == "__main__":
    try:
     client = timed("openSearch_connection", initialize_client)
    except ConnectionError as e:
         print(f"could not reach opensearch. make sure open search is running.....{e}")
         raise
      
    try:
        embedding_model = timed("pull embedding model",pull_embedding_model, "sentence-transformers/all-mpnet-base-v2")

    except Exception as e:
         print(f"embedding model not found. please the spelling of the model...{e}")
         raise 

    st.title("Personal legal Asistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []   

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
                st.write(message["content"])

        
    if question := st.chat_input("Ask something..."):
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            results = timed("search_data_base" ,semantic_search,query_text=question, model=embedding_model, index_name="the_1992_constitution_of_ghana")
            prompt = prompt_generation(results,question)
            stream = generate_response(prompt=prompt)
    
            response2 = timed("response_generation",st.write_stream, stream) 
    st.session_state.messages.append({"role": "assistant", "content": response2})