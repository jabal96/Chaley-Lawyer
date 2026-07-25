from PyPDF2 import PdfReader
import re
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
import streamlit as st
import ollama
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

st.cache_resource()
def initialize_client()->OpenSearch:
    client = OpenSearch(
        hosts= [{"host": "localhost", "port":"9200"}])
    
    return client

def load_pdf_documentation(file_path: str) -> str:
    """
    Load text from a multi-page PDF document.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: All extracted text from the PDF.
    """
    text_pages = []
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)

    return "\n\n".join(text_pages)

def clean_text(text:str)->str:
    text_list = text.splitlines()
    cleaned_text = "\n".join(text_list[119:-(6538-531)])
    return cleaned_text


def chunk_text(text:str,article_chapter:tuple) -> list:
    
    pattern = r'(?=^\d+\.\s*$)'
    #pattern = r'^\d+\.\s*$'
    chunks = re.split(pattern,text,flags=re.MULTILINE)
    clean_chunks : list[str] = []
    for item in chunks:
       
        clean_chunks.append( item.replace("\n",""))

    chunks = clean_chunks[:300][1:]  
    context_chunks = [] 
    
    for chunk, article, chapter in zip(chunks,article_chapter[0],article_chapter[1]):
        context_chunks.append((f"{chapter},{article}:{chunk}"))  
    return context_chunks

def get_chapter_article_name(constitution_text:str):
    cons_list = constitution_text.splitlines()
    table_of_content = cons_list[117:-(6741-541)]
    table_of_content=[item for item in table_of_content if item.strip()]
    
    for line in table_of_content:
        if line[:1].isdigit() or line.isupper():
            pass
        else:
            
            table_of_content.remove(line)

    return table_of_content

def article_chapter_list(chapter_article_list:list):
    chapter = ""
    reset = False
    article_name = []
    chapter_name =[]
    for item  in chapter_article_list:

        if item.isupper():
            if reset:
                chapter = ""
                reset = False
            chapter += item 
        
        else:
            article_name.append(item)
            chapter_name.append(chapter)
            reset = True
    article_name =[ f"Article{item}" for item in article_name]
    article_id = [f"Article_{id+1}" for id in range(len(article_name))]
    return (article_name,chapter_name,article_id)


st.cache_resource()
def pull_embedding_model(embedding_model_name: str) -> SentenceTransformer:
    return SentenceTransformer(embedding_model_name)
    



def embed_text_chunks(text_chunks, embedding_model) -> list:

    """
    Generates embeddings for a list of text chunks using the specified embedding model.

    Args:
        text_chunks (list): A list of text chunks to be embedded.
        embedding_model: The embedding model to use for generating embeddings.

    Returns:
        list: A list of embeddings corresponding to the input text chunks.
    """
    embeddings = []
    for chunk in text_chunks:
        embedding = embedding_model.encode(chunk).tolist()  # Convert tensor to list of floats
        embeddings.append(embedding)
    return embeddings


def convert_to_dic(chunks: list,embeddings :list, article_chapter_id)->dict:
    all_in_one = [
        {
            "ids": article_chapter_id[2],
            "embeddings": embeddings,
            "chunk_text" : chunks,
            "chapter" : article_chapter_id[1] ,
            "article_name":article_chapter_id[0]
        }
    ]
    return all_in_one

def create_VectorDB(client):
    

    index_config= {
        "settings":{
        "index" : {
            "knn": True
            }
            },
            
                    
             "mappings" : {
                 

             "properties": {
                 "embeddings": {
            "type":"knn_vector",
        "dimension": 768,
            "method": {
        "name":"hnsw",
        "engine" :"faiss",
        "space_type":"cosinesimil"}},
        "chunk_text":{"type":"text"},
        "article_name":{"type":"text"},
        "chapter":{"type":"keyword"}

           }
         }         

    }
    client.indices.create(index="the_1992_constitution_of_ghana",body= index_config)

    

def injest_vdb(data_dic:list[dict]):
    for dic in data_dic:
        for i,chunk in enumerate(dic["chunk_text"]):
            yield{
                "_index":"the_1992_constitution_of_ghana",
                "_id" : dic["ids"][i],
                "_source":{
                    
                    "embeddings":dic["embeddings"][i],
                    "chunk_text":chunk ,  
                    "article_name": dic["article_name"][i],
                    "chapter": dic["chapter"][i]
                           }
            }




if __name__ == "__main__":
    client = initialize_client()
    client.indices.delete(index="the_1992_constitution_of_ghana")
    
    if client.indices.exists(index="the_1992_constitution_of_ghana") == False:

        #pages = load_pdf_documentation(r"C:\Users\lenovo\Desktop\build_your_local_RAG_system\The_1992_Constitution.pdf")
        #with open("constitution.txt", "w", encoding="utf-8")as f:
        #    f.write(pages)
        #print(pages[:10000])

        #cln_text = clean_text(pages)
        with open("constitution.txt","r") as f:
            text = f.read()

        with open("clean_constitution.txt","r") as f:
            clean_chunk = f.read()   

        article_chapter_string = get_chapter_article_name(text)
        article_chapter_id = article_chapter_list(article_chapter_string)
        chnk = chunk_text(clean_chunk,article_chapter_id)
        
        model = pull_embedding_model("sentence-transformers/all-mpnet-base-v2")
        embeddings = embed_text_chunks(chnk, model)
        data_dic = convert_to_dic(chnk,embeddings,article_chapter_id)
        create_VectorDB(client=client)
        success, error = bulk(client,injest_vdb(data_dic), stats_only=False)
        print(success)
        print(error)
    #print(error)
    #print(client.count(index= "ghana_constitution"))
    #item_1 = client.get(index="ghana_constitution",id = "article_1")
    #print(item_1)
    #model = pull_embedding_model("sentence-transformers/all-mpnet-base-v2")
    
    #prompt_generatcion(results)
    #print(results)
    # print(results)
  
    # for hit in results:
    #     # print(hit.keys())
    #     print(hit["_id"])
    #     print(f"Score: {hit['_score']:.4f}")
    #     print(f"Article: {hit['_source'].get('Article_num')}")
    #     print(f"Text: {hit['_source'].get('chunks')}...")
    #     print("---")
        #print(hit["_explanation"]) 
        #print(results)   
    