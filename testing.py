import os
import requests
import requests
import streamlit as st
import ollama
token = os.environ["LAWS_AFRICA_API_KEY"]
KB_CODE = "legislation-gh"


token = os.environ["LAWS_AFRICA_API_KEY"]
print(token)
def request_context(question:str):
   
# print(response.json())

    response = requests.post(
    f"https://api.laws.africa/ai/v1/knowledge-bases/{KB_CODE}/retrieve",
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    json={
        "text": question,
        "top_k": 5,
        "filters": {
             "principal": True,
             "repealed": False,
            "frbr_place": "gh",
        },
    },
    timeout=30,
    )
    response.raise_for_status()


# print(response.json())

    results = response.json()["results"]

    context = []
    for result in results:
        metadata = result["metadata"]
        context.append(
            "\n\n".join(
            [
                f"Title: {metadata.get('title')}",
                f"Source: {metadata.get('portion_public_url') or metadata.get('public_url')}",
                f"Text: {result['content']['text']}",
                f"id:{metadata.get('portion_id')}"
            ]
        )
    )
    print("\n\n".join(context))
    return "\n".join(context)
    

def prompt_template(context:str,question:str,sys_prompt)->str:
    context_string = "\n".join(context)
    prompt = (f"system prompt: {system_prompt} \n CONTEXT: \n{context}\nUser question: {question}")
    return prompt

# crequest_context("what crime can lead to death sentence")
response2 = ""
with open("system_prompt.text","r") as f:
     system_prompt = f.read()
# print(system_prompt)     
st.title("Personal legal Asistant")
if "messages" not in st.session_state:
        st.session_state.messages = []   

for message in st.session_state.messages:

        with st.chat_message(message["role"]):
                st.write(message["content"])

model_name = "gemma4:31b-cloud"#""#"nemotron-3-super:cloud"#"llama3.2:3b"       
if question := st.chat_input("Ask something..."):
        
        with st.chat_message("user"):

            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            context = request_context(question)
            prompt=prompt_template(context,question,system_prompt)
            # print(prompt)
            try:
                stream = ollama.chat(
                model=model_name,
                messages=[
                    #  {"role":"system","content": system_prompt},
                          


                          {"role":"user","content":prompt}],
                stream=True)
            except ollama.RequestError as e:
                 print(f" error during stream {e} ")  
                 raise
            def response_stream(stream): 
                for respond in stream:
                    yield respond["message"]["content"]
            streamr = response_stream(stream)
            response2 = st.write_stream(streamr)
st.session_state.messages.append({"role": "assistant", "content": response2})