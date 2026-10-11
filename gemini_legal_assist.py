import os
import requests
import requests
import streamlit as st
import ollama
from google import genai
from dotenv import load_dotenv

from google.genai import types
load_dotenv()
api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

token = os.environ["LAWS_AFRICA_API_KEY"]
KB_CODE = "legislation-gh"

model_name = "gemma4:31b-cloud"#""#"nemotron-3-super:cloud"#"llama3.2:3b" 

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
                # f"id:{metadata.get('portion_id')}"
            ]
        )
    )
    print("\n\n".join(context))
    return "\n".join(context)
    

def prompt_template(context:str,question:str)->str:
    context_string = "\n".join(context)
    prompt = (f"CONTEXT: \n{context}\nUser question: {question}")
    return prompt

def query_rewrite(query:str)->str:
    regenerate_query_system_prompt = """ You are a legal search query optimizer for a knowledge base of Ghanaian
legislation, queried through the Laws.Africa Knowledge Base API. Your job
is to convert a user's question into a refined search string that
maximizes retrieval accuracy for both semantic and keyword matching.

CRITICAL RULE: The search API performs poorly on vague conversational
phrasing and performs well when the search text includes specific legal
domain keywords, the likely Act title, or the relevant section topic.
Your output must reflect this — do not simply paraphrase the user's
question in slightly more formal words.

Follow this process:

1. IDENTIFY THE LEGAL DOMAIN
   Determine which area of law the question falls under (e.g. labour,
   land, family, criminal, commercial, tax, property, constitutional).

2. RECALL THE LIKELY GOVERNING ACT
   Based on your knowledge of Ghanaian law, identify the specific Act
   (with year) that most likely governs this issue — e.g. "Labour Act,
   2003 (Act 651)", "Intestate Succession Act, 1985 (PNDCL 111)",
   "Matrimonial Causes Act, 1971 (Act 367)", "Land Act, 2020 (Act 1036)".

   IMPORTANT: Only name an Act if you are reasonably confident it exists
   and is correct. Never invent an Act title, year, or number. If you are
   not confident which specific Act applies, omit the Act title entirely
   and rely on domain keywords and section topics instead — a wrong Act
   name actively hurts retrieval by pulling the search toward the wrong
   document.

3. IDENTIFY THE LIKELY SECTION TOPIC
   Name the specific provision or topic within that Act the question
   would fall under (e.g. "termination of employment", "redundancy
   compensation", "distribution of intestate estate", "grounds for
   divorce", "encroachment on land").

4. CONSTRUCT THE REFINED SEARCH TEXT
   Combine these into a single search string, ordered from most to
   least specific:
   [Act title if confident] + [section topic / legal concept] + [supporting keywords]

   Rules for the output string:
   - No conversational phrasing, no questions, no filler words
   - Maximum ~20 words
   - Use exact legal terminology, not everyday synonyms
   - Do not include the user's personal narrative details (names,
     specific dates, emotional framing)

Output ONLY the refined search string. No explanation, no JSON, no
quotation marks around it.
"""
    

    response = client.models.generate_content(
    model="gemini-3.6-flash",
    config=types.GenerateContentConfig(
        system_instruction=regenerate_query_system_prompt
    ),
    contents=query
    )

    return response.text


     
# crequest_context("what crime can lead to death sentence")
response2 = ""
with open("system.txt","r") as f:
     system_prompt = f.read()
# print(system_prompt)     
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
            query_text = query_rewrite(question)
            print(query_text)
            context = request_context(query_text)
            # print(context)
            prompt=prompt_template(context,question)
            # print(prompt)

            
            def response_stream():
                response_stream = client.models.generate_content_stream(
                                model="gemini-3.6-flash",
                                 config=types.GenerateContentConfig(
                                       system_instruction= system_prompt
                                   ),
                                   contents=prompt
                                   )    
                # print(response_stream)
                for respond in response_stream:
                    yield respond.text
            
            response2 = st.write_stream(response_stream())
st.session_state.messages.append({"role": "assistant", "content": response2})