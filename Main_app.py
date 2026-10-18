import os
import requests
import requests
import streamlit as st
import ollama
from google import genai
from dotenv import load_dotenv
from openai import OpenAI
from google.genai import types
load_dotenv()
deepseek_model_name = "deepseek-flash"

token = os.environ["LAWS_AFRICA_API_KEY"]
KB_CODE_legis = "legislation-gh"
KB_CODE_judg = "judgments-gh"

print(token)
def request_legislation_context(question:str,kb_code:str):
   
# print(response.json())

    response = requests.post(
    f"https://api.laws.africa/ai/v1/knowledge-bases/{kb_code}/retrieve",
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
        # print(results)
    #print("\n\n".join(context))
    return context



def request_judgment_context(question:str,kb_code:str):
   
# print(response.json())

    response = requests.post(
    f"https://api.laws.africa/ai/v1/knowledge-bases/{kb_code}/retrieve",
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    json={
        "text": question,
        "top_k": 3,
        "filters": {
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
        # print(results)
    #print("\n\n".join(context))
    return context

    

def prompt_template(legistation_chunks:str,question:str,judgement_chunks,history:list[dict])->str:

    legis_chunk_strings = "\n".join(legistation_chunks)
    judge_chunk_strings = "\n".join(judgement_chunks)
    prompt = (f"CONTEXT-LEGISLATION LEGAL CHUNKS: \n{legis_chunk_strings}\nCONTEXT-JUDGMENT CASE LAW CHUNKS: \n{judge_chunk_strings}\nUser question: {question}\n")



    if len(history) != 0:
        max_history_messages = 10
        history = history[-max_history_messages:]
        prompt += "Conversation History:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            prompt += f"{role}: {content}\n"
        prompt += "\n"

    return prompt


def deepseek_chat(user_prompt:str,system_prompt:str):

    # api_key = os.environ["deepseek_API_KEY"]
    # client = genai.Client(api_key=os.environ["deepseek _API_KEY"])
    # response = client.models.generate_content(
    # model=deepseek _model_name,
    # config=types.GenerateContentConfig(
    #     system_instruction=system_prompt
    # ),
    # contents=f"question : {user_prompt}"
    # )        
    # return response
    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=deepseek_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"question : {user_prompt}"},
        ],
        stream=False,
    )
    return response



def deepseek_response_stream(main_user_prompt, main_sys_prompt):
    # api_key = os.environ["deepseek _API_KEY"]
    # client = genai.Client(api_key=os.environ["deepseek _API_KEY"])
    # response = client.models.generate_content_stream(
    # model=deepseek _model_name,
    # config=types.GenerateContentConfig(
    #     system_instruction=main_sys_prompt
    # ),
    # contents=main_user_prompt
    # )                    
    #             # response = deepseek _chat(main_user_prompt,main_sys_prompt)
    # for respond in response:
    #     yield respond.text

    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
 
    response = client.chat.completions.create(
        model=deepseek_model_name,
        messages=[
            {"role": "system", "content": main_sys_prompt},
            {"role": "user", "content": main_user_prompt},
        ],
        stream=True,
    )
 
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:  # skip None/empty chunks (role announcement, final chunk, etc.)
            yield delta
 




def query_rewrite(query:str)->str:
    regenerate_system_prompt = """You are a legal search query optimizer for a knowledge base of Ghanaian
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
    
    rewriten_query = deepseek_chat(query,regenerate_system_prompt)

    return rewriten_query.choices[0].message.content


if __name__ == "__main__":    
# crequest_context("what crime can lead to death sentence")
    generated_answer = ""
    with open("system3.txt","r") as f:
     system_prompt = f.read()
# print(system_prompt)     
    st.title("Personal legal Asistant")
    if "histories" not in st.session_state:
        st.session_state.histories = []   
    else:
     
        for history in st.session_state.histories:

            with st.chat_message(history["role"]):
                st.write(history["content"])

      
    if question := st.chat_input("Ask something..."):
        
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            query_text = query_rewrite(question)
            print(query_text)
            legistation_context = request_legislation_context(query_text,KB_CODE_legis)
            judgement_context = request_judgment_context(query_text,KB_CODE_judg)

            # print(context)
            user_prompt=prompt_template(legistation_chunks=legistation_context,question=question,judgement_chunks=judgement_context,history=st.session_state.histories)
            print(user_prompt)

            response_stream = deepseek_response_stream(main_user_prompt=user_prompt,main_sys_prompt=system_prompt)

            generated_answer = st.write_stream(response_stream)

        st.session_state.histories.append({"role": "user", "content": question})
        
        st.session_state.histories.append({"role": "assistant", "content": generated_answer})

