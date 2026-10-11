"""
Evaluates the legal assistant's RAG pipeline using Ragas, without
requiring hand-labeled ground truth. Runs each test query through the
REAL retrieval + generation pipeline, captures (query, retrieved
context, generated answer) triples, then scores them with three
reference-free Ragas metrics:

  - Faithfulness      -> does the answer stick to what was retrieved?
  - AnswerRelevancy   -> does the answer actually address the question?
  - ContextRelevance  -> was the retrieved context relevant to the question?
"""
import json
from pathlib import Path
import os
import json
import asyncio
from sentence_transformers import SentenceTransformer
import pandas as pd
from google import genai
from google.genai import types

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextRelevance,
)
from openai import AsyncOpenAI
from Main_app import query_rewrite, request_legislation_context, request_judgment_context, deepseek_model_name,deepseek_chat
from Main_app import KB_CODE_legis, KB_CODE_judg
from openai import OpenAI
TEST_SET_PATH = "eval_queries.json"
# GEMINI_MODEL_NAME = "gemini-3.6-flash"  # the JUDGE model - can differ from gemini_model_name (your app's generation model)
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EVALAUTION_MODEL= "deepseek-v4-pro"
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

def build_evaluator_llm():
    """
    The 'judge' model Ragas uses to score faithfulness/relevancy/precision.
    """
    # client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # return llm_factory(GEMINI_MODEL_NAME, provider="google", client=client)
    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return llm_factory(model=EVALAUTION_MODEL,provider="openai",client=client,max_tokens=4096,extra_body={"thinking": {"type": "disabled"}},)


def build_evaluator_embeddings():
    """
    AnswerRelevancy (unlike Faithfulness and ContextRelevance) needs an
    embedding model to compute similarity between the question and 
    hypothetical questions generated from the answer.
    """
    # client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # return GoogleEmbeddings(client=client, model=EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbeddings(model=embedding_model_name)
    


# def generate_full_answer(prompt_text: str, system_prompt: str) -> str:
#     """
#     Non-streaming generation for eval purposes. Ragas needs the complete
#     answer as one string, not a stream of chunks - so unlike your app's
#     gemini_response_stream, this collects everything before returning.

#     Uses gemini_model_name (your app's actual generation model) here -
#     NOT GEMINI_MODEL_NAME (the judge model) - since this call is meant to
#     reproduce exactly what your app itself would generate.
#     """
#     client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
#     response = client.models.generate_content(
#         model=gemini_model_name,
#         config=types.GenerateContentConfig(system_instruction=system_prompt),
#         contents=prompt_text,
#     )
#     return response.text


def prompt_template(legistation_chunks: list, question: str, judgement_chunks: list) -> str:
    legis_chunk_strings = "\n".join(legistation_chunks)
    judge_chunk_strings = "\n".join(judgement_chunks)
    prompt = (
        f"CONTEXT-LEGISLATION LEGAL CHUNKS: \n{legis_chunk_strings}\n"
        f"CONTEXT-JUDGMENT CASE LAW CHUNKS: \n{judge_chunk_strings}\n"
        f"User question: {question}\n"
    )
    return prompt


def run_pipeline_for_query(question: str, system_prompt: str) -> dict:
    """
    Runs ONE query through your real retrieval + generation pipeline and
    returns the (query, retrieved_contexts, answer) triple that Ragas needs.

    """
    rewritten_query = query_rewrite(question)

    legislation_chunks = request_legislation_context(rewritten_query, kb_code=KB_CODE_legis)
    judgment_chunks = request_judgment_context(rewritten_query, kb_code=KB_CODE_judg)

    retrieved_contexts = legislation_chunks + judgment_chunks

    full_prompt = prompt_template(
        legistation_chunks=legislation_chunks,
        question=question,
        judgement_chunks=judgment_chunks,
    )

    # answer = generate_full_answer(full_prompt, system_prompt)
    answer = deepseek_chat(user_prompt=full_prompt,system_prompt=system_prompt)
    

    return {
        "user_input": question,
        "retrieved_contexts": retrieved_contexts,
        "response": answer.choices[0].message.content
    }


# def build_results(test_set_path: str, system_prompt: str) -> list[dict]:
#     """
#     Loads the test queries and runs each one through the real pipeline,
#     returning a plain list of dicts (query/contexts/answer) - one per query.
#     """
#     with open(test_set_path, "r", encoding="utf-8") as f:
#         test_cases = json.load(f)

#     results = []
#     for case in test_cases:
#         query = case["question"]
#         print(f"Running pipeline for: {query[:60]}...")
#         results.append(run_pipeline_for_query(query, system_prompt))

#     return results




def build_results(test_set_path: str, system_prompt: str, cache_path: str = "pipeline_results.json") -> list[dict]:
    """
    Loads the test queries and runs each one through the real pipeline,
    returning a plain list of dicts (query/contexts/answer) - one per query.
    Caches results to disk so a crash during scoring doesn't require re-running the pipeline.
    """
    cache_file = Path(cache_path)
    if cache_file.exists():
        print(f"Found cached results at {cache_path}, loading instead of re-running pipeline...")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(test_set_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []
    for case in test_cases:
        query = case["question"]
        print(f"Running pipeline for: {query[:60]}...")
        results.append(run_pipeline_for_query(query, system_prompt))

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved pipeline results to {cache_path}")

    return results

async def score_all(results: list[dict]) -> pd.DataFrame:
    """
    Scores every (query, contexts, answer) triple using the three collections
    metrics directly via their .ascore() method.
    """
    evaluator_llm = build_evaluator_llm()
    evaluator_embeddings = build_evaluator_embeddings()

    faithfulness = Faithfulness(llm=evaluator_llm)
    answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    context_relevance = ContextRelevance(llm=evaluator_llm)

    rows = []
    for r in results:
        print(f"Scoring: {r['user_input'][:60]}...")

        faithfulness_result = await faithfulness.ascore(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
        )
        relevancy_result = await answer_relevancy.ascore(
            user_input=r["user_input"],
            response=r["response"],
            # retrieved_contexts=r["retrieved_contexts"],
        )
        # ContextRelevance judges the context against the question only -
        # it does not take the generated response as input at all.
        context_result = await context_relevance.ascore(
            user_input=r["user_input"],
            retrieved_contexts=r["retrieved_contexts"],
        )

        rows.append({
            "user_input": r["user_input"],
            "faithfulness": faithfulness_result.value,
            "answer_relevancy": relevancy_result.value,
            "context_relevance": context_result.value,
        })

    return pd.DataFrame(rows)


def run_evaluation(system_prompt_path: str = "system3.txt"):
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    print("Running pipeline for each test query...")
    results = build_results(TEST_SET_PATH, system_prompt)

    print("Scoring with Ragas metrics (this calls the judge LLM multiple times per query)...")
    df = asyncio.run(score_all(results))

    print("\n" + "=" * 70)
    print("RESULTS (sorted by faithfulness, worst first)")
    print("=" * 70)
    print(df.sort_values("faithfulness").to_string(index=False))

    df.to_csv("ragas_results.csv", index=False)
    print("\nFull results saved to ragas_results.csv")

    return df


if __name__ == "__main__":
    run_evaluation()