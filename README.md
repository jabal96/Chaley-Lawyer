# Obeng: AI Legal Assistant for Ghanaian Law

Understanding your rights shouldn't require a law degree or an expensive consultation. Obeng is your personal legal guide: like a knowledgeable friend who has read the law and can explain it in plain language, in English or Pidgin, any time you need it.

Describe your situation the way you'd tell a friend. Obeng searches real Ghanaian legislation and court judgments, explains what the law says in simple terms, shows you exactly which sections and cases it relied on, and tells you what to do next, including when it's time to see a real lawyer.

- **Know where you stand.** Find out what the law says about your situation before you sign, pay, or argue.
- **Grounded in real law.** Every answer is tied to actual Acts and judgments, with source links so you can check them yourself.
- **Honest about its limits.** If the documents don't cover your question, Obeng says so instead of guessing.

> **Important:** Obeng is an AI legal information tool, not a lawyer. It cannot represent you or give formal legal advice. For anything serious (arrest, eviction, custody, contracts, court), speak to a qualified lawyer or visit Legal Aid Ghana.

## How it works

1. **Query rewriting:** the user's casual question (for example, "my uncle died and left no will") is rewritten into a search query suited to legal text.
2. **Retrieval:** the query is sent to the [Laws.Africa Knowledge Bases API](https://developers.laws.africa/knowledge-bases/knowledge-bases) to fetch relevant chunks from two sources:
   - Ghanaian **legislation** (Acts and sections)
   - Ghanaian **judgments** (case law)
3. **Grounded generation:** the retrieved chunks and the question go to the LLM (DeepSeek, via the OpenAI-compatible SDK) with a strict system prompt. The prompt requires every legal claim to come from the retrieved text, to be cited, and to be explained in simple language.
4. **Answer with sources:** the response streams into a Streamlit chat UI, followed by links to the sources used.

## Features

- Plain-language answers, in English or Pidgin
- Citations tied to specific sections and cases
- Declines to guess when the retrieved documents don't cover the question
- Retrieval from both statutes and case law
- Evaluation scripts for measuring answer quality (see below)

## Tech stack

- Python 3.12+, [uv](https://docs.astral.sh/uv/) for dependencies
- Streamlit for the chat UI
- Laws.Africa Knowledge Bases API for retrieval
- DeepSeek (via the `openai` SDK) for generation
- [Ragas](https://docs.ragas.io/) for reference-free evaluation

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/jabal96/Chaley-Lawyer.git
cd Chaley-Lawyer
```

**2. Install dependencies**
```bash
uv sync
```

**3. Add your API keys.** Copy `.env.example` to `.env` and fill in your values:
```
DEEPSEEK_API_KEY=
GEMINI_API_KEY=
LAWS_AFRICA_API_KEY=
```

| Variable | Used for | Where to get it |
|---|---|---|
| `LAWS_AFRICA_API_KEY` | Retrieving legislation and judgments | [Laws.Africa](https://developers.laws.africa/) |
| `DEEPSEEK_API_KEY` | Generating answers | [DeepSeek Platform](https://platform.deepseek.com/) |
| `GEMINI_API_KEY` | Evaluation judge model (and anything still using Gemini) | [Google AI Studio](https://aistudio.google.com/) |

Never commit your `.env` file. It is already listed in `.gitignore`.

**4. Run the app**
```bash
streamlit run Main_app.py
```

## Project structure

```
Main_app.py                  # Streamlit app: query rewrite, retrieval, generation
system3.txt                  # System prompt that governs Obeng's behaviour
RAGAS_evaluation_script.py   # Ragas-based evaluation of the full pipeline
eval_queries.json            # Test questions used for evaluation
```

## Evaluation

Retrieval and answer quality are measured with Ragas, using three reference-free metrics:

- **Faithfulness:** does the answer stick to what was retrieved?
- **Answer relevancy:** does the answer address the question asked?
- **Context relevance:** was the retrieved context relevant to the question?

Run it with:
```bash
python RAGAS_evaluation_script.py
```
Results are written to `ragas_results.csv`. Test questions live in `eval_queries.json`, written the way real users ask, including Pidgin phrasing.

## Known limitations

- Retrieval is imperfect. Some questions return loosely related sections instead of the governing Act. This is the main area of ongoing work.
- Answers are only as complete as the documents retrieved. When coverage is partial, Obeng says so.
- Coverage depends on what Laws.Africa has indexed for Ghana.

## Roadmap

- Better query rewriting and multi-query retrieval
- Reranking of retrieved chunks
- Larger evaluation set and tracked scores over time
- Deployment for public use

## License

Add a license before making the repo public (MIT is a common choice).

## Acknowledgements

Legal content is provided through [Laws.Africa](https://laws.africa/) and the [Ghana Legal Information Institute](https://ghalii.org/). Check their terms of use before redistributing any content.