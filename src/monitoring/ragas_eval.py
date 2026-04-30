import os
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain_groq import ChatGroq
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy

from src.auth.rbac import ROLE_NAMESPACE_MAP
from src.config import settings
from src.vectorstore.store import get_embeddings, get_rbac_retriever

EVAL_QUESTIONS = [
    # General
    {"question": "What is the company's remote work policy?",                    "role": "General"},
    {"question": "What are the core values of the company?",                     "role": "General"},
    # Finance
    {"question": "What are the quarterly revenue figures?",                      "role": "Finance"},
    {"question": "What is the company's net profit margin?",                     "role": "Finance"},
    # Engineering
    {"question": "What technologies does the engineering team use?",             "role": "Engineering"},
    {"question": "What is the software development process used by the team?",   "role": "Engineering"},
    # HR
    {"question": "What are the employee benefits?",                              "role": "HR"},
    {"question": "What is the leave policy for employees?",                      "role": "HR"},
    # Marketing
    {"question": "What were the Q4 marketing results?",                          "role": "Marketing"},
    {"question": "What is the target customer segment for marketing campaigns?", "role": "Marketing"},
]


def run_evaluation() -> None:
    from src.rag.chain import run_query

    # Use a larger model for scoring — small 8b models are weak at the NLI
    # decomposition task that RAGAS faithfulness relies on.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        eval_llm = LangchainLLMWrapper(
            ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)
        )
        eval_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    # strictness=1 → generates 1 synthetic question per sample, avoiding
    # Groq's rejection of n>1 which caused answer_relevancy to always return 0.
    metrics = [
        Faithfulness(llm=eval_llm),
        AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings, strictness=1),
    ]

    samples = []
    for item in EVAL_QUESTIONS:
        namespaces = ROLE_NAMESPACE_MAP.get(item["role"], ["general"])
        result = run_query(item["question"], namespaces)
        if result["guardrail_blocked"]:
            continue

        retriever = get_rbac_retriever(namespaces)
        docs = retriever.invoke(item["question"])
        retrieved_contexts = [doc.page_content for doc in docs]

        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=result["answer"],
            retrieved_contexts=retrieved_contexts,
        ))

    scores = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
    )
    df = scores.to_pandas()
    print("\n=== RAGAS Evaluation Results ===")
    print(f"  faithfulness    : {df['faithfulness'].mean():.4f}")
    print(f"  answer_relevancy: {df['answer_relevancy'].mean():.4f}")
    print("================================\n")


if __name__ == "__main__":
    run_evaluation()
