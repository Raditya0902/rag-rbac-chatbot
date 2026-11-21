"""Tests for refusal detection and refusal-accuracy scoring of no_answer questions."""
import pytest
from ragas import SingleTurnSample

from src.monitoring.refusal import is_refusal, split_by_answerability, summarize_refusals
from src.rag.prompts import REFUSAL_MESSAGE


def _sample(question, response):
    return SingleTurnSample(user_input=question, response=response, retrieved_contexts=["ctx"])


META = {
    "ceo?": {"type": "no_answer", "role": "General"},
    "ticker?": {"type": "no_answer", "role": "General"},
    "leave?": {"type": "factual", "role": "General"},
    "budget?": {"type": "multi_hop", "role": "Finance"},
}


def test_prompt_instructs_the_exact_refusal_message():
    from src.rag.prompts import RAG_PROMPT

    system_template = RAG_PROMPT.messages[0].prompt.template

    assert REFUSAL_MESSAGE in system_template


@pytest.mark.parametrize("response", [
    REFUSAL_MESSAGE,
    "  i don't have enough information to answer that  ",
    "'I don't have enough information to answer that.'",
    "I don’t have enough information to answer that.",  # curly apostrophe
    "I don't have enough information to answer that. The context covers Q1 only.",
])
def test_detects_refusal_variants(response):
    assert is_refusal(response)


@pytest.mark.parametrize("response", [
    "Employees get 12 days of sick leave per year.",
    "I don't have the exact figure, but revenue grew 20%.",
    "",
])
def test_ignores_non_refusals(response):
    assert not is_refusal(response)


def test_split_sends_only_no_answer_questions_to_refusal_scoring():
    samples = [_sample("ceo?", "x"), _sample("leave?", "y"), _sample("new?", "z")]

    answerable, no_answer = split_by_answerability(samples, META)

    assert [s.user_input for s in answerable] == ["leave?", "new?"]  # unknown type is answerable
    assert [s.user_input for s in no_answer] == ["ceo?"]


def test_summary_counts_correct_refusals_and_false_refusals():
    no_answer = [_sample("ceo?", REFUSAL_MESSAGE), _sample("ticker?", "It's FINS.")]
    answerable = [_sample("leave?", "12 days."), _sample("budget?", REFUSAL_MESSAGE)]

    summary = summarize_refusals(answerable, no_answer)

    assert summary["n_no_answer"] == 2
    assert summary["correct"] == 1
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["missed"] == ["ticker?"]
    assert summary["n_answerable"] == 2
    assert summary["false_refusals"] == 1
    assert summary["false_refusal_questions"] == ["budget?"]


def test_summary_with_no_no_answer_questions_has_no_accuracy():
    summary = summarize_refusals([_sample("leave?", "12 days.")], [])

    assert summary["n_no_answer"] == 0
    assert summary["accuracy"] is None
