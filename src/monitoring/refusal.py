"""Refusal scoring for questions the corpus cannot answer.

RAGAS gives a correct "I don't know" 0 on both faithfulness (no claims to
verify) and answer relevancy (non-committal). So no_answer questions are scored
here instead, as refusal accuracy, and kept out of the RAGAS means. Answerable
questions that were refused are counted as false refusals and stay in RAGAS,
where they score 0 as they should.
"""
from ragas import SingleTurnSample

from src.rag.prompts import REFUSAL_MESSAGE

NO_ANSWER_TYPE = "no_answer"


def _normalize(text: str) -> str:
    text = text.replace("’", "'").lower()
    return " ".join(text.split()).strip(" '\".")


_REFUSAL_KEY = _normalize(REFUSAL_MESSAGE)


def is_refusal(response: str) -> bool:
    """True if the response contains the prompt's exact refusal sentence."""
    return _REFUSAL_KEY in _normalize(response or "")


def split_by_answerability(
    samples: list[SingleTurnSample], question_meta: dict[str, dict]
) -> tuple[list[SingleTurnSample], list[SingleTurnSample]]:
    """Return (answerable, no_answer). Questions without metadata count as answerable."""
    def is_no_answer(s: SingleTurnSample) -> bool:
        return question_meta.get(s.user_input, {}).get("type") == NO_ANSWER_TYPE

    answerable = [s for s in samples if not is_no_answer(s)]
    no_answer = [s for s in samples if is_no_answer(s)]
    return answerable, no_answer


def summarize_refusals(
    answerable: list[SingleTurnSample], no_answer: list[SingleTurnSample]
) -> dict:
    """Refusal accuracy on no_answer questions, plus false refusals on answerable ones."""
    missed = [s.user_input for s in no_answer if not is_refusal(s.response)]
    false_refusals = [s.user_input for s in answerable if is_refusal(s.response)]
    n_correct = len(no_answer) - len(missed)
    return {
        "n_no_answer": len(no_answer),
        "correct": n_correct,
        "accuracy": n_correct / len(no_answer) if no_answer else None,
        "missed": missed,
        "n_answerable": len(answerable),
        "false_refusals": len(false_refusals),
        "false_refusal_questions": false_refusals,
    }
