"""Tests for building the RAGAS judge LLM from provider + model settings.

Chat model classes are replaced with mocks; no network calls are made.
"""
from unittest.mock import MagicMock, patch

import pytest

import src.monitoring.judge as judge


@pytest.fixture
def chat_classes():
    groq, openai = MagicMock(name="ChatGroq"), MagicMock(name="ChatOpenAI")
    with patch.object(judge, "ChatGroq", groq), patch.object(judge, "ChatOpenAI", openai), \
         patch.object(judge.settings, "GROQ_API_KEY", "groq-test-key"), \
         patch.object(judge.settings, "DEEPSEEK_API_KEY", "deepseek-test-key"):
        yield groq, openai


def test_resolve_uses_provider_default_model_when_none_given():
    assert judge.resolve_judge("groq", None) == ("groq", "llama-3.1-8b-instant")
    assert judge.resolve_judge("deepseek", None) == ("deepseek", "deepseek-chat")


def test_resolve_falls_back_to_settings(monkeypatch):
    monkeypatch.setattr(judge.settings, "EVAL_JUDGE_PROVIDER", "deepseek")
    monkeypatch.setattr(judge.settings, "EVAL_JUDGE_MODEL", "deepseek-reasoner")

    assert judge.resolve_judge(None, None) == ("deepseek", "deepseek-reasoner")


def test_resolve_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown judge provider 'openia'"):
        judge.resolve_judge("openia", None)


def test_groq_judge_uses_chat_groq(chat_classes):
    groq, openai = chat_classes

    judge.build_judge_chat_model("groq", "llama-3.3-70b-versatile")

    groq.assert_called_once_with(model="llama-3.3-70b-versatile", api_key="groq-test-key")
    openai.assert_not_called()


def test_deepseek_judge_uses_openai_compatible_endpoint(chat_classes):
    groq, openai = chat_classes

    judge.build_judge_chat_model("deepseek", "deepseek-chat")

    kwargs = openai.call_args.kwargs
    assert kwargs["model"] == "deepseek-chat"
    assert kwargs["api_key"] == "deepseek-test-key"
    assert kwargs["base_url"] == "https://api.deepseek.com"
    groq.assert_not_called()


@pytest.mark.parametrize("provider,key_attr", [("groq", "GROQ_API_KEY"), ("deepseek", "DEEPSEEK_API_KEY")])
def test_missing_api_key_fails_fast(chat_classes, provider, key_attr):
    with patch.object(judge.settings, key_attr, ""):
        with pytest.raises(ValueError, match=key_attr):
            judge.build_judge_chat_model(provider, "any-model")
