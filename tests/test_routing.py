from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

# Unit tests use explicit fake tokens and must not require a local HF login.
with patch("gradio.oauth._get_mocked_oauth_info", return_value={}):
    import app


COMMON_ARGS = (
    "Resume",
    "General Resume Review",
    "",
    300,
    0.3,
)


def test_local_mode_routes_without_oauth(monkeypatch):
    local = Mock(return_value=("local feedback", "local info"))
    remote = Mock()
    monkeypatch.setattr(app, "local_resume_review", local)
    monkeypatch.setattr(app, "remote_resume_review", remote)

    result = app.analyze_resume(*COMMON_ARGS, "Local", None)

    assert result == ("local feedback", "local info")
    local.assert_called_once_with(*COMMON_ARGS)
    remote.assert_not_called()


def test_remote_mode_routes_with_oauth(monkeypatch):
    token = SimpleNamespace(token="secret")
    remote = Mock(return_value=("remote feedback", "remote info"))
    monkeypatch.setattr(app, "remote_resume_review", remote)

    result = app.analyze_resume(*COMMON_ARGS, "Remote", token)

    assert result == ("remote feedback", "remote info")
    remote.assert_called_once_with(*COMMON_ARGS, token)


def test_remote_mode_requires_oauth():
    with pytest.raises(app.gr.Error, match="Please sign in with Hugging Face"):
        app.remote_resume_review(*COMMON_ARGS, None)


def test_remote_request_uses_low_reasoning_effort(monkeypatch):
    message = SimpleNamespace(content="Useful feedback")
    choice = SimpleNamespace(message=message, finish_reason="stop")
    client = Mock()
    client.chat_completion.return_value = SimpleNamespace(choices=[choice])
    monkeypatch.setattr(app, "InferenceClient", Mock(return_value=client))

    result = app.remote_resume_review(
        *COMMON_ARGS, SimpleNamespace(token="secret")
    )

    assert result[0] == "Useful feedback"
    assert client.chat_completion.call_args.kwargs["extra_body"] == {
        "reasoning_effort": "low"
    }
    assert client.chat_completion.call_args.kwargs["max_tokens"] == 1536


def test_empty_remote_response_reports_finish_reason(monkeypatch):
    message = SimpleNamespace(content="")
    choice = SimpleNamespace(message=message, finish_reason="length")
    client = Mock()
    client.chat_completion.return_value = SimpleNamespace(choices=[choice])
    monkeypatch.setattr(app, "InferenceClient", Mock(return_value=client))

    with pytest.raises(app.gr.Error, match="finish reason: length"):
        app.remote_resume_review(*COMMON_ARGS, SimpleNamespace(token="secret"))


def test_invalid_mode_is_rejected():
    with pytest.raises(app.gr.Error, match="Invalid inference mode"):
        app.analyze_resume(*COMMON_ARGS, "Other", None)
