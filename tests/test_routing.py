from types import SimpleNamespace
from unittest.mock import ANY, Mock, call, patch

import pytest

# Unit tests use explicit fake tokens and must not require a local HF login.
with patch("gradio.oauth._get_mocked_oauth_info", return_value={}):
    import app


COMMON_ARGS = ("Resume", "General Resume Review", "", 2048, 0.3)


def test_local_mode_routes_with_automatic_failover(monkeypatch):
    local = Mock(return_value=("local feedback", "local info"))
    remote = Mock()
    monkeypatch.setattr(app, "local_resume_review", local)
    monkeypatch.setattr(app, "remote_resume_review", remote)

    result = app.analyze_resume(*COMMON_ARGS, True, "Local", None)

    assert result == ("local feedback", "local info")
    local.assert_called_once_with(*COMMON_ARGS)
    remote.assert_not_called()


def test_remote_mode_routes_with_automatic_failover(monkeypatch):
    token = SimpleNamespace(token="secret")
    remote = Mock(return_value=("remote feedback", "remote info"))
    monkeypatch.setattr(app, "remote_resume_review", remote)

    result = app.analyze_resume(*COMMON_ARGS, True, "Remote", token)

    assert result == ("remote feedback", "remote info")
    remote.assert_called_once_with(*COMMON_ARGS, token)


def test_manual_model_selection_bypasses_failover(monkeypatch):
    local = Mock(return_value=("feedback", "info"))
    monkeypatch.setattr(app, "local_resume_review", local)

    result = app.analyze_resume(*COMMON_ARGS, False, app.LOCAL_BACKUP_MODEL, None)

    assert result == ("feedback", "info")
    local.assert_called_once_with(*COMMON_ARGS, models=[app.LOCAL_BACKUP_MODEL])


def test_local_failure_uses_backup(monkeypatch):
    run_model = Mock(side_effect=[RuntimeError("primary unavailable"), "feedback"])
    monkeypatch.setattr(app, "_run_local_model", run_model)

    result = app.local_resume_review(*COMMON_ARGS)

    assert result[0] == "feedback"
    assert "Backup model used" in result[1]
    assert app.LOCAL_BACKUP_MODEL in result[1]
    assert run_model.call_args_list == [
        call(app.LOCAL_PRIMARY_MODEL, ANY, 2048, 0.3),
        call(app.LOCAL_BACKUP_MODEL, ANY, 2048, 0.3),
    ]


def test_remote_failure_uses_backup(monkeypatch):
    run_model = Mock(side_effect=[RuntimeError("primary unavailable"), "feedback"])
    monkeypatch.setattr(app, "_run_remote_model", run_model)
    token = SimpleNamespace(token="secret")

    result = app.remote_resume_review(*COMMON_ARGS, token)

    assert result[0] == "feedback"
    assert "Backup model used" in result[1]
    assert app.REMOTE_BACKUP_MODEL in result[1]


def test_both_remote_failures_are_reported(monkeypatch):
    monkeypatch.setattr(
        app,
        "_run_remote_model",
        Mock(side_effect=[RuntimeError("first"), RuntimeError("second")]),
    )

    with pytest.raises(app.gr.Error, match=app.REMOTE_BACKUP_MODEL):
        app.remote_resume_review(*COMMON_ARGS, SimpleNamespace(token="secret"))


def test_remote_mode_requires_oauth(monkeypatch):
    run_model = Mock()
    monkeypatch.setattr(app, "_run_remote_model", run_model)

    with pytest.raises(app.gr.Error, match="Please sign in with Hugging Face"):
        app.remote_resume_review(*COMMON_ARGS, None)

    run_model.assert_not_called()


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (
            app.REMOTE_PRIMARY_MODEL,
            {"reasoning_effort": "low"},
        ),
        (
            app.REMOTE_BACKUP_MODEL,
            {"reasoning_effort": "low", "clear_thinking": True},
        ),
    ],
)
def test_remote_models_use_low_reasoning_effort(monkeypatch, model_id, expected):
    message = SimpleNamespace(content="Useful feedback")
    choice = SimpleNamespace(message=message, finish_reason="stop")
    client = Mock()
    client.chat_completion.return_value = SimpleNamespace(choices=[choice])
    monkeypatch.setattr(app, "InferenceClient", Mock(return_value=client))

    result = app._run_remote_model(model_id, "secret", "prompt", 2048, 0.3)

    assert result == "Useful feedback"
    assert client.chat_completion.call_args.kwargs["extra_body"] == expected
    assert client.chat_completion.call_args.kwargs["max_tokens"] == 2048


def test_empty_remote_response_can_trigger_failover(monkeypatch):
    empty_message = SimpleNamespace(content="")
    empty_choice = SimpleNamespace(message=empty_message, finish_reason="length")
    good_message = SimpleNamespace(content="Backup feedback")
    good_choice = SimpleNamespace(message=good_message, finish_reason="stop")
    first_client = Mock()
    first_client.chat_completion.return_value = SimpleNamespace(choices=[empty_choice])
    second_client = Mock()
    second_client.chat_completion.return_value = SimpleNamespace(choices=[good_choice])
    monkeypatch.setattr(
        app, "InferenceClient", Mock(side_effect=[first_client, second_client])
    )

    result = app.remote_resume_review(
        *COMMON_ARGS, SimpleNamespace(token="secret")
    )

    assert result[0] == "Backup feedback"
    assert "Backup model used" in result[1]


def test_execution_choices_follow_failover_setting():
    failover = app.execution_choices(True).get_config()
    manual = app.execution_choices(False).get_config()

    assert failover["value"] == "Remote"
    assert [choice[1] for choice in failover["choices"]] == ["Remote", "Local"]
    assert manual["value"] == app.REMOTE_PRIMARY_MODEL
    assert [choice[1] for choice in manual["choices"]] == [
        app.REMOTE_PRIMARY_MODEL,
        app.REMOTE_BACKUP_MODEL,
        app.LOCAL_PRIMARY_MODEL,
        app.LOCAL_BACKUP_MODEL,
    ]


def test_invalid_mode_is_rejected():
    with pytest.raises(app.gr.Error, match="Invalid inference mode"):
        app.analyze_resume(*COMMON_ARGS, True, "Other", None)


def test_invalid_manual_model_is_rejected():
    with pytest.raises(app.gr.Error, match="Invalid model"):
        app.analyze_resume(*COMMON_ARGS, False, "unknown/model", None)
