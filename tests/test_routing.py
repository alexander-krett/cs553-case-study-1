from unittest.mock import Mock, call, ANY
import httpx
import pytest
import app

ARGS = ("Resume", "General Resume Review", "", 512, 0.3)

@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-secret")

@pytest.mark.parametrize("mode", ["Local", "Remote"])
def test_first_mode(monkeypatch, mode):
    first = Mock(return_value=("feedback", "info"))
    other = Mock()
    monkeypatch.setattr(app, mode.lower() + "_resume_review", first)
    monkeypatch.setattr(app, ("remote" if mode == "Local" else "local") + "_resume_review", other)
    assert app.analyze_resume(*ARGS, True, mode, app.LOCAL_PRIMARY_MODEL) == ("feedback", "info")
    other.assert_not_called()

@pytest.mark.parametrize("mode", ["Local", "Remote"])
def test_cross_mode_failover(monkeypatch, mode):
    monkeypatch.setattr(app, mode.lower() + "_resume_review", Mock(side_effect=RuntimeError("unavailable")))
    other = "remote" if mode == "Local" else "local"
    monkeypatch.setattr(app, other + "_resume_review", Mock(return_value=("feedback", "Response time: 1")))
    assert "Switched from" in app.analyze_resume(*ARGS, True, mode, app.LOCAL_PRIMARY_MODEL)[1]

@pytest.mark.parametrize("execution", ["local", "remote"])
def test_backup(monkeypatch, execution):
    run = Mock(side_effect=[httpx.ReadTimeout("timeout"), "feedback"])
    monkeypatch.setattr(app, "_run_" + execution + "_model", run)
    assert "Backup model used" in getattr(app, execution + "_resume_review")(*ARGS)[1]
    assert run.call_count == 2

@pytest.mark.parametrize("model", [app.LOCAL_PRIMARY_MODEL, app.REMOTE_BACKUP_MODEL])
def test_manual_selection(monkeypatch, model):
    execution = app.MODEL_EXECUTION[model].lower()
    review = Mock(return_value=("feedback", "info"))
    monkeypatch.setattr(app, execution + "_resume_review", review)
    app.analyze_resume(*ARGS, False, "Local", model)
    review.assert_called_once_with(*ARGS, models=[model])

def test_missing_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(RuntimeError, match="HF_TOKEN is required"):
        app.require_hf_token()
    with pytest.raises(RuntimeError, match="HF_TOKEN is required"):
        app.remote_resume_review(*ARGS)

def test_validation_before_failover(monkeypatch):
    run = Mock()
    monkeypatch.setattr(app, "_run_local_model", run)
    with pytest.raises(app.gr.Error, match="paste a resume"):
        app.analyze_resume("", *ARGS[1:], True, "Local", app.LOCAL_PRIMARY_MODEL)
    run.assert_not_called()

def test_invalid_selection():
    with pytest.raises(app.gr.Error, match="Invalid model"):
        app.analyze_resume(*ARGS, False, "Local", "invalid")
    with pytest.raises(app.gr.Error, match="Invalid inference mode"):
        app.analyze_resume(*ARGS, True, "invalid", app.LOCAL_PRIMARY_MODEL)

def test_combined_errors_redact_credentials(monkeypatch):
    monkeypatch.setattr(app, "_run_local_model", Mock(side_effect=RuntimeError("test-secret")))
    monkeypatch.setattr(app, "_run_remote_model", Mock(side_effect=RuntimeError("test-secret")))
    with pytest.raises(app.gr.Error) as error:
        app.analyze_resume(*ARGS, True, "Local", app.LOCAL_PRIMARY_MODEL)
    assert "test-secret" not in str(error.value)

def test_remote_empty_response(monkeypatch):
    client = Mock()
    client.chat_completion.return_value = Mock(choices=[Mock(message=Mock(content=""), finish_reason="length")])
    monkeypatch.setattr(app, "InferenceClient", Mock(return_value=client))
    with pytest.raises(RuntimeError, match="empty response"):
        app._run_remote_model(app.REMOTE_PRIMARY_MODEL, "test-secret", "prompt", 512, .3)
    assert client.chat_completion.call_args.kwargs["extra_body"] == {"reasoning_effort": "low"}

@pytest.mark.parametrize("token_count,content,expected_error", [(100," feedback ",None),(4096,"feedback","exceeds local context"),(100,"","empty response")])
def test_local_capacity_and_response(monkeypatch, token_count, content, expected_error):
    client = Mock()
    client.post.side_effect = [Mock(json=lambda: {"prompt":"formatted"}), Mock(json=lambda: {"tokens":[1]*token_count}), Mock(json=lambda: {"choices":[{"message":{"content":content}}]})]
    manager = Mock()
    manager.__enter__ = Mock(return_value=client)
    manager.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(app.httpx, "Client", Mock(return_value=manager))
    if expected_error:
        with pytest.raises(RuntimeError, match=expected_error):
            app._run_local_model(app.LOCAL_PRIMARY_MODEL, "prompt", 512, .3)
    else:
        assert app._run_local_model(app.LOCAL_PRIMARY_MODEL, "prompt", 512, .3) == "feedback"
        assert client.post.call_args.kwargs["json"]["chat_template_kwargs"] == {"enable_thinking":False}
    if token_count == 4096:
        assert client.post.call_count == 2

def test_control_visibility():
    mode, model = app.execution_control_visibility(True)
    assert mode.get_config()["visible"]
    assert not model.get_config()["visible"]
