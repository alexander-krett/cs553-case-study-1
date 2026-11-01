from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import app

ARGS = ("Resume", "General Resume Review", "", 512, 0.3)


@pytest.fixture(params=["local", "api"])
def mode(request, monkeypatch):
    monkeypatch.setattr(app, "PRODUCT_MODE", request.param)
    return request.param


def _review_name(mode):
    return "remote_resume_review" if mode == "api" else "local_resume_review"


def _run_name(mode):
    return "_run_remote_model" if mode == "api" else "_run_local_model"


def test_failover_dispatches_to_active_product(mode, monkeypatch):
    review = Mock(return_value=("feedback", "info"))
    monkeypatch.setattr(app, _review_name(mode), review)
    assert app.analyze_resume(*ARGS, True, None, "hf_test") == ("feedback", "info")
    review.assert_called_once()
    assert review.call_args.kwargs["models"] is None


def test_backup_model_used(mode, monkeypatch):
    run = Mock(side_effect=[TimeoutError("timeout"), "feedback"])
    monkeypatch.setattr(app, _run_name(mode), run)
    response, information = app.analyze_resume(*ARGS, True, None, "hf_test")
    assert response == "feedback"
    assert "Backup model used" in information
    assert [c.args[0] for c in run.call_args_list] == list(app.product_models())


def test_manual_selection_uses_only_selected_model(mode, monkeypatch):
    backup = app.product_models()[1]
    review = Mock(return_value=("feedback", "info"))
    monkeypatch.setattr(app, _review_name(mode), review)
    app.analyze_resume(*ARGS, False, backup, "hf_test")
    assert review.call_args.kwargs["models"] == [backup]


def test_model_from_other_product_is_rejected(mode, monkeypatch):
    other = "api" if mode == "local" else "local"
    run = Mock()
    monkeypatch.setattr(app, _run_name(mode), run)
    with pytest.raises(app.gr.Error, match="Invalid model"):
        app.analyze_resume(*ARGS, False, app.PRODUCT_MODELS[other][0], "hf_test")
    run.assert_not_called()


def test_validation_before_inference(mode, monkeypatch):
    run = Mock()
    monkeypatch.setattr(app, _run_name(mode), run)
    with pytest.raises(app.gr.Error, match="paste a resume"):
        app.analyze_resume("", *ARGS[1:], True, None, "hf_test")
    run.assert_not_called()


def test_token_comes_only_from_ui(monkeypatch):
    monkeypatch.setattr(app, "PRODUCT_MODE", "api")
    monkeypatch.setenv("HF_TOKEN", "env-secret")
    run = Mock()
    monkeypatch.setattr(app, "_run_remote_model", run)
    with pytest.raises(app.gr.Error, match="token is required"):
        app.analyze_resume(*ARGS, True, None, "  ")
    run.assert_not_called()

    run.return_value = "feedback"
    app.analyze_resume(*ARGS, True, None, " hf_ui ")
    assert run.call_args.args[1] == "hf_ui"


def test_errors_redact_ui_token(monkeypatch):
    monkeypatch.setattr(app, "PRODUCT_MODE", "api")
    monkeypatch.setattr(app, "_run_remote_model", Mock(side_effect=RuntimeError("bad hf_secret")))
    with pytest.raises(app.gr.Error) as error:
        app.analyze_resume(*ARGS, True, None, "hf_secret")
    assert "hf_secret" not in str(error.value)
    assert "[redacted]" in str(error.value)


def test_remote_empty_response(monkeypatch):
    client = Mock()
    client.chat_completion.return_value = Mock(choices=[Mock(message=Mock(content=""), finish_reason="length")])
    monkeypatch.setattr(app, "InferenceClient", Mock(return_value=client))
    with pytest.raises(RuntimeError, match="empty response"):
        app._run_remote_model(app.REMOTE_PRIMARY_MODEL, "hf_test", "prompt", 512, .3)
    assert client.chat_completion.call_args.kwargs["extra_body"] == {"reasoning_effort": "low"}


class FakeLlama:
    def __init__(self, token_count, content):
        self.token_count = token_count
        self.content = content
        self.request = None

    def tokenize(self, text):
        return [1] * self.token_count

    def create_chat_completion(self, **request):
        self.request = request
        return {"choices": [{"message": {"content": self.content}}]}


@pytest.mark.parametrize("token_count,content,expected_error", [
    (100, " feedback ", None),
    (4096, "feedback", "exceeds local context"),
    (100, "<think>\n\n</think>\n\n", "empty response"),
])
def test_local_capacity_and_response(monkeypatch, token_count, content, expected_error):
    llm = FakeLlama(token_count, content)
    monkeypatch.setattr(app, "_load_local_model", Mock(return_value=llm))
    if expected_error:
        with pytest.raises(RuntimeError, match=expected_error):
            app._run_local_model(app.LOCAL_PRIMARY_MODEL, "prompt", 512, .3)
    else:
        assert app._run_local_model(app.LOCAL_PRIMARY_MODEL, "prompt", 512, .3) == "feedback"
        assert llm.request["max_tokens"] == 512
    if token_count == 4096:
        assert llm.request is None


@pytest.mark.parametrize("model,suffix", [
    (app.LOCAL_PRIMARY_MODEL, True),
    (app.LOCAL_BACKUP_MODEL, False),
])
def test_qwen_thinking_is_disabled(monkeypatch, model, suffix):
    llm = FakeLlama(10, "feedback")
    monkeypatch.setattr(app, "_load_local_model", Mock(return_value=llm))
    app._run_local_model(model, "prompt", 512, .3)
    assert llm.request["messages"][-1]["content"].endswith("/no_think") is suffix


def test_every_local_model_has_a_q4_file():
    for model in app.PRODUCT_MODELS["local"]:
        assert "Q4" in app.LOCAL_MODEL_FILES[model]


def test_model_picker_state():
    assert not app.model_picker_state(True).get_config()["interactive"]
    assert app.model_picker_state(False).get_config()["interactive"]
