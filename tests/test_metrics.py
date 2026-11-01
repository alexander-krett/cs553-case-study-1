from unittest.mock import Mock

import pytest
from prometheus_client import REGISTRY

import app

ARGS = ("Resume", "General Resume Review", "", 512, 0.3)


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture(autouse=True)
def api_mode(monkeypatch):
    monkeypatch.setattr(app, "PRODUCT_MODE", "api")


def test_success_with_failover_records_request_and_model_calls(monkeypatch):
    primary, backup = app.product_models()
    before = {
        "success": sample("resumelens_requests_total", product="api", result="success"),
        "primary_error": sample("resumelens_model_calls_total", product="api", model=primary, result="error"),
        "backup_success": sample("resumelens_model_calls_total", product="api", model=backup, result="success"),
        "duration": sample("resumelens_request_duration_seconds_count", product="api"),
        "output": sample("resumelens_output_characters_sum", product="api"),
    }
    monkeypatch.setattr(app, "_run_remote_model", Mock(side_effect=[RuntimeError("down"), "feedback"]))

    app.analyze_resume(*ARGS, True, None, "hf_test")

    assert sample("resumelens_requests_total", product="api", result="success") == before["success"] + 1
    assert sample("resumelens_model_calls_total", product="api", model=primary, result="error") == before["primary_error"] + 1
    assert sample("resumelens_model_calls_total", product="api", model=backup, result="success") == before["backup_success"] + 1
    assert sample("resumelens_request_duration_seconds_count", product="api") == before["duration"] + 1
    assert sample("resumelens_output_characters_sum", product="api") == before["output"] + len("feedback")
    assert sample("resumelens_in_progress_requests", product="api") == 0


@pytest.mark.parametrize("token,resume,error_type", [
    ("", "Resume", "missing_token"),
    ("hf_test", "", "validation"),
])
def test_failures_record_error_type(monkeypatch, token, resume, error_type):
    before_errors = sample("resumelens_errors_total", product="api", error_type=error_type)
    before_requests = sample("resumelens_requests_total", product="api", result="error")
    monkeypatch.setattr(app, "_run_remote_model", Mock())

    with pytest.raises(app.gr.Error):
        app.analyze_resume(resume, *ARGS[1:], True, None, token)

    assert sample("resumelens_errors_total", product="api", error_type=error_type) == before_errors + 1
    assert sample("resumelens_requests_total", product="api", result="error") == before_requests + 1


def test_all_timeouts_are_classified(monkeypatch):
    before = sample("resumelens_errors_total", product="api", error_type="timeout")
    monkeypatch.setattr(app, "_run_remote_model", Mock(side_effect=TimeoutError("slow")))
    with pytest.raises(app.gr.Error):
        app.analyze_resume(*ARGS, True, None, "hf_test")
    assert sample("resumelens_errors_total", product="api", error_type="timeout") == before + 1


def test_build_info_is_exported():
    primary, backup = app.PRIMARY_MODEL, app.BACKUP_MODEL
    assert REGISTRY.get_sample_value("resumelens_build_info", {
        "product": "local", "version": app.APP_VERSION,
        "primary_model": primary, "backup_model": backup,
    }) == 1
