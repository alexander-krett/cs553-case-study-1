from types import SimpleNamespace
from unittest.mock import Mock

import pytest

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


def test_invalid_mode_is_rejected():
    with pytest.raises(app.gr.Error, match="Invalid inference mode"):
        app.analyze_resume(*COMMON_ARGS, "Other", None)
