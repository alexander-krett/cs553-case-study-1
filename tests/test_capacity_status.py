import json
import time

import app


def test_capacity_status_reports_normal_overload_and_stale_state(tmp_path, monkeypatch):
    status_file = tmp_path / "status.json"
    monkeypatch.setattr(app, "RESOURCE_STATUS_FILE", status_file)

    status_file.write_text(json.dumps({
        "timestamp": time.time(), "cpu_percent": 41.2,
        "memory_percent": 52.6, "overloaded": False,
    }))
    normal = app.resource_capacity_status()
    assert "Operating normally" in normal
    assert "41%" in normal and "53%" in normal

    status_file.write_text(json.dumps({
        "timestamp": time.time(), "cpu_percent": 91.0,
        "memory_percent": 84.0, "overloaded": True,
    }))
    warning = app.resource_capacity_status()
    assert "Near capacity" in warning
    assert "reviews remain available" in warning

    status_file.write_text(json.dumps({
        "timestamp": time.time() - 20, "cpu_percent": 20,
        "memory_percent": 30, "overloaded": False,
    }))
    assert "monitoring unavailable" in app.resource_capacity_status()


def test_missing_status_does_not_disable_application(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RESOURCE_STATUS_FILE", tmp_path / "missing.json")
    assert "monitoring unavailable" in app.resource_capacity_status()
    assert callable(app.analyze_resume)
