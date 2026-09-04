import pytest

from src.prompts import build_prompt


def test_empty_resume_is_rejected():
    with pytest.raises(ValueError, match="Please paste a resume"):
        build_prompt("  ", "General Resume Review")


def test_invalid_review_type_is_rejected():
    with pytest.raises(ValueError, match="Invalid review type"):
        build_prompt("Resume", "Unknown Review")


def test_job_match_requires_job_description():
    with pytest.raises(ValueError, match="Please provide a job description"):
        build_prompt("Resume", "Job Description Match", "  ")
