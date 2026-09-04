from src.prompts import build_prompt


def test_build_prompt_includes_resume_and_review_instructions():
    prompt = build_prompt("Built a web application.", "General Resume Review")

    assert "Built a web application." in prompt
    assert "Evaluate the resume's overall clarity" in prompt
    assert "JOB DESCRIPTION" not in prompt


def test_build_prompt_includes_job_description_when_supplied():
    prompt = build_prompt(
        "Python developer",
        "Job Description Match",
        "Seeking a Python developer",
    )

    assert "JOB DESCRIPTION" in prompt
    assert "Seeking a Python developer" in prompt
    assert prompt.endswith('"not evident from the resume."')
