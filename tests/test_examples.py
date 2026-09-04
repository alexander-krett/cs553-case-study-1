from src.examples import (
    EXAMPLES,
    JOB_DESCRIPTION_ML_ENGINEER,
    RESUME_BULLET_POINT_STRENGTH,
    RESUME_JOB_MATCH,
)
from src.prompts import REVIEW_TYPES, build_prompt


def test_examples_exist_and_not_empty():
    assert len(EXAMPLES) >= 2
    for example in EXAMPLES:
        assert len(example) == 3
        resume_text, review_type, job_description = example
        assert isinstance(resume_text, str) and resume_text.strip()
        assert isinstance(review_type, str) and review_type in REVIEW_TYPES
        assert isinstance(job_description, str)


def test_job_description_match_example_has_valid_job_description():
    match_examples = [ex for ex in EXAMPLES if ex[1] == "Job Description Match"]
    assert len(match_examples) > 0
    for resume_text, review_type, job_description in match_examples:
        assert job_description.strip() != ""


def test_examples_build_prompts_successfully():
    for resume_text, review_type, job_description in EXAMPLES:
        prompt = build_prompt(resume_text, review_type, job_description)
        assert len(prompt) > 0
        assert review_type in prompt
