"""
Prompt construction for ResumeLens AI.

This module intentionally contains no Gradio, Transformers,
or Hugging Face API code so that it can be tested independently.
"""

# Each review type maps to a different set of instructions.
# This lets the user decide what kind of resume feedback they
# want without changing the underlying application structure.
REVIEW_INSTRUCTIONS = {
    "General Resume Review": """
Evaluate the resume's overall clarity, organization, impact,
and effectiveness.

Focus on:
1. The strongest aspects of the resume
2. The most important weaknesses
3. Three concrete improvements
4. One example of how a weak bullet could be improved
""",

    "Bullet Point Strength": """
Focus specifically on the experience bullet points.

Evaluate:
1. Action verbs
2. Specificity
3. Evidence of impact
4. Quantifiable results
5. Technical clarity

Identify the weakest bullet and explain how it could be improved.
Provide one suggested rewrite without inventing accomplishments.
""",

    "Skills Analysis": """
Analyze the skills demonstrated by the resume.

Identify:
1. Skills clearly demonstrated by experience
2. Skills listed but weakly supported by experience
3. Important technical skills that are difficult to find
4. Ways the applicant could make existing skills more visible

Do not invent skills that are not present in the resume.
""",

    "Job Description Match": """
Compare the resume with the supplied job description.

Identify:
1. Requirements that are clearly supported by the resume
2. Requirements that are partially supported
3. Requirements that are not evident in the resume
4. Existing experience that could be emphasized more clearly

Do not produce an arbitrary ATS score.
Do not make a hiring recommendation.
Do not invent qualifications that are not present in the resume.
""",
}

# This list is created directly from the dictionary keys so that
# the dropdown options and prompt-validation logic remain consistent.
REVIEW_TYPES = list(REVIEW_INSTRUCTIONS.keys())

# This prompt defines the behavior we expect from both the remote
# and local models. Using the same system prompt for both models 
# helps make the comparison between them more meaningful.
SYSTEM_PROMPT = """
You are ResumeLens AI, an assistant designed to help job seekers
improve their own resumes.

Provide specific, constructive, evidence-based feedback using only
the resume and job description supplied by the user.

Important rules:

- Do not invent employment history, education, skills,
accomplishments, certifications, or numerical results.

- Do not infer or comment on protected or sensitive personal
characteristics.

- Do not estimate age, race, ethnicity, religion, disability status,
gender, sexual orientation, health status, or other sensitive traits.

- Do not recommend whether an employer should hire or reject someone.

- Do not rank applicants or compare this applicant with other people.

- Do not generate arbitrary ATS scores.

- When comparing a resume with a job description, discuss only
whether specific qualifications are evident in the supplied text.

- Make feedback concrete and actionable.

- If information is missing, say that it is not evident from the
resume instead of assuming that the applicant lacks the skill.
""".strip()


def build_prompt(
    resume_text: str,
    review_type: str,
    job_description: str = "",
) -> str:
    """
    Build a resume-review prompt used by the local and remote models.

    Parameters
    resume_text: Resume text supplied by the user.

    review_type: One of the supported review categories.

    job_description: Optional job description. Required when using
    "Job Description Match".

    Returns
    str: Formatted prompt sent to the language model.

    Raises
    ValueError: If the resume is empty, the review type is invalid,
    or Job Description Match is selected without a job description.
    """
    # Validate that the user actually supplied a resume
    if not resume_text or not resume_text.strip():
        raise ValueError(
            "Please paste a resume before requesting a review."
        )
    # Validate the selected review category
    # Normally the Gradio dropdown prevents invalid values,
    # but validating here makes the function safer and easier
    # to test independently.
    if review_type not in REVIEW_INSTRUCTIONS:
        raise ValueError(
            f"Invalid review type: {review_type}"
        )
    # Job Description Match requires a second text input
    if (
        review_type == "Job Description Match"
        and (
            not job_description
            or not job_description.strip()
        )
    ):
        raise ValueError(
            "Please provide a job description when using "
            "Job Description Match."
        )

    instruction = REVIEW_INSTRUCTIONS[review_type]
    # The sections are separated clearly so the model can distinguish
    # the review instructions from the actual resume content.
    prompt = f"""
REVIEW TYPE
{review_type}


REVIEW INSTRUCTIONS
{instruction.strip()}


RESUME
------
{resume_text.strip()}
"""
    # Add the job description only when one was supplied
    if job_description and job_description.strip():
        # Add final response constraints
        # These rules reduce the chance that the model invents details
        # while rewriting resume content.
        prompt += f"""


JOB DESCRIPTION
{job_description.strip()}
"""

    prompt += """


RESPONSE GUIDELINES
Base your response only on the supplied material.

Be specific and concise.

When suggesting rewritten resume bullets, do not invent
numbers, technologies, responsibilities, or accomplishments.

If relevant information is missing, state that it is
"not evident from the resume."
"""
# Remove unnecessary whitespace before returning the prompt.
    return prompt.strip()
