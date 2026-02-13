"""LLM-as-Judge evaluation using GPT-4o-mini.

Scores predicted answers on a 0-100 scale by comparing with gold answers.
"""

from __future__ import annotations

from openai import OpenAI

from config import get_settings


def judge_answer(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    openai_client: OpenAI | None = None,
) -> dict:
    """Use LLM to judge answer quality on 0-100 scale.

    Returns:
        dict with: score (int), reasoning (str)
    """
    settings = get_settings()
    client = openai_client or OpenAI(api_key=settings.openai.api_key)

    prompt = (
        f"You are an expert evaluator. Score the predicted answer on a scale of 0-100 "
        f"based on how well it matches the reference answer.\n\n"
        f"Question: {question}\n"
        f"Reference Answer: {gold_answer}\n"
        f"Predicted Answer: {predicted_answer}\n\n"
        f"Scoring criteria:\n"
        f"- 90-100: Fully correct, captures all key information\n"
        f"- 70-89: Mostly correct, minor omissions\n"
        f"- 50-69: Partially correct, significant gaps\n"
        f"- 30-49: Some relevant info but mostly wrong\n"
        f"- 0-29: Incorrect or irrelevant\n\n"
        f"Respond with ONLY a JSON object: {{\"score\": <int>, \"reasoning\": \"<brief explanation>\"}}"
    )

    try:
        response = client.chat.completions.create(
            model=settings.openai.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = response.choices[0].message.content or ""

        import json
        result = json.loads(text)
        return {
            "score": int(result.get("score", 0)),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        return {"score": 0, "reasoning": f"Evaluation error: {e}"}
