import os
import re
import sys
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from env.environment import SupportTicketEnv
from env.grader import grade_episode


MOCK_ANSWER = (
    "Category: billing\n"
    "Priority: high\n"
    "Response: We apologize for the inconvenience. We will investigate the billing issue immediately."
)


def parse_answer(text: str) -> Dict[str, str]:
    # Expect format with Category:, Priority:, Response:
    parsed = {"assign_category": "other", "set_priority": "low", "response": ""}
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("category"):
            parsed["assign_category"] = re.sub(r"^.*?:", "", line, flags=re.IGNORECASE).strip().lower()
        elif line.lower().startswith("priority"):
            parsed["set_priority"] = re.sub(r"^.*?:", "", line, flags=re.IGNORECASE).strip().lower()
        elif line.lower().startswith("response"):
            parsed["response"] = re.sub(r"^.*?:", "", line, flags=re.IGNORECASE).strip()
    # if no explicit response lines, use full text after prefix
    if not parsed["response"]:
        body = text.strip()
        if "response" in body.lower():
            body_split = re.split(r"response\s*:", body, flags=re.IGNORECASE)
            if len(body_split) > 1:
                parsed["response"] = body_split[1].strip()
        else:
            parsed["response"] = body

    return parsed


def generate_answer(prompt: str) -> str:
    use_mock = os.getenv("BASELINE_USE_MOCK", "").lower() in {"1", "true", "yes"}
    api_key = os.getenv("OPENAI_API_KEY")

    if use_mock:
        return MOCK_ANSWER

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable must be set to run the baseline. "
            "Set BASELINE_USE_MOCK=1 only if you want the offline mock fallback."
        )

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an accurate ticket triage model."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=400,
    )

    return completion.choices[0].message.content or ""


def run_baseline():
    levels = ["easy", "medium", "hard"]
    results = {}

    for level in levels:
        env = SupportTicketEnv(level=level)
        trajectories = []

        for i in range(5):
            obs = env.reset()
            prompt = (
                f"You are a customer support triage assistant.\n"
                f"Ticket ID: {obs.ticket_id}\n"
                f"Message: {obs.message}\n"
                f"User history: {obs.user_history}\n"
                f"Current status: {obs.current_status}\n"
                f"Urgency hint: {obs.urgency_hint}\n"
                "Respond with exact format:\n"
                "Category: <billing/technical/account/other>\n"
                "Priority: <low/medium/high>\n"
                "Response: <supportful agent message>\n"
            )

            answer = generate_answer(prompt)
            action = parse_answer(answer)

            obs_next, reward, done, info = env.step(action)
            trajectories.append({"observation": obs_next, "action": action, "reward": reward.model_dump()})

        score = grade_episode(trajectories)
        results[level] = score

    print("Baseline task scores:")
    for level, score in results.items():
        print(f"  {level}: {score:.3f}")

    overall = sum(results.values()) / len(results)
    print(f"Overall score: {overall:.3f}")


if __name__ == "__main__":
    run_baseline()
