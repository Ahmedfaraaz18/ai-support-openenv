import json
import os
import re
from typing import Any, Dict, List

from openai import OpenAI

from env.environment import SupportTicketEnv
from env.grader import grade_episode


MOCK_ANSWER = (
    "Category: billing\n"
    "Priority: high\n"
    "Response: We apologize for the inconvenience. We will investigate the billing issue immediately."
)

LEVELS = ["easy", "medium", "hard"]
EPISODES_PER_LEVEL = 5


def emit_log(tag: str, **fields: Any) -> None:
    ordered = {key: fields[key] for key in sorted(fields)}
    print(f"{tag} {json.dumps(ordered, ensure_ascii=True, separators=(',', ':'))}")


def serialize_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def parse_answer(text: str) -> Dict[str, str]:
    parsed = {"assign_category": "other", "set_priority": "low", "response": ""}
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    for line in lines:
        if line.lower().startswith("category"):
            parsed["assign_category"] = re.sub(r"^.*?:", "", line, flags=re.IGNORECASE).strip().lower()
        elif line.lower().startswith("priority"):
            parsed["set_priority"] = re.sub(r"^.*?:", "", line, flags=re.IGNORECASE).strip().lower()
        elif line.lower().startswith("response"):
            parsed["response"] = re.sub(r"^.*?:", "", line, flags=re.IGNORECASE).strip()

    if not parsed["response"]:
        body = text.strip()
        if "response" in body.lower():
            body_split = re.split(r"response\s*:", body, flags=re.IGNORECASE)
            if len(body_split) > 1:
                parsed["response"] = body_split[1].strip()
        else:
            parsed["response"] = body

    return parsed


def get_client() -> OpenAI | None:
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")
    use_mock = os.getenv("BASELINE_USE_MOCK", "").lower() in {"1", "true", "yes"}

    if use_mock:
        return None

    missing = [
        name
        for name, value in {
            "API_BASE_URL": api_base_url,
            "MODEL_NAME": model_name,
            "HF_TOKEN": hf_token,
        }.items()
        if not value
    ]
    if missing:
        raise EnvironmentError(
            "Missing required environment variables for inference: "
            + ", ".join(missing)
            + ". Set BASELINE_USE_MOCK=1 only for offline fallback."
        )

    return OpenAI(base_url=api_base_url, api_key=hf_token)


def generate_answer(prompt: str, client: OpenAI | None, model_name: str | None) -> str:
    if client is None:
        return MOCK_ANSWER

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an accurate ticket triage model."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    return completion.choices[0].message.content or ""


def build_prompt(obs: Any) -> str:
    return (
        "You are a customer support triage assistant.\n"
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


def run_inference() -> Dict[str, float]:
    results: Dict[str, float] = {}
    use_mock = os.getenv("BASELINE_USE_MOCK", "").lower() in {"1", "true", "yes"}
    client = None if use_mock else get_client()
    model_name = os.getenv("MODEL_NAME")

    emit_log(
        "[START]",
        episodes_per_level=EPISODES_PER_LEVEL,
        levels=LEVELS,
        mock_mode=use_mock,
        model_name=model_name or "mock",
    )

    for level in LEVELS:
        env = SupportTicketEnv(level=level, seed=42)
        trajectories: List[Dict[str, Any]] = []

        for episode_index in range(1, EPISODES_PER_LEVEL + 1):
            obs = env.reset()
            answer = generate_answer(build_prompt(obs), client, model_name)
            action = parse_answer(answer)
            obs_next, reward, done, info = env.step(action)
            trajectory_step = {
                "observation": serialize_model(obs_next),
                "action": action,
                "reward": serialize_model(reward),
            }
            trajectories.append(trajectory_step)

            emit_log(
                "[STEP]",
                action=action,
                done=done,
                episode=episode_index,
                info=info,
                level=level,
                observation=serialize_model(obs),
                reward=serialize_model(reward),
                state=serialize_model(env.state()),
            )

        results[level] = grade_episode(trajectories)

    overall = sum(results.values()) / len(results)
    emit_log("[END]", overall_score=overall, task_scores=results)
    return results


def main():
    run_inference()


if __name__ == "__main__":
    main()
