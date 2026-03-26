import json
import os
from typing import Dict, List, Optional

from flask import Flask, request, jsonify
from openai import OpenAI

from env.environment import SupportTicketEnv
from env.grader import grade_episode

app = Flask(__name__)

# Store active environments per session
environments: Dict[str, SupportTicketEnv] = {}
trajectories: Dict[str, List] = {}


def serialize_observation(obs):
    """Convert Pydantic Observation to dict."""
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    return obs.dict() if hasattr(obs, "dict") else obs


def serialize_reward(reward):
    """Convert Pydantic Reward to dict."""
    if hasattr(reward, "model_dump"):
        return reward.model_dump()
    return reward.dict() if hasattr(reward, "dict") else reward


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "version": "1.0"}), 200


@app.route("/reset", methods=["POST"])
def reset():
    """Reset environment and return initial observation."""
    data = request.get_json() or {}
    level = data.get("level", "easy")
    session_id = data.get("session_id", "default")

    env = SupportTicketEnv(level=level, seed=42)
    obs = env.reset()

    environments[session_id] = env
    trajectories[session_id] = []

    return jsonify(
        {
            "observation": serialize_observation(obs),
            "session_id": session_id,
            "level": level,
        }
    ), 200


@app.route("/step", methods=["POST"])
def step():
    """Execute one step in the environment."""
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    action = data.get("action", {})

    if session_id not in environments:
        return jsonify({"error": "Session not found. Call /reset first."}), 400

    env = environments[session_id]
    try:
        obs, reward, done, info = env.step(action)
        state = env.state()

        # Track trajectory
        trajectories[session_id].append(
            {
                "observation": serialize_observation(obs),
                "action": action,
                "reward": serialize_reward(reward),
                "done": done,
                "info": info,
            }
        )

        return jsonify(
            {
                "observation": serialize_observation(obs),
                "reward": serialize_reward(reward),
                "done": done,
                "info": info,
                "state": {
                    "step_count": state.step_count,
                    "resolved": state.resolved,
                    "escalated": state.escalated,
                    "total_reward": state.total_reward,
                },
            }
        ), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/state", methods=["POST"])
def get_state():
    """Get current environment state."""
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")

    if session_id not in environments:
        return jsonify({"error": "Session not found."}), 400

    env = environments[session_id]
    state = env.state()

    return jsonify(
        {
            "step_count": state.step_count,
            "resolved": state.resolved,
            "escalated": state.escalated,
            "total_reward": state.total_reward,
        }
    ), 200


@app.route("/grader", methods=["POST"])
def grader():
    """Compute episode grade."""
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")

    if session_id not in trajectories or not trajectories[session_id]:
        return jsonify({"error": "No trajectory found."}), 400

    score = grade_episode(trajectories[session_id])
    return jsonify({"episode_score": score}), 200


@app.route("/tasks", methods=["GET"])
def tasks():
    """Return task list and action schema."""
    return jsonify(
        {
            "tasks": [
                {
                    "name": "easy",
                    "description": "Single clear issue, no escalation needed",
                },
                {
                    "name": "medium",
                    "description": "Ambiguous intent, moderate reasoning required",
                },
                {
                    "name": "hard",
                    "description": "Multiple issues, emotional users, escalation may be required",
                },
            ],
            "action_schema": {
                "assign_category": {
                    "type": "string",
                    "choices": ["billing", "technical", "account", "abuse", "other"],
                    "required": True,
                },
                "set_priority": {
                    "type": "string",
                    "choices": ["low", "medium", "high", "critical"],
                    "required": True,
                },
                "response": {
                    "type": "string",
                    "required": True,
                    "description": "Support agent response message",
                },
                "escalate": {
                    "type": "boolean",
                    "required": True,
                    "description": "Whether to escalate to human",
                },
            },
        }
    ), 200


@app.route("/baseline", methods=["POST"])
def baseline():
    """Run baseline agent on all tasks and return scores."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 500

    try:
        client = OpenAI(api_key=api_key)

        results = {}
        for level in ["easy", "medium", "hard"]:
            env = SupportTicketEnv(level=level, seed=42)
            scores = []

            for _ in range(3):
                obs = env.reset()
                prompt = (
                    f"You are a customer support triage assistant.\n"
                    f"Ticket ID: {obs.ticket_id}\n"
                    f"Message: {obs.message}\n"
                    f"User history: {obs.user_history}\n"
                    f"Sentiment: {obs.sentiment}\n"
                    f"Urgency: {obs.urgency_hint}\n"
                    f"Previous attempts: {obs.previous_attempts}\n"
                    f"Respond with exact format:\n"
                    f"Category: <billing/technical/account/abuse/other>\n"
                    f"Priority: <low/medium/high/critical>\n"
                    f"Response: <supportful agent message>\n"
                    f"Escalate: <true/false>\n"
                )

                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an accurate ticket triage model.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=400,
                )

                answer = completion.choices[0].message.content
                action = parse_answer(answer)

                obs_next, reward, done, info = env.step(action)
                scores.append(reward.score)

            results[level] = sum(scores) / len(scores)

        return (
            jsonify({
                "baseline_scores": results,
                "overall": sum(results.values()) / len(results),
            }),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def parse_answer(text: str) -> Dict[str, any]:
    """Parse model response into action."""
    import re

    parsed = {
        "assign_category": "other",
        "set_priority": "low",
        "response": "",
        "escalate": False,
    }
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    for line in lines:
        if line.lower().startswith("category"):
            parsed["assign_category"] = (
                re.sub(r"^.*?:", "", line, flags=re.IGNORECASE)
                .strip()
                .lower()
            )
        elif line.lower().startswith("priority"):
            parsed["set_priority"] = (
                re.sub(r"^.*?:", "", line, flags=re.IGNORECASE)
                .strip()
                .lower()
            )
        elif line.lower().startswith("escalate"):
            val = (
                re.sub(r"^.*?:", "", line, flags=re.IGNORECASE)
                .strip()
                .lower()
            )
            parsed["escalate"] = val in ["true", "yes", "1"]
        elif line.lower().startswith("response"):
            parsed["response"] = (
                re.sub(r"^.*?:", "", line, flags=re.IGNORECASE).strip()
            )

    if not parsed["response"]:
        body = text.strip()
        if "response" in body.lower():
            body_split = re.split(r"response\s*:", body, flags=re.IGNORECASE)
            if len(body_split) > 1:
                parsed["response"] = body_split[1].strip()
        else:
            parsed["response"] = body

    return parsed


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
