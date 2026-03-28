import os
from typing import Any, Dict, List

from flask import Flask, request, jsonify

from env.environment import SupportTicketEnv
from env.grader import grade_episode
from inference import generate_answer, parse_answer

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


@app.route("/", methods=["GET"])
def index():
    """Simple landing page for browser-based checks."""
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>OpenEnv Support Agent</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.5; }
          h1 { margin-bottom: 8px; }
          code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
          ul { padding-left: 20px; }
        </style>
      </head>
      <body>
        <h1>OpenEnv Support Agent</h1>
        <p>The server is running.</p>
        <p>Useful endpoints:</p>
        <ul>
          <li><code>GET /health</code></li>
          <li><code>GET /tasks</code></li>
          <li><code>POST /reset</code></li>
          <li><code>POST /step</code></li>
          <li><code>POST /state</code></li>
          <li><code>POST /grader</code></li>
          <li><code>POST /baseline</code></li>
        </ul>
      </body>
    </html>
    """, 200


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
                    "resolved": state.ticket_resolved,
                    "ticket_resolved": state.ticket_resolved,
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
            "resolved": state.ticket_resolved,
            "ticket_resolved": state.ticket_resolved,
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
                    "description": "Multiple issues, emotional users, high reasoning required",
                },
            ],
            "action_schema": {
                "assign_category": {
                    "type": "string",
                    "choices": ["billing", "technical", "account", "other"],
                    "required": True,
                },
                "set_priority": {
                    "type": "string",
                    "choices": ["low", "medium", "high"],
                    "required": True,
                },
                "response": {
                    "type": "string",
                    "required": True,
                    "description": "Support agent response message",
                },
            },
        }
    ), 200


@app.route("/baseline", methods=["POST"])
def baseline():
    """Run baseline agent on all tasks and return scores."""
    try:
        use_mock = os.getenv("BASELINE_USE_MOCK", "").lower() in {"1", "true", "yes"}
        client = None
        model_name = os.getenv("MODEL_NAME")
        if not use_mock:
            from inference import get_client

            client = get_client()

        results = {}
        for level in ["easy", "medium", "hard"]:
            env = SupportTicketEnv(level=level, seed=42)
            trajectories_for_level = []

            for _ in range(3):
                obs = env.reset()
                prompt = (
                    f"You are a customer support triage assistant.\n"
                    f"Ticket ID: {obs.ticket_id}\n"
                    f"Message: {obs.message}\n"
                    f"User history: {obs.user_history}\n"
                    f"Current status: {obs.current_status}\n"
                    f"Urgency hint: {obs.urgency_hint}\n"
                    f"Respond with exact format:\n"
                    f"Category: <billing/technical/account/other>\n"
                    f"Priority: <low/medium/high>\n"
                    f"Response: <supportful agent message>\n"
                )

                answer = generate_answer(prompt, client, model_name)
                action = parse_answer(answer)

                obs_next, reward, done, info = env.step(action)
                trajectories_for_level.append(
                    {
                        "observation": serialize_observation(obs_next),
                        "action": action,
                        "reward": serialize_reward(reward),
                    }
                )

            results[level] = grade_episode(trajectories_for_level)

        return (
            jsonify({
                "baseline_scores": results,
                "overall": sum(results.values()) / len(results),
            }),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
