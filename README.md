

# AI Customer Support Triage + Escalation Environment
=======
# AI Customer Support Triage Environment
>>>>>>> a77f3af (updated files)

This repository implements a realistic OpenEnv task for customer support ticket triage. The agent receives support tickets, classifies them into the right queue, assigns a handling priority, and drafts a helpful reply. The task is operational and real-world, not a game or toy benchmark.

## Overview

The environment models a support workflow with three difficulty levels:

1. `easy`: single-issue tickets with clear intent
2. `medium`: ambiguous tickets with overlapping cues
3. `hard`: noisy multi-problem tickets that require stronger reasoning

The benchmark is designed to satisfy the OpenEnv hackathon requirements:

- Typed models for observation, action, reward, and state
- Full `reset()`, `step()`, and `state()` environment API
- Three graded task levels from easy to hard
- Dense reward shaping with partial-progress signals
- Deterministic episode grading in the `0.0` to `1.0` range
- Reproducible baseline inference script
- Docker deployment suitable for Hugging Face Spaces

## Observation Space

The environment returns the following typed observation:

```python
Observation(
    ticket_id: int,
    message: str,
    user_history: list[str],
    current_status: str,
    urgency_hint: str,
)
```

## Action Space

The agent acts with:

```python
Action(
    assign_category: str,  # billing | technical | account | other
    set_priority: str,     # low | medium | high
    response: str,
)
```

## Reward Function

The step reward is dense and provides partial credit:

- Category correctness contributes `40%`
- Priority correctness contributes `30%`
- Response quality contributes `30%`
- Invalid or empty actions incur penalties
- Repeated invalid actions incur an additional penalty

Step rewards are clipped to `[-1.0, 1.0]`.

The episode grader in [env/grader.py](/c:/Users/ahmed/OneDrive/Desktop/openenv-support-agent/env/grader.py) converts trajectories to a deterministic `0.0` to `1.0` score by combining normalized reward history with final-step correctness.

## State

The environment exposes:

```python
State(
    step_count: int,
    ticket_resolved: bool,
    total_reward: float,
)
```

## OpenEnv Components

The main implementation lives in:

- [env/models.py](/c:/Users/ahmed/OneDrive/Desktop/openenv-support-agent/env/models.py)
- [env/environment.py](/c:/Users/ahmed/OneDrive/Desktop/openenv-support-agent/env/environment.py)
- [env/grader.py](/c:/Users/ahmed/OneDrive/Desktop/openenv-support-agent/env/grader.py)
- [env/tasks.py](/c:/Users/ahmed/OneDrive/Desktop/openenv-support-agent/env/tasks.py)
- [openenv.yaml](/c:/Users/ahmed/OneDrive/Desktop/openenv-support-agent/openenv.yaml)

## Tasks

Task metadata is declared in [openenv.yaml](/c:/Users/ahmed/OneDrive/Desktop/openenv-support-agent/openenv.yaml) and backed by realistic ticket fixtures in [env/tasks.py](/c:/Users/ahmed/OneDrive/Desktop/openenv-support-agent/env/tasks.py).

- `easy`
  Clear issues such as duplicate charges or straightforward password-reset problems.
- `medium`
  Tickets with mixed signals, overlapping account and billing clues, or less direct phrasing.
- `hard`
  Multi-issue tickets with conflicting symptoms, requiring better prioritization and response content.

## Baseline Inference

The baseline runner is implemented in [scripts/baseline.py](/c:/Users/ahmed/OneDrive/Desktop/openenv-support-agent/scripts/baseline.py).

By default it uses the OpenAI API client and reads credentials from `OPENAI_API_KEY`:

```bash
set OPENAI_API_KEY=sk-...
python -m scripts.baseline
```

For an offline fallback during local development only:

```bash
set BASELINE_USE_MOCK=1
python -m scripts.baseline
```

Mock-mode output on this repo:

```text
Baseline task scores:
  easy: 0.325
  medium: 0.744
  hard: 0.718
Overall score: 0.595
```

On Linux or macOS:

```bash
export OPENAI_API_KEY=sk-...
python -m scripts.baseline
```

## Setup

### Local Development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If the plain `python` launcher is unreliable on your Windows setup, use:

```bash
.\.venv\Scripts\python.exe -m scripts.baseline
.\.venv\Scripts\python.exe validate.py
```

### Validation

Run the local compliance validator:

```bash
python validate.py
```

## API Server

The Flask API is implemented in [scripts/server.py](/c:/Users/ahmed/OneDrive/Desktop/openenv-support-agent/scripts/server.py).

Run it locally with:

```bash
python scripts/server.py
```

Endpoints:

- `GET /health`
- `POST /reset`
- `POST /step`
- `POST /state`
- `POST /grader`
- `GET /tasks`
- `POST /baseline`

## Docker and Hugging Face Spaces

The repository includes a Dockerfile that installs dependencies from [requirements.txt](/c:/Users/ahmed/OneDrive/Desktop/openenv-support-agent/requirements.txt) and serves the Flask app on port `7860`.
Inside the container, the app is started with `gunicorn` rather than Flask's development server.

Build and run locally:

```bash
docker build -t openenv-support-agent .
docker run -p 7860:7860 openenv-support-agent
```

### Hugging Face Spaces

1. Create a new Space on Hugging Face.
2. Choose the `Docker` SDK.
3. Push this repository to the Space or upload the repository contents.
4. In the Space settings, add `OPENAI_API_KEY` so the baseline can use the OpenAI client.
5. Set `BASELINE_USE_MOCK=1` only if you want the offline fallback instead of live model calls.
6. Let the Space build the Docker image from the included [Dockerfile](/c:/Users/ahmed/OneDrive/Desktop/openenv-support-agent/Dockerfile).
7. After deployment, verify:
   `GET /health`
   `GET /tasks`
   `POST /baseline`
8. Share the generated Space URL in your submission.

## Project Structure

```text
.
|-- env/
|   |-- models.py
|   |-- environment.py
|   |-- grader.py
|   `-- tasks.py
|-- scripts/
|   |-- baseline.py
|   `-- server.py
|-- configs/
|-- data/
|-- Dockerfile
|-- openenv.yaml
|-- requirements.txt
`-- validate.py
```

## License

<<<<<<< HEAD
## 📝 Technical Details

### Deterministic Grading

The grader evaluates episodes consistently:
1. Computes average reward across all steps
2. Compares final action against expected outcomes
3. Returns normalized 0.0–1.0 score

Reproducible with fixed seeds (seed=42 in all examples).

### Multi-Step Episodes

Each episode allows up to 3 steps:
- **Step 1**: Initial response to ticket
- **Step 2**: Refinement or escalation decision
- **Step 3**: Final resolution or confirmed escalation

Episodes terminate when: resolved=True OR escalated=True OR step_count=3

### Reward Shaping Philosophy

- **Dense rewards** (not just terminal): agent gets feedback each step
- **Partial credit**: wrong category still earns some points
- **Incentive alignment**: penalizing bad escalation, rewarding politeness
- **Learning-friendly**: reward signal guides toward human-like behavior

---

## 🎓 Further Reading

- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv)
- [Gym Interface Documentation](https://gymnasium.farama.org/)
- [Reward Shaping in RL](https://arxiv.org/abs/1811.02762)

---

## 📬 Support

For hackathon-specific questions: `help_openenvhackathon@scaler.com`

---

## 📄 License

MIT (Open for educational and commercial use)

---

**Built for**: OpenEnv Hackathon 2026  
**Submission Deadline**: 7 April 2026 11:59 PM UTC

# ai-support-openenv
OpenEnv-based AI environment for customer support triage and escalation, featuring realistic ticket simulation, multi-step decision making, and reward shaping for agent evaluation.
=======
MIT

