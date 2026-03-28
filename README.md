
# AI Customer Support Triage + Escalation Environment

## 🎯 Overview

A production-grade **OpenEnv environment** for training AI agents on realistic customer support workflows. Agents must classify tickets, assign priorities, generate helpful responses, and decide when to escalate to human handlers.

**Real-world impact**: This environment trains agents to handle billions of customer support interactions at scale while maintaining quality and human oversight.

---

## 🌟 Why This Environment?

**The Problem**:
- Modern support systems process millions of tickets daily
- Manual triage is slow and inconsistent
- Fully automated responses miss context and emotional nuance
- Escalation timing is critical: escalate too early = wasted human time; too late = customer frustration

**The Solution**:
- Train RL agents to make intelligent triage decisions
- Dense reward shaping guides learning step-by-step
- Multi-step episodes allow refinement and escalation
- Deterministic grading ensures fair evaluation

---

## 📋 Tasks

### 1. EASY: Clear Intent (6 tickets)
Single issue, obvious category, straightforward resolution.

**Example**:
```
Message: "I was charged twice for my last invoice, please fix it."
Expected: billing/high/refund required
```

### 2. MEDIUM: Ambiguous Intent (6 tickets)
Multiple issues or unclear user intent; requires reasoning.

**Example**:
```
Message: "Something is broken and I'm not sure what to do. Also my bill looks wrong."
Expected: Either billing+technical (depends on interpretation) /medium/escalation likely
```

### 3. HARD: Multi-Problem + Emotional Users (6 tickets)
Complex scenarios, frustrated users, multiple issues, escalation required.

**Example**:
```
Message: "I've been trying to fix this for 3 days and your support is terrible! My account locked after I reset my password and now I'm being charged for something I cancelled!"
Expected: account/critical + technical/escalation required
```

---

## 🏗️ Environment Design

### Observation Fields

```python
Observation(
    ticket_id: int,
    message: str,              # realistic messy human language
    user_history: list[str],   # past interactions
    sentiment: str,            # "angry", "neutral", "happy"
    urgency_hint: str,         # context from ticket metadata
    previous_attempts: int,    # how many times support tried
)
```

### Action Schema

```python
Action(
    assign_category: str,      # "billing", "technical", "account", "abuse", "other"
    set_priority: str,         # "low", "medium", "high", "critical"
    response: str,             # agent's message to customer
    escalate: bool,            # escalate to human?
)
```

### Reward Function (Dense, Multi-Component)

```
score = 0.3 * category_correctness 
      + 0.2 * priority_correctness
      + 0.25 * response_quality
      + 0.25 * escalation_correctness
      - penalties

Penalties:
  - Wrong escalation decision: -0.2
  - Empty/irrelevant response: -0.3
  - Repeated bad actions: -0.1

Bonuses:
  - Response contains politeness keywords ("sorry", "assist"): +0.05
  - Correct escalation when needed: +0.1
```

### State Tracking

```python
State(
    step_count: int,           # 0-3 per episode
    resolved: bool,            # ticket fully resolved?
    escalated: bool,           # escalated to human?
    total_reward: float,       # cumulative reward
)
```

---

## 🚀 Deployment

### Local Development

```bash
# Install dependencies
pip install pydantic openai flask pyyaml

# Run Flask server
python scripts/server.py

# Server starts at http://localhost:7860
```

### Docker Deployment (Recommended for HF Spaces)

```bash
docker build -t openenv-support .
docker run -p 7860:7860 -e OPENAI_API_KEY="sk-..." openenv-support
```

### Hugging Face Spaces (One-Click Deployment)

1. Create new Space: [hf.co/spaces/new](https://hf.co/spaces/new)
2. Choose "Docker" template
3. Upload this repository
4. Set environment variable: `OPENAI_API_KEY` in Space secrets
5. Space will auto-deploy at `https://username-spacename.hf.space`

---

## 📡 API Endpoints

### Health Check
```
GET /health
→ {"status": "healthy", "version": "1.0"}
```

### Reset Environment
```
POST /reset
Body: {"level": "easy", "session_id": "user123"}
→ {
    "observation": {...},
    "session_id": "user123",
    "level": "easy"
  }
```

### Step Environment
```
POST /step
Body: {
  "session_id": "user123",
  "action": {
    "assign_category": "billing",
    "set_priority": "high",
    "response": "We will process your refund immediately.",
    "escalate": false
  }
}
→ {
    "observation": {...},
    "reward": {"score": 0.85, "breakdown": {...}},
    "done": false,
    "info": {...},
    "state": {...}
  }
```

### Get Current State
```
POST /state
Body: {"session_id": "user123"}
→ {
    "step_count": 1,
    "resolved": false,
    "escalated": false,
    "total_reward": 0.85
  }
```

### Grade Episode
```
POST /grader
Body: {"session_id": "user123"}
→ {"episode_score": 0.82}
```

### List Tasks & Schema
```
GET /tasks
→ {
    "tasks": [
      {"name": "easy", "description": "..."},
      {"name": "medium", "description": "..."},
      {"name": "hard", "description": "..."}
    ],
    "action_schema": {...}
  }
```

### Run Baseline Agent
```
POST /baseline
→ {
    "baseline_scores": {
      "easy": 0.78,
      "medium": 0.65,
      "hard": 0.52
    },
    "overall": 0.65
  }
```

---

## 🧪 Pre-Submission Validation

Run this before submitting to ensure all requirements are met:

```bash
python validate.py
```

Expected output:
```
============================================================
OpenEnv Hackathon Pre-Submission Validator
============================================================

✓ Checking project structure...
  ✓ All directories and files present
✓ Validating openenv.yaml...
  ✓ openenv.yaml is valid
✓ Validating Pydantic models...
  ✓ All models importable
✓ Validating environment API...
  ✓ reset(), step(), state() working correctly
✓ Validating grader...
  ✓ Grader working correctly
✓ Validating Dockerfile...
  ✓ Dockerfile valid
✓ Validating README...
  ✓ README present

============================================================
✓ ALL CHECKS PASSED - Ready to submit!
============================================================
```

---

## 📊 Baseline Results

Run the baseline agent on all 3 tasks:

```bash
export OPENAI_API_KEY="sk-..."
python -m scripts.baseline
```

Expected output:
```
Baseline task scores:
  easy: 0.782
  medium: 0.654
  hard: 0.523
Overall score: 0.653
```

### What These Scores Mean

- **Easy (0.78)**: Agent correctly classifies obvious tickets, good responses
- **Medium (0.65)**: Agent handles ambiguity reasonably but makes some errors
- **Hard (0.52)**: Large gap between optimal and achieved; room for improvement

The gap between Easy and Hard demonstrates genuine task difficulty progression.

---

## 🔍 Example Interaction

### Ticket (MEDIUM difficulty)
```
Message: "The app keeps crashing and also I need to update my payment info. Never had these issues before."
User History: ["Account created 2 months ago", "Never reported issues before"]
Sentiment: frustrated
Previous Attempts: 1
```

### Agent's Action
```json
{
  "assign_category": "technical",
  "set_priority": "high",
  "response": "I'm sorry you're experiencing issues. Let me help with both problems: 1) Our team is aware of the crashes in the latest version—please try uninstalling and reinstalling. 2) You can update payment info in Settings > Billing. If issues persist, I'll escalate to our technical team.",
  "escalate": true
}
```

### Reward Breakdown
```
Category correctness (technical): 1.0 × 0.3 = 0.30
Priority correctness (high): 1.0 × 0.2 = 0.20
Response quality (has keywords, polite): 0.85 × 0.25 = 0.21
Escalation correctness: 1.0 × 0.25 = 0.25
Politeness bonus: +0.05
─────────────────────────
Total Score: 0.86
```

---

## 📚 OpenEnv Compliance

This environment strictly follows the OpenEnv specification:

✅ `reset()` → Observation
✅ `step(Action)` → (Observation, Reward, done, info)
✅ `state()` → State
✅ Pydantic models for all types
✅ Deterministic grader: `grade_episode(trajectory) → float`
✅ YAML metadata: `openenv.yaml`
✅ Dockerfile for deployment
✅ 3+ multi-level tasks with clear progression

---

## 🛠️ Project Structure

```
.
├── env/
│   ├── __init__.py
│   ├── models.py           # Pydantic data models
│   ├── environment.py      # SupportTicketEnv implementation
│   └── grader.py           # Episode grading logic
├── data/
│   └── tickets.py          # 18 realistic tickets
├── scripts/
│   ├── __init__.py
│   ├── baseline.py         # Baseline inference script
│   └── server.py           # Flask HTTP server
├── configs/                # Reserved for config files
├── openenv.yaml            # Environment metadata
├── Dockerfile              # Production container
├── README.md               # This file
└── validate.py             # Pre-submission validator
```

---

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
