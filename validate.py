#!/usr/bin/env python
"""
Pre-submission validation script for the OpenEnv environment.
Run this before submitting to ensure the repo is structurally complete.
"""

from pathlib import Path
import sys

import yaml


OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def validate_project_structure():
    """Check required directories and files."""
    print(f"{OK} Checking project structure...")
    required_dirs = ["env", "data", "scripts", "configs"]
    required_files = [
        "openenv.yaml",
        "Dockerfile",
        "README.md",
        "env/models.py",
        "env/environment.py",
        "env/grader.py",
        "env/tasks.py",
        "scripts/baseline.py",
        "scripts/server.py",
    ]

    for directory in required_dirs:
        if not Path(directory).exists():
            print(f"  {FAIL} Missing directory: {directory}")
            return False

    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"  {FAIL} Missing file: {file_path}")
            return False

    print(f"  {OK} All directories and files present")
    return True


def validate_openenv_yaml():
    """Validate openenv.yaml structure."""
    print(f"{OK} Validating openenv.yaml...")
    try:
        with open("openenv.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        required_keys = [
            "name",
            "version",
            "description",
            "observation_space",
            "action_space",
            "reward_range",
            "tasks",
        ]
        for key in required_keys:
            if key not in config:
                print(f"  {FAIL} Missing key in openenv.yaml: {key}")
                return False

        if not isinstance(config["tasks"], list) or len(config["tasks"]) < 3:
            print(f"  {FAIL} Must have at least 3 tasks")
            return False

        if config["reward_range"] != [-1.0, 1.0]:
            print(f"  {FAIL} reward_range should be [-1.0, 1.0]")
            return False

        print(f"  {OK} openenv.yaml is valid")
        return True
    except Exception as exc:
        print(f"  {FAIL} Error validating openenv.yaml: {exc}")
        return False


def validate_pydantic_models():
    """Check typed models are importable."""
    print(f"{OK} Validating typed models...")
    try:
        from env.models import Action, Observation, Reward, State

        _ = Action, Observation, Reward, State
        print(f"  {OK} All models importable")
        return True
    except Exception as exc:
        print(f"  {FAIL} Error importing models: {exc}")
        return False


def validate_environment():
    """Test environment reset/step/state."""
    print(f"{OK} Validating environment API...")
    try:
        from env.environment import SupportTicketEnv

        env = SupportTicketEnv(level="easy", seed=42)
        obs = env.reset()
        if obs is None:
            print(f"  {FAIL} reset() returned None")
            return False

        action = {
            "assign_category": "billing",
            "set_priority": "high",
            "response": "We will review the invoice and help resolve the charge.",
        }
        _, reward, _, _ = env.step(action)
        if reward is None:
            print(f"  {FAIL} step() returned None reward")
            return False

        if reward.score < -1.0 or reward.score > 1.0:
            print(f"  {FAIL} Reward out of range: {reward.score}")
            return False

        state = env.state()
        if state is None:
            print(f"  {FAIL} state() returned None")
            return False

        if not hasattr(state, "ticket_resolved"):
            print(f"  {FAIL} State model missing ticket_resolved")
            return False

        print(f"  {OK} reset(), step(), state() working correctly")
        return True
    except Exception as exc:
        print(f"  {FAIL} Error testing environment: {exc}")
        return False


def validate_grader():
    """Test grader function."""
    print(f"{OK} Validating grader...")
    try:
        from env.grader import grade_episode

        trajectory = [
            {
                "observation": {"ticket_id": 1},
                "action": {
                    "assign_category": "billing",
                    "set_priority": "high",
                    "response": "We will investigate the charge and help resolve it.",
                },
                "reward": {"score": 0.8, "breakdown": {}},
            }
        ]
        score = grade_episode(trajectory)
        if not isinstance(score, float) or score < 0.0 or score > 1.0:
            print(f"  {FAIL} Grader returned invalid score: {score}")
            return False

        print(f"  {OK} Grader working correctly")
        return True
    except Exception as exc:
        print(f"  {FAIL} Error testing grader: {exc}")
        return False


def validate_dockerfile():
    """Check Dockerfile exists and has key deployment features."""
    print(f"{OK} Validating Dockerfile...")
    try:
        with open("Dockerfile", "r", encoding="utf-8") as handle:
            content = handle.read()

        if "python" not in content.lower():
            print(f"  {FAIL} Dockerfile missing Python")
            return False

        if "7860" not in content and "8080" not in content:
            print(f"  {FAIL} Dockerfile not exposing port 7860 or 8080")
            return False

        if "requirements.txt" not in content:
            print(f"  {WARN} Dockerfile does not install from requirements.txt")
        else:
            print(f"  {OK} Dockerfile installs dependencies from requirements.txt")

        print(f"  {OK} Dockerfile valid")
        return True
    except Exception as exc:
        print(f"  {FAIL} Error validating Dockerfile: {exc}")
        return False


def validate_baseline():
    """Check baseline script is importable."""
    print(f"{OK} Validating baseline script...")
    try:
        from scripts.baseline import run_baseline

        if not callable(run_baseline):
            print(f"  {FAIL} run_baseline is not callable")
            return False

        print(f"  {OK} Baseline script is importable")
        return True
    except Exception as exc:
        print(f"  {FAIL} Error validating baseline script: {exc}")
        return False


def validate_readme():
    """Check README includes the required compliance sections."""
    print(f"{OK} Validating README...")
    try:
        with open("README.md", "r", encoding="utf-8") as handle:
            content = handle.read()

        required_sections = [
            "Overview",
            "Observation Space",
            "Action Space",
            "Reward Function",
            "Setup",
            "Docker and Hugging Face Spaces",
        ]
        for section in required_sections:
            if section.lower() not in content.lower():
                print(f"  {FAIL} Missing section '{section}' in README")
                return False

        print(f"  {OK} README present and includes required sections")
        return True
    except Exception as exc:
        print(f"  {FAIL} Error validating README: {exc}")
        return False


def main():
    print("\n" + "=" * 60)
    print("OpenEnv Hackathon Pre-Submission Validator")
    print("=" * 60 + "\n")

    checks = [
        validate_project_structure,
        validate_openenv_yaml,
        validate_pydantic_models,
        validate_environment,
        validate_grader,
        validate_dockerfile,
        validate_baseline,
        validate_readme,
    ]

    results = [check() for check in checks]

    print("\n" + "=" * 60)
    if all(results):
        print(f"{OK} ALL CHECKS PASSED - Ready to submit!")
        print("=" * 60 + "\n")
        return 0

    print(f"{FAIL} Some checks failed - Please fix issues above")
    print("=" * 60 + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
