#!/usr/bin/env python
"""
Pre-submission validation script for OpenEnv hackathon.
Run this before submitting to ensure compliance.
"""

import sys
import yaml
from pathlib import Path

def validate_project_structure():
    """Check project structure."""
    print("✓ Checking project structure...")
    required_dirs = ["env", "data", "scripts", "configs"]
    required_files = [
        "openenv.yaml",
        "Dockerfile",
        "README.md",
        "env/models.py",
        "env/environment.py",
        "env/grader.py",
        "data/tickets.py",
        "scripts/baseline.py",
        "scripts/server.py",
    ]

    for d in required_dirs:
        if not Path(d).exists():
            print(f"  ✗ Missing directory: {d}")
            return False

    for f in required_files:
        if not Path(f).exists():
            print(f"  ✗ Missing file: {f}")
            return False

    print("  ✓ All directories and files present")
    return True


def validate_openenv_yaml():
    """Validate openenv.yaml structure."""
    print("✓ Validating openenv.yaml...")
    try:
        with open("openenv.yaml", "r") as f:
            config = yaml.safe_load(f)

        required_keys = ["name", "version", "description", "observation_space", "action_space", "reward_range", "tasks"]
        for key in required_keys:
            if key not in config:
                print(f"  ✗ Missing key in openenv.yaml: {key}")
                return False

        if not isinstance(config["tasks"], list) or len(config["tasks"]) < 3:
            print("  ✗ Must have at least 3 tasks")
            return False

        print("  ✓ openenv.yaml is valid")
        return True
    except Exception as e:
        print(f"  ✗ Error validating openenv.yaml: {e}")
        return False


def validate_pydantic_models():
    """Check Pydantic models are importable."""
    print("✓ Validating Pydantic models...")
    try:
        from env.models import Observation, Action, Reward, State
        print("  ✓ All models importable")
        return True
    except Exception as e:
        print(f"  ✗ Error importing models: {e}")
        return False


def validate_environment():
    """Test environment reset/step/state."""
    print("✓ Validating environment API...")
    try:
        from env.environment import SupportTicketEnv

        env = SupportTicketEnv(level="easy", seed=42)
        obs = env.reset()

        if obs is None:
            print("  ✗ reset() returned None")
            return False

        action = {
            "assign_category": "billing",
            "set_priority": "high",
            "response": "We will resolve this.",
            "escalate": False,
        }
        obs2, reward, done, info = env.step(action)

        if reward is None:
            print("  ✗ step() returned None reward")
            return False

        if reward.score < -1.0 or reward.score > 1.0:
            print(f"  ✗ Reward out of range: {reward.score}")
            return False

        state = env.state()
        if state is None:
            print("  ✗ state() returned None")
            return False

        print("  ✓ reset(), step(), state() working correctly")
        return True
    except Exception as e:
        print(f"  ✗ Error testing environment: {e}")
        return False


def validate_grader():
    """Test grader function."""
    print("✓ Validating grader...")
    try:
        from env.grader import grade_episode

        trajectory = [
            {
                "observation": {"ticket_id": 1},
                "action": {"assign_category": "billing"},
                "reward": {"score": 0.8, "breakdown": {}},
            }
        ]
        score = grade_episode(trajectory)

        if not isinstance(score, float) or score < 0.0 or score > 1.0:
            print(f"  ✗ Grader returned invalid score: {score}")
            return False

        print("  ✓ Grader working correctly")
        return True
    except Exception as e:
        print(f"  ✗ Error testing grader: {e}")
        return False


def validate_dockerfile():
    """Check Dockerfile exists and has key features."""
    print("✓ Validating Dockerfile...")
    try:
        with open("Dockerfile", "r") as f:
            content = f.read()

        if "python" not in content.lower():
            print("  ✗ Dockerfile missing Python")
            return False

        if "7860" not in content and "8080" not in content:
            print("  ✗ Dockerfile not exposing port 7860 or 8080")
            return False

        print("  ✓ Dockerfile valid")
        return True
    except Exception as e:
        print(f"  ✗ Error validating Dockerfile: {e}")
        return False


def validate_readme():
    """Check README exists and has required sections."""
    print("✓ Validating README...")
    try:
        with open("README.md", "r") as f:
            content = f.read()

        required_sections = ["Overview", "Tasks", "Reward", "Deployment"]
        for section in required_sections:
            if section.lower() not in content.lower():
                print(f"  ! WARNING: Missing section '{section}' in README")

        print("  ✓ README present")
        return True
    except Exception as e:
        print(f"  ✗ Error validating README: {e}")
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
        validate_readme,
    ]

    results = [check() for check in checks]

    print("\n" + "=" * 60)
    if all(results):
        print("✓ ALL CHECKS PASSED - Ready to submit!")
        print("=" * 60 + "\n")
        return 0
    else:
        print("✗ Some checks failed - Please fix issues above")
        print("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
