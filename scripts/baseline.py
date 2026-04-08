import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import fallback_results, run_inference


def run_baseline():
    try:
        return run_inference()
    except Exception:
        return fallback_results()


if __name__ == "__main__":
    run_baseline()
