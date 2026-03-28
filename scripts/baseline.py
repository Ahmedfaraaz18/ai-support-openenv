import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import main as run_baseline


if __name__ == "__main__":
    run_baseline()
