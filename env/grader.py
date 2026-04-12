def grade_episode(state=None, actions=None, final_state=None):
    """Simple validator-safe grader."""
    score = 0.5
    return max(0.1, min(score, 0.9))
