from typing import Any, Dict, List

from .tasks import get_ticket_by_id

EPSILON = 0.01


def normalize_score(score: float) -> float:
    """Keep platform-facing task scores strictly inside (0.01, 0.99)."""
    return float(min(0.99, max(EPSILON, score)))


def grade_episode(trajectory: List[Dict[str, Any]]) -> float:
    """Grade an episode trajectory deterministically.

    trajectory: list of dicts {"observation": Observation, "action": Action, "reward": Reward}
    """
    if not trajectory:
        return normalize_score(0.5)

    total_score = 0.0
    for step in trajectory:
        reward = step.get("reward")
        if reward is None:
            continue
        
        # Handle both dict and Reward object
        if isinstance(reward, dict):
            score = reward.get("score")
        else:
            score = getattr(reward, "score", None)
        
        if score is not None:
            total_score += float(score)

    avg_score = total_score / len(trajectory)

    # Start with average step score
    final_score = avg_score

    # Add task-based measure by comparing predicted fields at final step
    last_step = trajectory[-1]
    obs = last_step.get("observation")
    act = last_step.get("action")
    
    if obs and act:
        ticket = None
        
        # Extract ticket_id from observation
        ticket_id = None
        if isinstance(obs, dict):
            ticket_id = obs.get("ticket_id")
        else:
            ticket_id = getattr(obs, "ticket_id", None)
        
        if ticket_id:
            ticket = get_ticket_by_id(ticket_id)
        
        if ticket and isinstance(act, dict):
            # Evaluate final action against expected values
            category_ok = 1.0 if act.get("assign_category") == ticket.expected_category else 0.0
            priority_ok = 1.0 if act.get("set_priority") == ticket.expected_priority else 0.0
            
            # Response keyword matching
            response = (act.get("response") or "").lower()
            matched = sum(1 for kw in ticket.expected_keywords if kw.lower() in response)
            response_score = matched / max(1, len(ticket.expected_keywords))
            
            # Combine with step-by-step average
            task_score = 0.3 * category_ok + 0.2 * priority_ok + 0.5 * response_score
            final_score = 0.5 * final_score + 0.5 * task_score

    return normalize_score(float(final_score))
