from typing import Any, Dict, List

from .tasks import get_ticket_by_id


def grade_episode(trajectory: List[Dict[str, Any]]) -> float:
    """Grade an episode trajectory deterministically.

    trajectory: list of dicts {"observation": Observation, "action": Action, "reward": Reward}
    """
    if not trajectory:
        return 0.0

    total_score = 0.0
    for step in trajectory:
        reward = step.get("reward")
        if reward is None or not isinstance(reward, dict):
            continue
        score = reward.get("score")
        if score is not None:
            total_score += float(score)

    avg_score = total_score / len(trajectory)

    # normalize from [-1,1] to [0,1]
    normalized = (avg_score + 1.0) / 2.0
    normalized = max(0.0, min(1.0, normalized))

    # add separate task-based measure by comparing predicted fields at final step
    last_step = trajectory[-1]
    obs = last_step.get("observation")
    act = last_step.get("action")
    if obs and act:
        ticket = get_ticket_by_id(obs.get("ticket_id") if isinstance(obs, dict) else getattr(obs, "ticket_id", None))
        if ticket and isinstance(act, dict):
            category_ok = 1.0 if act.get("assign_category") == ticket.expected_category else 0.0
            priority_ok = 1.0 if act.get("set_priority") == ticket.expected_priority else 0.0
            # response keywords
            response = (act.get("response") or "").lower()
            matched = sum(1 for kw in ticket.expected_keywords if kw.lower() in response)
            response_score = matched / max(1, len(ticket.expected_keywords))
            normalized = 0.5 * normalized + 0.5 * ((0.4 * category_ok + 0.3 * priority_ok + 0.3 * response_score))

    return float(max(0.0, min(1.0, normalized)))
