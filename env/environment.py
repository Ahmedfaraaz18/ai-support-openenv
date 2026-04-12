import random
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError

from .models import Action, Observation, Reward, State
from .tasks import Ticket, get_ticket_by_id, get_tickets


def normalize_score(score: float) -> float:
    """Normalize score to strictly stay within (0.1, 0.9)."""
    return max(0.1, min(score, 0.9))


class SupportTicketEnv:
    VALID_CATEGORIES = {"billing", "technical", "account", "other"}
    VALID_PRIORITIES = {"low", "medium", "high"}

    def __init__(self, level: str = "easy", seed: int = 42, task: Optional[str] = None):
        selected_level = task if task is not None else level
        self.level = selected_level
        self.seed = seed
        self.rng = random.Random(seed)
        self.tickets: List[Ticket] = get_tickets(selected_level)
        if not self.tickets:
            raise ValueError(f"No tickets found for level '{selected_level}'")
        self.current_ticket: Optional[Ticket] = None
        self._state = State(step_count=0, ticket_resolved=False, total_reward=0.0)
        self._last_action_invalid: bool = False
        self._ticket_pool = self.tickets.copy()

    def reset(self) -> Observation:
        if not self._ticket_pool:
            self._ticket_pool = self.tickets.copy()
        self.current_ticket = self.rng.choice(self._ticket_pool)
        # remove chosen randomly to avoid immediate repeats in same epoch
        self._ticket_pool.remove(self.current_ticket)
        self._state = State(step_count=0, ticket_resolved=False, total_reward=0.0)
        self._last_action_invalid = False
        return self._build_observation(self.current_ticket)

    def step(self, action: Dict[str, str]) -> Tuple[Observation, Reward, bool, Dict]:
        if self.current_ticket is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        try:
            action_model = Action(**action)
            invalid_action = False
            invalid_reasons: List[str] = []
        except (ValidationError, ValueError) as e:
            # action is invalid
            action_model = None
            invalid_action = True
            invalid_reasons = ["invalid_action_format"]

        self._state.step_count += 1

        ticket = self.current_ticket
        expected_category = ticket.expected_category
        expected_priority = ticket.expected_priority

        category_score = 0.0
        priority_score = 0.0
        response_score = 0.0
        penalty = 0.0

        if invalid_action:
            # Invalid action penalties
            penalty = 0.3  # empty response penalty
            if self._last_action_invalid:
                penalty += 0.1  # repeated invalid penalty
            self._last_action_invalid = True
            
            # Severely penalize invalid actions
            base_score = -0.5
            reward_score = normalize_score(base_score - penalty)
            
            reward = Reward(
                score=reward_score, 
                breakdown={
                    "category": 0.0, 
                    "priority": 0.0, 
                    "response": 0.0, 
                    "penalty": penalty
                }
            )
            done = self._state.step_count >= 3
            self._state.total_reward += reward_score
            return self._build_observation(ticket), reward, done, {
                "error": "invalid action", 
                "invalid_reasons": invalid_reasons
            }

        # Valid action model exists
        
        if action_model.assign_category in self.VALID_CATEGORIES:
            if action_model.assign_category == expected_category:
                category_score = 1.0
            elif expected_category in ["billing", "account"] and action_model.assign_category in ["billing", "account"]:
                category_score = 0.5  # Partial credit for related categories
            else:
                category_score = 0.0
                penalty += 0.2  # penalty for wrong category
        else:
            category_score = 0.0
            penalty += 0.2  # invalid category value
            invalid_reasons.append("invalid_category")

        if action_model.set_priority in self.VALID_PRIORITIES:
            if action_model.set_priority == expected_priority:
                priority_score = 1.0
            elif action_model.set_priority == "medium" and expected_priority in {"low", "high"}:
                priority_score = 0.5  # Partial credit for reasonable middle ground
            else:
                priority_score = 0.0
                penalty += 0.2  # penalty for wrong priority
        else:
            priority_score = 0.0
            penalty += 0.2  # invalid priority value
            invalid_reasons.append("invalid_priority")

        normalized_response = action_model.response.lower().strip()
        if not normalized_response:
            penalty += 0.3  # empty response penalty
            invalid_reasons.append("empty_response")
        else:
            # Response quality: percentage of expected keywords matched
            matched = sum(1 for kw in ticket.expected_keywords if kw.lower() in normalized_response)
            response_score = min(1.0, matched / max(1, len(ticket.expected_keywords)))
        
        # Repeated invalid penalty
        if invalid_reasons and self._last_action_invalid:
            penalty += 0.1
        
        self._last_action_invalid = bool(invalid_reasons)

        # Weighted scoring: category 30%, priority 20%, response 50%
        base_score = 0.3 * category_score + 0.2 * priority_score + 0.5 * response_score

        difficulty_penalty = {
            "easy": 0.1,
            "medium": 0.3,
            "hard": 0.5,
        }.get(self.level, 0.0)

        # Apply penalty and difficulty adjustment so tasks differ by level
        total_score = base_score - penalty - difficulty_penalty

        # Clamp to valid range and normalize
        reward_score = normalize_score(total_score)

        reward_breakdown = {
            "category": category_score,
            "priority": priority_score,
            "response": response_score,
            "penalty": penalty,
        }

        reward = Reward(score=reward_score, breakdown=reward_breakdown)
        self._state.total_reward += reward_score

        # Resolved when conditions met
        self._state.ticket_resolved = (
            category_score >= 1.0
            and priority_score >= 1.0
            and response_score >= 0.6
            and not invalid_reasons
        )

        done = self._state.ticket_resolved or self._state.step_count >= 3

        info = {
            "expected_category": expected_category,
            "expected_priority": expected_priority,
            "matched_keywords": response_score,
            "invalid_reasons": invalid_reasons,
            "step_count": self._state.step_count,
        }

        return self._build_observation(ticket), reward, done, info

    def state(self) -> State:
        return self._state

    def _build_observation(self, ticket: Ticket) -> Observation:
        return Observation(
            ticket_id=ticket.ticket_id,
            message=ticket.message,
            user_history=ticket.user_history,
            current_status=ticket.current_status,
            urgency_hint=ticket.urgency_hint,
        )
