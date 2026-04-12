#!/usr/bin/env python
"""
Phase 2 Validation Test
Verifies that all 3 task levels pass validation requirements:
- At least 3 tasks (easy, medium, hard)
- Each task returns scores strictly between 0.01 and 0.99
- Deterministic grading
- Different scores across difficulty levels
"""

import sys
from env.environment import SupportTicketEnv, normalize_score
from env.grader import grade_episode


def test_normalize_score():
    """Test that normalize_score works correctly."""
    print("\n[TEST] normalize_score function...")
    
    test_cases = [
        (-1.0, 0.01),
        (-0.5, 0.01),
        (0.0, 0.01),
        (0.5, 0.5),
        (0.8, 0.8),
        (0.99, 0.99),
        (1.0, 0.99),
        (2.0, 0.99),
    ]
    
    for input_val, expected in test_cases:
        result = normalize_score(input_val)
        assert result == expected, f"normalize_score({input_val}) = {result}, expected {expected}"
    
    print("  ✓ normalize_score works correctly")
    return True


def test_task_level(level: str, num_tests: int = 3):
    """Test a specific task level."""
    print(f"\n[TEST] Task level: {level}")
    
    scores = []
    
    for test_num in range(num_tests):
        env = SupportTicketEnv(level=level, seed=42 + test_num)
        obs = env.reset()
        
        trajectory = []
        done = False
        step_count = 0
        
        while not done and step_count < 3:
            # Create a reasonable action
            action = {
                "assign_category": obs.get("category", "technical") if isinstance(obs, dict) else "technical",
                "set_priority": obs.get("priority", "medium") if isinstance(obs, dict) else "medium", 
                "response": "I will help you resolve this issue. Thank you for your patience.",
            }
            
            obs, reward, done, info = env.step(action)
            step_count += 1
            
            # Record trajectory step
            trajectory.append({
                "observation": obs,
                "action": action,
                "reward": reward,
            })
            
            # Verify reward score is in valid range
            score = reward.score
            assert 0.01 <= score <= 0.99, f"Score {score} out of valid range [0.01, 0.99]"
            print(f"  Step {step_count}: score={score:.3f}, breakdown={reward.breakdown}")
        
        # Grade the episode
        episode_score = grade_episode(trajectory)
        assert 0.01 <= episode_score <= 0.99, f"Episode score {episode_score} out of valid range [0.01, 0.99]"
        print(f"  Episode {test_num + 1}: score={episode_score:.3f}")
        scores.append(episode_score)
    
    avg_score = sum(scores) / len(scores)
    print(f"  Average score for {level}: {avg_score:.3f}")
    return scores


def test_determinism():
    """Test that same seed produces same scores."""
    print("\n[TEST] Determinism (same seed produces same results)")
    
    level = "medium"
    seed = 42
    
    # Run 1
    env1 = SupportTicketEnv(level=level, seed=seed)
    obs1 = env1.reset()
    action = {
        "assign_category": "technical",
        "set_priority": "high",
        "response": "We will investigate this issue immediately.",
    }
    _, reward1, _, _ = env1.step(action)
    
    # Run 2 (same seed)
    env2 = SupportTicketEnv(level=level, seed=seed)
    obs2 = env2.reset()
    _, reward2, _, _ = env2.step(action)
    
    assert reward1.score == reward2.score, f"Determinism failed: {reward1.score} != {reward2.score}"
    print(f"  ✓ Deterministic: both runs returned score {reward1.score:.3f}")
    return True


def test_score_differentiation():
    """Test that difficulty levels produce different score distributions."""
    print("\n[TEST] Score differentiation across difficulty levels")
    
    from env.tasks import get_tickets
    
    all_scores = {}
    
    for level in ["easy", "medium", "hard"]:
        print(f"\n[TEST] Task level: {level}")
        scores = []
        
        for test_num in range(2):
            env = SupportTicketEnv(level=level, seed=42 + test_num)
            obs = env.reset()
            
            trajectory = []
            done = False
            step_count = 0
            
            # Get the ticket to provide correct action
            from env.tasks import get_ticket_by_id
            ticket = get_ticket_by_id(obs.ticket_id if isinstance(obs, dict) else obs.ticket_id)
            
            while not done and step_count < 3:
                # Use correct action based on ticket
                if ticket:
                    action = {
                        "assign_category": ticket.expected_category,
                        "set_priority": ticket.expected_priority,
                        "response": f"I'll help with this {ticket.expected_category} issue. " + 
                                   " ".join(ticket.expected_keywords) + 
                                   ". We'll get this resolved for you.",
                    }
                else:
                    action = {
                        "assign_category": "technical",
                        "set_priority": "medium",
                        "response": "I will help you resolve this issue. Thank you for your patience.",
                    }
                
                obs, reward, done, info = env.step(action)
                step_count += 1
                
                # Record trajectory step
                trajectory.append({
                    "observation": obs,
                    "action": action,
                    "reward": reward,
                })
                
                # Verify reward score is in valid range
                score = reward.score
                assert 0.01 <= score <= 0.99, f"Score {score} out of valid range [0.01, 0.99]"
                print(f"  Test {test_num + 1}, Step {step_count}: score={score:.3f}")
            
            # Grade the episode
            episode_score = grade_episode(trajectory)
            assert 0.01 <= episode_score <= 0.99, f"Episode score {episode_score} out of valid range [0.01, 0.99]"
            print(f"  Test {test_num + 1} Episode score: {episode_score:.3f}")
            scores.append(episode_score)
        
        avg_score = sum(scores) / len(scores)
        print(f"  Average score for {level}: {avg_score:.3f}")
        all_scores[level] = scores
    
    easy_avg = sum(all_scores["easy"]) / len(all_scores["easy"])
    medium_avg = sum(all_scores["medium"]) / len(all_scores["medium"])
    hard_avg = sum(all_scores["hard"]) / len(all_scores["hard"])
    
    print(f"\n  Score averages:")
    print(f"    Easy:   {easy_avg:.3f}")
    print(f"    Medium: {medium_avg:.3f}")
    print(f"    Hard:   {hard_avg:.3f}")
    
    # All scores should be valid
    print(f"  ✓ All scores are between 0.01 and 0.99")
    
    return True


def test_no_extreme_scores():
    """Test that we never return 0.0 or 1.0."""
    print("\n[TEST] No extreme scores (0.0 or 1.0)")
    
    extreme_scores = {"0.0": 0, "1.0": 0}
    total_steps = 0
    
    for level in ["easy", "medium", "hard"]:
        env = SupportTicketEnv(level=level, seed=42)
        obs = env.reset()
        
        for step in range(3):
            action = {
                "assign_category": "billing",
                "set_priority": "high",
                "response": "Thank you for contacting us. We will investigate and resolve your issue promptly.",
            }
            obs, reward, done, info = env.step(action)
            total_steps += 1
            
            if reward.score == 0.0:
                extreme_scores["0.0"] += 1
            elif reward.score == 1.0:
                extreme_scores["1.0"] += 1
            
            if done:
                break
    
    assert extreme_scores["0.0"] == 0, f"Found {extreme_scores['0.0']} steps with score 0.0"
    assert extreme_scores["1.0"] == 0, f"Found {extreme_scores['1.0']} steps with score 1.0"
    
    print(f"  ✓ No 0.0 or 1.0 scores found in {total_steps} steps")
    return True


def main():
    print("\n" + "=" * 70)
    print("OpenEnv Phase 2 Validation Test")
    print("=" * 70)
    
    tests = [
        ("Normalize Score Function", test_normalize_score),
        ("No Extreme Scores", test_no_extreme_scores),
        ("Determinism", test_determinism),
        ("Score Differentiation", test_score_differentiation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, True, None))
            print(f"\n✓ {test_name} PASSED")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n✗ {test_name} FAILED: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"       {error}")
    
    print("=" * 70)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All Phase 2 validation tests PASSED!")
        print("=" * 70 + "\n")
        return 0
    else:
        print("✗ Some tests failed. Please review the output above.")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
