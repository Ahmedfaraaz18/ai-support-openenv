from env.environment import SupportTicketEnv
from env.models import Action


env = SupportTicketEnv()

# Step 1: Reset
obs = env.reset()
print("Initial Observation:", obs)

# Step 2: Take action
action = Action(
    assign_category="billing",
    set_priority="high",
    response="We are sorry, we will resolve your issue."
)

obs2, reward, done, info = env.step(action.model_dump())
print("Next Observation:", obs2)
print("Reward:", reward)
print("Done:", done)
print("Info:", info)
print("Total Reward:", env._state.total_reward)
print("State:", env._state)
print(" Current Ticket:", env.current_ticket)
print("Expected Category:", env.current_ticket.expected_category)
print("Expected Priority:", env.current_ticket.expected_priority)
