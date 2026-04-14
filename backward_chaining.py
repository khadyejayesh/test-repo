# Step 1: Define known facts
facts = {"rain"}   # Set of known true facts

# Step 2: Define rules
rules = [
  ({"rain"}, "wet_ground"),
  ({"wet_ground"}, "slippery")]

# Step 3: Define backward chaining function
def backward_chaining(goal, facts, rules, visited=None):
    # Initialize visited set (for recursion safety)
    if visited is None:
        visited = set()
    print("Trying to prove:", goal)

    # If goal is already a known fact, return True
    if goal in facts:
        print(goal, "is a known fact.")
        return True

    # Avoid infinite loops
    if goal in visited:
        return False

    visited.add(goal)
    for conditions, conclusion in rules:
        if conclusion == goal:
            print("Found rule:", conditions, "→", conclusion)

            # Try to prove all conditions recursively
            all_proved = True
            for condition in conditions:
                if not backward_chaining(condition, facts, rules, visited):
                    all_proved = False
                    break

            # If all conditions are proven
            if all_proved:
                print("Therefore,", goal, "is TRUE\n")
                return True

    # If no rule proves the goal
    print("Cannot prove:", goal, "\n")
    return False

# Step 4: Test the system
goal = "slippery"
result = backward_chaining(goal, facts, rules)
print("Final Result:", goal, "=", result)
