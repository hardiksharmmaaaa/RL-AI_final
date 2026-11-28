# Hardik's Project Work — COSC 604 Reinforcement Learning

**Project:** Pacman Reinforcement Learning Project  
**Course:** COSC 604 Artificial Intelligence | Fall 2026  
**Instructor:** Dr. Andreas Henschel  
**Collaborator:** Damjan  
**Status:** All assigned questions complete and passing autograder

---

## What I Was Assigned (5 Questions)

| Q# | Topic | Points | File | Status |
|----|-------|--------|------|--------|
| Q1 | Value Iteration | 5 | `submission/valueIterationAgents.py` | ✅ 5/5 |
| Q4 | Prioritized Sweeping (extra credit) | +1 | `submission/valueIterationAgents.py` | ✅ 1/0 |
| Q5 | Q-Learning core methods | 5 | `submission/qlearningAgents.py` | ✅ 5/5 |
| Q8 | Q-Learning + Pacman | 1 | `submission/qlearningAgents.py` | ✅ 1/1 |
| Q2 | Bridge Crossing Analysis | 1 | `submission/analysis.py` | ✅ 1/1 |

**Total: 13 points** (12 regular + 1 extra credit)

---

## Q1 — Value Iteration

**File:** `submission/valueIterationAgents.py`  
**Concept:** Value iteration is an offline planning algorithm. The agent knows the MDP (the rules of the game) and computes the best policy by thinking ahead before taking any action.

### The Value Iteration Update Formula

```
V(s) = max over actions a of  Σ over next states s' of  P(s'|s,a) × [R(s,a,s') + γ × V(s')]
```

In plain English: "The value of a state is the best expected reward you can get, considering all possible actions, all possible outcomes of each action, their probabilities, the immediate reward, and the discounted future value."

---

### Function 1: `runValueIteration`

```python
def runValueIteration(self):
    for i in range(self.iterations):
        new_values = util.Counter()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                new_values[state] = 0.0
                continue
            best_q = float('-inf')
            for action in self.mdp.getPossibleActions(state):
                q = self.computeQValueFromValues(state, action)
                if q > best_q:
                    best_q = q
            new_values[state] = best_q
        self.values = new_values
```

**Line-by-line meaning:**

| Line | Meaning |
|------|---------|
| `for i in range(self.iterations):` | Repeat the value update `iterations` times (e.g., 100). Each pass improves the value estimates. |
| `new_values = util.Counter()` | Create a fresh dictionary (default 0) to hold the NEW values computed from the OLD values. This is the **batch** version — we don't update in place. |
| `for state in self.mdp.getStates():` | Loop over every state in the grid (every square). |
| `if self.mdp.isTerminal(state):` | If the state is a terminal state (the game ends here), its value is 0 (no future rewards). |
| `new_values[state] = 0.0` | Set terminal state value to 0. |
| `continue` | Skip to the next state — terminal states have no actions to evaluate. |
| `best_q = float('-inf')` | Start with negative infinity so any real Q-value will be larger. |
| `for action in self.mdp.getPossibleActions(state):` | Try every legal action from this state (e.g., North, South, East, West). |
| `q = self.computeQValueFromValues(state, action)` | Compute the Q-value of taking this action in this state using current values. |
| `if q > best_q:` | Track the maximum Q-value across all actions. |
| `best_q = q` | Update best_q if we found a better action. |
| `new_values[state] = best_q` | The value of this state = the best Q-value across all actions. |
| `self.values = new_values` | After computing all new values from old values, replace the old with the new. |

**Why batch (not in-place):** The spec says to use the batch version where each V vector is computed from a fixed V vector from the previous iteration. If we updated in place, some states would use newly computed values while others use old values, which is incorrect for value iteration.

---

### Function 2: `computeQValueFromValues`

```python
def computeQValueFromValues(self, state, action):
    q_value = 0.0
    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
        reward = self.mdp.getReward(state, action, next_state)
        q_value += prob * (reward + self.discount * self.values[next_state])
    return q_value
```

**Line-by-line meaning:**

| Line | Meaning |
|------|---------|
| `q_value = 0.0` | Initialize the Q-value accumulator to 0. |
| `for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):` | For each possible outcome of taking this action: `next_state` is where you might end up, `prob` is the probability of ending up there (e.g., 80% go north, 10% go east, 10% go west due to noise). |
| `reward = self.mdp.getReward(state, action, next_state)` | Get the immediate reward for transitioning from state → next_state via this action. |
| `q_value += prob * (reward + self.discount * self.values[next_state])` | Add: probability × (immediate reward + discount × value of next state). The discount (γ=0.9) makes future rewards worth less than immediate rewards. |
| `return q_value` | Return the total expected Q-value. |

**Q-value formula:** Q(s,a) = Σ P(s'|s,a) × [R(s,a,s') + γ × V(s')]

---

### Function 3: `computeActionFromValues`

```python
def computeActionFromValues(self, state):
    possible_actions = self.mdp.getPossibleActions(state)
    if not possible_actions:
        return None
    best_action = None
    best_q = float('-inf')
    for action in possible_actions:
        q = self.computeQValueFromValues(state, action)
        if q > best_q:
            best_q = q
            best_action = action
    return best_action
```

**Line-by-line meaning:**

| Line | Meaning |
|------|---------|
| `possible_actions = self.mdp.getPossibleActions(state)` | Get all legal actions from this state. |
| `if not possible_actions:` | If there are no legal actions (terminal state), return None. |
| `return None` | Terminal states have no policy action. |
| `best_action = None` | Track which action is currently best. |
| `best_q = float('-inf')` | Track the best Q-value found so far. |
| `for action in possible_actions:` | Try every legal action. |
| `q = self.computeQValueFromValues(state, action)` | Compute Q-value for this action. |
| `if q > best_q:` | If this action is better than the current best... |
| `best_q = q` | Update best Q-value. |
| `best_action = action` | Remember this action as the best. |
| `return best_action` | Return the action with the highest Q-value. |

---

## Q4 — Prioritized Sweeping Value Iteration (Extra Credit)

**File:** `submission/valueIterationAgents.py`  
**Concept:** Prioritized sweeping is smarter than regular value iteration. Instead of updating ALL states every iteration, it only updates states whose values are likely to change. It uses a priority queue to focus on states with the biggest errors first.

**Key ideas:**
- **Predecessors**: States that can reach a given state. If state A can transition to state B, then A is a predecessor of B. When B's value changes, A might need updating too.
- **Priority queue**: States with the largest value difference (error) get updated first.
- **Theta (θ)**: A threshold. Only push a predecessor into the queue if its value difference is greater than θ.

---

### Q4 Implementation Line-by-Line

```python
def runValueIteration(self):
    # Step 1: Compute predecessors of all states
    predecessors = {}
    for state in self.mdp.getStates():
        predecessors[state] = set()

    for state in self.mdp.getStates():
        if self.mdp.isTerminal(state):
            continue
        for action in self.mdp.getPossibleActions(state):
            for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                if prob > 0:
                    predecessors[next_state].add(state)

    # Step 2: Initialize empty priority queue
    pq = util.PriorityQueue()

    # Step 3: For each non-terminal state, compute diff and push
    for state in self.mdp.getStates():
        if self.mdp.isTerminal(state):
            continue
        best_q = float('-inf')
        for action in self.mdp.getPossibleActions(state):
            q = self.computeQValueFromValues(state, action)
            if q > best_q:
                best_q = q
        diff = abs(self.values[state] - best_q)
        pq.push(state, -diff)

    # Step 4: Iterate
    for i in range(self.iterations):
        if pq.isEmpty():
            break
        s = pq.pop()
        if self.mdp.isTerminal(s):
            continue
        # Update value of s
        best_q = float('-inf')
        for action in self.mdp.getPossibleActions(s):
            q = self.computeQValueFromValues(s, action)
            if q > best_q:
                best_q = q
        self.values[s] = best_q

        # For each predecessor p of s
        for p in predecessors[s]:
            if self.mdp.isTerminal(p):
                continue
            best_q_p = float('-inf')
            for action in self.mdp.getPossibleActions(p):
                q = self.computeQValueFromValues(p, action)
                if q > best_q_p:
                    best_q_p = q
            diff = abs(self.values[p] - best_q_p)
            if diff > self.theta:
                pq.update(p, -diff)
```

**Line-by-line meaning:**

| Section | Meaning |
|---------|---------|
| `predecessors = {}` | Dictionary mapping each state to a set of its predecessors. |
| `predecessors[state] = set()` | Initialize each state's predecessor set as empty. |
| `for state in self.mdp.getStates():` | Loop over all states. |
| `if self.mdp.isTerminal(state): continue` | Skip terminal states — they have no outgoing actions. |
| `for action in self.mdp.getPossibleActions(state):` | For each action from this state... |
| `for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):` | For each possible outcome... |
| `if prob > 0:` | If there's a non-zero chance of reaching next_state... |
| `predecessors[next_state].add(state)` | Add this state as a predecessor of next_state. |
| `pq = util.PriorityQueue()` | Create a min-heap priority queue. Lower priority = popped first. We use negative diff so largest diff gets popped first. |
| `diff = abs(self.values[state] - best_q)` | Difference between current value and best possible Q-value. This is the "error" — how much the value would change if updated. |
| `pq.push(state, -diff)` | Push state into queue with priority = negative diff (so largest diff is highest priority). |
| `if pq.isEmpty(): break` | If queue is empty, all states are converged — stop early. |
| `s = pq.pop()` | Pop the state with the largest error (highest priority). |
| `self.values[s] = best_q` | Update the value of state s to its best Q-value. |
| `for p in predecessors[s]:` | For every state that can reach s... |
| `diff = abs(self.values[p] - best_q_p)` | Compute p's error. |
| `if diff > self.theta:` | Only care if the error is meaningful (above threshold θ). |
| `pq.update(p, -diff)` | Push/update p in the queue so it gets processed. |

**Why this is faster:** Regular value iteration updates every state every iteration. Prioritized sweeping only updates states that actually need it, converging faster.

---

## Q5 — Q-Learning Core Methods

**File:** `submission/qlearningAgents.py`  
**Concept:** Q-learning is an online reinforcement learning algorithm. Unlike value iteration, the agent does NOT know the MDP. It learns by trial and error — taking actions, observing rewards, and updating its Q-value estimates.

**Q-Learning Update Formula:**

```
Q(s,a) ← (1−α) × Q(s,a) + α × [reward + γ × max over a' of Q(s',a')]
```

In plain English: "After taking action a in state s and ending up in s' with reward r, update Q(s,a) by moving it partway (learning rate α) toward the sample: the immediate reward plus the discounted best future value from s'."

---

### Q5 Implementation Line-by-Line

```python
def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)
    self.qValues = util.Counter()
```

| Line | Meaning |
|------|---------|
| `self.qValues = util.Counter()` | Create a dictionary to store Q-values. Key is (state, action) tuple, value is the Q-value. Default is 0.0 for unseen state-action pairs. |

---

```python
def getQValue(self, state, action):
    return self.qValues[(state, action)]
```

| Line | Meaning |
|------|---------|
| `return self.qValues[(state, action)]` | Return the stored Q-value for this state-action pair. If never seen, `util.Counter` returns 0.0 by default. |

---

```python
def computeValueFromQValues(self, state):
    legalActions = self.getLegalActions(state)
    if not legalActions:
        return 0.0
    return max(self.getQValue(state, action) for action in legalActions)
```

| Line | Meaning |
|------|---------|
| `legalActions = self.getLegalActions(state)` | Get all legal actions from this state. |
| `if not legalActions: return 0.0` | Terminal state — no actions, value is 0. |
| `return max(...)` | Return the maximum Q-value across all legal actions. This is V(s) = max_a Q(s,a). |

---

```python
def computeActionFromQValues(self, state):
    legalActions = self.getLegalActions(state)
    if not legalActions:
        return None
    best_q = self.computeValueFromQValues(state)
    best_actions = [action for action in legalActions
                    if self.getQValue(state, action) == best_q]
    return random.choice(best_actions)
```

| Line | Meaning |
|------|---------|
| `best_q = self.computeValueFromQValues(state)` | Find the best Q-value for this state. |
| `best_actions = [...]` | Collect all actions that achieve this best Q-value (for tie-breaking). |
| `return random.choice(best_actions)` | Randomly pick one of the best actions. Breaks ties randomly. |

---

```python
def getAction(self, state):
    legalActions = self.getLegalActions(state)
    if not legalActions:
        return None
    if util.flipCoin(self.epsilon):
        return random.choice(legalActions)
    return self.computeActionFromQValues(state)
```

| Line | Meaning |
|------|---------|
| `if util.flipCoin(self.epsilon):` | With probability epsilon, explore (take a random action). |
| `return random.choice(legalActions)` | Pick any random legal action (could even be the best one). |
| `return self.computeActionFromQValues(state)` | Otherwise, exploit — take the best known action. |

**Epsilon-greedy:** ε fraction of time explore randomly, (1−ε) fraction of time use current best policy.

---

```python
def update(self, state, action, nextState, reward):
    sample = reward + self.discount * self.computeValueFromQValues(nextState)
    current_q = self.getQValue(state, action)
    self.qValues[(state, action)] = (1 - self.alpha) * current_q + self.alpha * sample
```

| Line | Meaning |
|------|---------|
| `sample = reward + self.discount * self.computeValueFromQValues(nextState)` | The "sample" — what we observed (reward) plus what we expect (discounted best future value from nextState). |
| `current_q = self.getQValue(state, action)` | The current Q-value estimate for (state, action). |
| `self.qValues[(state, action)] = (1 - self.alpha) * current_q + self.alpha * sample` | Move current Q-value partway (learning rate α) toward the sample. If α=1, we fully adopt the sample. If α=0, we never change. |

---

## Q8 — Q-Learning + Pacman

**File:** `submission/qlearningAgents.py`  
**Concept:** No new code needed for Q8. PacmanQAgent is a subclass of QLearningAgent with Pacman-specific default parameters (epsilon=0.05, alpha=0.2, gamma=0.8). Once Q5 is implemented correctly, Pacman learns to win.

**Results:** Pacman achieves **100% win rate** (100/100 test games, average score ~500).

**Why Q8 works:** The Q-learning agent learns from 2000 training games, then exploits the learned policy in 100 test games (epsilon=0, alpha=0 — no exploration or learning during testing).

---

## Q2 — Bridge Crossing Analysis

**File:** `submission/analysis.py`  
**Concept:** BridgeGrid has two exits: a low-reward exit (+1) nearby and a high-reward exit (+10) across a narrow bridge. With default parameters (discount=0.9, noise=0.2), the agent is too afraid to cross the bridge because noise might push it into the chasm (−100). We need to change ONLY ONE parameter so the optimal policy crosses the bridge.

**Answer:** `question2()` returns `(0.9, 0.0)` — keep discount at 0.9, change noise to 0.0.

**Why:** With noise=0.0, the agent always moves exactly where it intends. There's no risk of being pushed into the chasm, so crossing the bridge to get +10 becomes optimal.

```python
def question2():
    answerDiscount = 0.9
    answerNoise = 0.0
    return answerDiscount, answerNoise
```

| Line | Meaning |
|------|---------|
| `answerDiscount = 0.9` | Keep the discount factor at 0.9 (unchanged). |
| `answerNoise = 0.0` | Set noise to 0.0 — the agent always moves in the intended direction. |
| `return answerDiscount, answerNoise` | Return the tuple (discount, noise). |

**Constraint:** Change ONLY one parameter. We changed noise from 0.2 → 0.0, kept discount at 0.9.

---

## Files I Edited

| File | Questions | What I Implemented |
|------|-----------|-------------------|
| `submission/valueIterationAgents.py` | Q1, Q4 | `runValueIteration`, `computeQValueFromValues`, `computeActionFromValues`, `PrioritizedSweepingValueIterationAgent.runValueIteration` |
| `submission/qlearningAgents.py` | Q5, Q8 | `__init__`, `getQValue`, `computeValueFromQValues`, `computeActionFromQValues`, `getAction`, `update` |
| `submission/analysis.py` | Q2 | `question2()` returning `(0.9, 0.0)` |

---

## Possible Viva Questions & Answers

### Q1: What is the difference between value iteration and Q-learning?

**A:** Value iteration is an offline planning algorithm — it requires knowing the full MDP (transition probabilities and rewards) and computes the optimal policy before acting. Q-learning is an online reinforcement learning algorithm — it learns from experience by interacting with the environment, without knowing the transition probabilities or rewards in advance.

### Q2: What is the batch version of value iteration?

**A:** In the batch version, all new state values are computed from the OLD value vector simultaneously. We create `new_values`, compute every state's new value from `self.values` (old), then replace `self.values = new_values` at the end. This ensures all updates in iteration k use values from iteration k−1, not partially updated values.

### Q3: What does the discount factor γ do?

**A:** The discount factor (γ=0.9) determines how much future rewards are worth compared to immediate rewards. γ close to 0 means the agent only cares about immediate rewards. γ close to 1 means the agent values future rewards almost as much as immediate ones.

### Q4: What is epsilon-greedy exploration?

**A:** With probability ε, the agent takes a random action (exploration). With probability (1−ε), it takes the best action according to current Q-values (exploitation). This balances exploring new actions with exploiting known good ones.

### Q5: What is the Q-learning update rule?

**A:** Q(s,a) ← (1−α)Q(s,a) + α × [reward + γ × max Q(s',a')]. We move the current Q-value partway (learning rate α) toward the observed sample (immediate reward + discounted best future value).

### Q6: What happens at a terminal state?

**A:** Terminal states have no legal actions. Their value is 0 (no future rewards). `computeActionFromValues` returns None. `computeValueFromQValues` returns 0.0.

### Q7: What is prioritized sweeping?

**A:** Prioritized sweeping focuses value iteration updates on states whose values are likely to change. It uses a priority queue ordered by value difference (error), updating the highest-error states first. When a state's value changes, its predecessors are checked and re-queued if their error exceeds threshold θ.

### Q8: Why did you set noise=0.0 for Q2?

**A:** With noise=0.2, there's a 20% chance the agent ends up in an unintended state, making bridge crossing too risky (might fall into the −100 chasm). With noise=0.0, the agent always moves where it intends, so crossing the bridge to reach +10 becomes optimal. We changed only one parameter (noise) as required.

---

## How to Test

```bash
# Run autograder for all your questions
python3 autograde/autograder.py -q q1
python3 autograde/autograder.py -q q4
python3 autograde/autograder.py -q q5
python3 autograder/autograder.py -q q8
python3 autograder/autograder.py -q q2

# Run full autograder
python3 autograde/autograder.py
```

**All your questions pass:** Q1: 5/5, Q4: 1/0, Q5: 5/5, Q8: 1/1, Q2: 1/1
