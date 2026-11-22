# COSC 604 — Pacman Reinforcement Learning Project

**Course:** COSC 604 Artificial Intelligence | Fall 2026  
**Instructor:** Dr. Andreas Henschel  
**Due:** Wednesday, December 2, 2026  
**Points:** 25 (+1 optional bonus)  
**Collaborators:** Hardik Sharma & Damjan

---

## What This Project Is About

You are building **AI agents that learn to make decisions** through reinforcement learning (RL). Instead of telling the agent exactly what to do in every situation, you implement algorithms that let the agent learn from trial and error.

You will implement two core RL algorithms:

| Algorithm | What It Does |
|-----------|-------------|
| **Value Iteration** | An offline planner that computes the optimal policy by thinking through the MDP (Markov Decision Process) model before acting. It's like solving the puzzle in your head before making a move. |
| **Q-Learning** | An online learner that figures out the best actions by actually trying them out and seeing what happens. No model needed — just experience. |

You will test your agents on three environments:

1. **Gridworld** — A simple grid navigation task (from class)
2. **Crawler** — A simulated robot controller (2-joint robot arm)
3. **Pacman** — The classic game: eat food, avoid ghosts

---

## What You Need to Do

There are **9 questions** (Q1–Q9). Q4 is optional extra credit (+1 point).

| Q | Topic | Points | File |
|---|-------|--------|------|
| Q1 | Value Iteration | 5 | `valueIterationAgents.py` |
| Q2 | Bridge Crossing Analysis | 1 | `analysis.py` |
| Q3 | Policies (discount/noise/living reward tuning) | 6 | `analysis.py` |
| Q4 | Prioritized Sweeping (extra credit) | +1 | `valueIterationAgents.py` |
| Q5 | Q-Learning | 5 | `qlearningAgents.py` |
| Q6 | Epsilon-Greedy + Crawler | 2 | `qlearningAgents.py` |
| Q7 | Bridge Crossing Revisited | 1 | `analysis.py` |
| Q8 | Q-Learning + Pacman | 1 | `qlearningAgents.py` |
| Q9 | Approximate Q-Learning | 4 | `qlearningAgents.py` |

### Files You Edit (your work goes here)

| File | What You Implement |
|------|-------------------|
| `submission/valueIterationAgents.py` | Value iteration agent + prioritized sweeping (Q1, Q4) |
| `submission/qlearningAgents.py` | Q-learning agent + approximate Q-learning agent (Q5, Q6, Q9) |
| `submission/analysis.py` | Answers to Q2, Q3, Q7 |

### Files You Read Only (DO NOT edit)

These are in `readonly/`. If you change them, the autograder will fail.

- `mdp.py` — MDP base class
- `learningAgents.py` — Base agent classes
- `util.py` — Utilities (Counter, PriorityQueue, etc.)
- `gridworld.py` — Gridworld environment
- `featureExtractors.py` — Feature extractors for approximate Q-learning
- `crawler.py`, `pacman.py`, `game.py`, `environment.py`, etc.

---

## Project Structure

```
AI_Final_Proj/
├── submission/          <-- EDIT THESE (your work)
│   ├── analysis.py          (Q2, Q3, Q7)
│   ├── qlearningAgents.py   (Q5, Q6, Q9)
│   └── valueIterationAgents.py (Q1, Q4 extra credit)
├── readonly/            <-- READ ONLY (don't touch)
│   ├── mdp.py, util.py, gridworld.py, learningAgents.py, featureExtractors.py
│   ├── game.py, pacman.py, crawler.py, environment.py, etc.
├── autograde/           <-- autograder files (don't edit)
│   ├── autograder.py, grading.py, testClasses.py, etc.
├── layouts/             <-- map layouts
├── test_cases/          <-- test cases
├── analysis.py          <-- symlink to submission/
├── qlearningAgents.py   <-- symlink to submission/
├── valueIterationAgents.py <-- symlink to submission/
├── reinforcementTestClasses.py <-- symlink to autograde/
└── KU_COSC604_RL_Project.pdf  <-- original project spec
```

---

## How to Run

All commands from the project root (`AI_Final_Proj/`):

```bash
# Run value iteration on Gridworld
python3 readonly/gridworld.py -a value -i 100

# Run Q-learning on the crawler robot
python3 readonly/crawler.py

# Train Pacman Q-learning agent
python3 readonly/pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

# Run autograder for a specific question
python3 autograde/autograder.py -q q1

# Run full autograder
python3 autograde/autograder.py
```

**Note:** The GUI (tkinter) requires macOS 26+. If you're on an older macOS, you'll get a GUI error — the logic still works, you just can't see the visual display. Use text-only mode with `-t` flag if needed.

---

## Work Split: Hardik & Damjan (Equal)

9 questions total (Q4 is optional extra credit). Split **5 for Hardik, 4 for Damjan**. Both of you must understand all the code — you will be quizzed on every line you submit.

### Hardik — 5 Questions

| Question | Topic | Points | File |
|----------|-------|--------|------|
| Q1 | Value Iteration | 5 | `submission/valueIterationAgents.py` |
| Q4 | Prioritized Sweeping (extra credit) | +1 | `submission/valueIterationAgents.py` |
| Q5 | Q-Learning | 5 | `submission/qlearningAgents.py` |
| Q8 | Q-Learning + Pacman | 1 | `submission/qlearningAgents.py` |
| Q2 | Bridge Crossing Analysis | 1 | `submission/analysis.py` |

**Hardik's tasks:**
- Implement `ValueIterationAgent` (runValueIteration, computeActionFromValues, computeQValueFromValues) — Q1
- Implement `PrioritizedSweepingValueIterationAgent` (Q4, optional extra credit)
- Implement Q-learning core methods: `update`, `computeValueFromQValues`, `getQValue`, `computeActionFromQValues` — Q5
- Verify Q8 runs (Pacman Q-learning wins 80%+)
- Answer Q2: tune discount/noise so the agent crosses the bridge
- Verify Q1, Q5 pass autograder

**Hardik total points: 13** (5 + 1 + 5 + 1 + 1)

---

### Damjan — 4 Questions

| Question | Topic | Points | File |
|----------|-------|--------|------|
| Q3 | Policies (discount/noise/living reward tuning) | 6 | `submission/analysis.py` |
| Q6 | Epsilon-Greedy + Crawler | 2 | `submission/qlearningAgents.py` |
| Q7 | Bridge Crossing Revisited | 1 | `submission/analysis.py` |
| Q9 | Approximate Q-Learning | 4 | `submission/qlearningAgents.py` |

**Damjan's tasks:**
- Answer Q3: find discount/noise/living reward settings for all 5 policy types
- Implement epsilon-greedy action selection in `getAction` — Q6
- Verify Q6 works on Crawler robot (`python3 readonly/crawler.py`)
- Answer Q7: find epsilon/learning rate for bridge crossing (99%+ success)
- Implement `ApproximateQAgent` — Q9
- Verify Q9 passes autograder

**Damjan total points: 13** (6 + 2 + 1 + 4)

---

### Shared Responsibilities (Both)

- Q8 (Pacman) is primarily Hardik's, but Damjan should verify it works too since it builds on Q5/Q6
- Review each other's code before submission
- Make sure you both understand every line in all three files
- Submit together on Blackboard
- Run the full autograder before submitting: `python3 autograde/autograder.py`

**Damjan's tasks:**
- Implement `QLearningAgent` (update, computeValueFromQValues, getQValue, computeActionFromQValues, getAction)
- Implement `ApproximateQAgent` (Q9)
- Answer Q3: find discount/noise/living reward settings for 5 policy types
- Answer Q7: find epsilon/learning rate for bridge crossing
- Verify Q5, Q6, Q8, Q9 pass autograder

### Shared Responsibilities (Both)

- Q8 (Q-Learning + Pacman) — works once Q5/Q6 are done; both should verify it runs
- Review each other's code before submission
- Make sure you both understand every line in all three files
- Submit together on Blackboard

---

## Important Rules

1. **Submit your own work.** Submissions are checked for similarity. Do not copy from classmates or online sources.
2. **Don't change function/class names** — the autograder will fail.
3. **You will be quizzed on your code.** Make sure you understand every line you submit.
4. **Run the full autograder** before submitting: `python3 autograde/autograder.py`

---

## Getting Started

1. Hardik: open `submission/valueIterationAgents.py` and start with Q1
2. Damjan: open `submission/qlearningAgents.py` and start with Q5
3. Both: read `submission/analysis.py` to see the stub functions you need to fill in
4. Test frequently with the autograder: `python3 autograde/autograder.py -q q1` (or q2, q3, etc.)

Good luck!
