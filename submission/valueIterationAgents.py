# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import os, sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ["submission", "readonly", "autograde"]:
    _p = os.path.join(_root, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

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

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q_value = 0.0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
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

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

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

