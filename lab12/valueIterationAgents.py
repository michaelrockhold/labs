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


import mdp, util
import copy

from learningAgents import ValueEstimationAgent

### Policy

class Policy:
    """
    Inspired by Sutton & Barlo, 'Policy Iteration', pg 65
    """
    def __init__(self, mdp, discount, threshhold):
        self.mdp = mdp
        self.ɣ = discount
        self.θ = threshhold

        # initialize values and policy arbitrarily
        self.V = util.Counter()

        self.π = {}
        a = mdp.getPossibleActions(mdp.getStartState())[0]
        for s in self.mdp.getStates():
            self.π[s] = a
        self.iterate()


    def getBestActionForState(self, state):
        return self.π[state]

    def evaluate(self):
        while True:
            Δ = 0
            for s in self.mdp.getStates():
                v = self.V[s]
                self.V[s] = self.summarizeAction(s, self.π[s])
                Δ = max([Δ, abs(v - self.V[s])])
            if Δ < self.θ: break

    def improve(self):
        def find_max_action(s):
            actions = self.mdp.getPossibleActions(s)
            if actions == None or len(actions) == 0:
                raise Exception("no action to take out of %s" % (s,))
            return max(actions, key=lambda a:self.summarizeAction(s,a))

        policy_stable = True
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue
            old_action = self.π[s]
            self.π[s] = find_max_action(s)
            if old_action != self.π[s]:
                policy_stable = False
        return policy_stable

    def iterate(self):
        while True:
            self.evaluate()
            stable = self.improve()
            if stable: break

    def summarizeAction(self, current_state, action):

        if self.mdp.isTerminal(current_state):
            return 0
        if not action in self.mdp.getPossibleActions(current_state):
            return 0

        def value(st, act, prob, nxt):
            r = self.mdp.getReward(st,act,nxt)
            v = self.V[nxt]
            return prob * (r + self.ɣ * v)
        return sum([value(current_state, action, probability, nxtState) for (nxtState, probability) in self.mdp.getTransitionStatesAndProbs(current_state,action)])

### ValueIterationAgent

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
        self.ɣ = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # inspired by Sutton & Barto, 4.4 Value Iteration, pg 67
        def find_max_value(state):
            if self.mdp.isTerminal(state):
                return 0
            return max([(lambda a:self.payoff(state, a))(action) for action in self.mdp.getPossibleActions(state)])

        θ = 0
        for i in range(self.iterations):
            Δ = 0
            kPlus1values = copy.deepcopy(self.values)
            for s in mdp.getStates():
                v = self.values[s]
                kPlus1values[s] = find_max_value(s)
                Δ = max([Δ, abs(v - kPlus1values[s])])
            self.values = kPlus1values
            if Δ < θ:
                print("INFO: converging? with Δ < θ (%f < %f)" % (Δ, θ))
                break

        self.policy = Policy(self.mdp, self.ɣ, 0.12)


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def payoff(self, state, action):
        def reward(nextState,probability):
            r = self.mdp.getReward(state, action, nextState)
            return probability * (r + self.ɣ * self.values[nextState])
        payoffs = [reward(nxt,prob) for (nxt,prob) in self.mdp.getTransitionStatesAndProbs(state, action)]
        return sum(payoffs)

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        return self.payoff(state, action)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        return self.policy.getBestActionForState(state)
        def summarizeAction(s, a):
            if self.mdp.isTerminal(s):
                return 0
            if not a in self.mdp.getPossibleActions(s):
                print("WARN: action %s is not in state %s" % (a, s))
                return 0
            def value(st, act, prob, nxt):
                r = self.mdp.getReward(st,act,nxt)
                v = self.values[nxt]
                return prob * (r + self.ɣ * v)
            return sum([value(s, a, probability, nxtState) for (nxtState, probability) in self.mdp.getTransitionStatesAndProbs(s,a)])

        if self.mdp.isTerminal(state):
            return 0
        actions = self.mdp.getPossibleActions(state)
        if actions == None or len(actions) == 0:
            raise Exception("no action to take out of %s" % (state,))
        return max(actions, key=lambda a:summarizeAction(state,a))

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
