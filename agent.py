import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy_s = self._get_epsilon_greedy_probs(state, i_episode)

        action = np.random.choice(np.arange(0, 6), p=policy_s)

        return action, policy_s

    def step(self, state, action, reward, next_state, done, alpha, policy_s, method='q_learning' ):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if method == 'q_learning':
            self.Q[state][action] = (1-alpha) * self.Q[state][action] + alpha * (reward + np.max(self.Q[next_state]))
        if method == 'expected_sarsa':
            self.Q[state][action] = (1-alpha) * self.Q[state][action] + alpha * (reward + np.dot(self.Q[next_state], policy_s))

    def _get_epsilon_greedy_probs(self, state, i_episode, eps=None):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        epsilon = 1.0 / i_episode
        if eps is not None:
            epsilon = eps
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(self.Q[state])] = 1 - epsilon + (epsilon / self.nA)
        return policy_s
