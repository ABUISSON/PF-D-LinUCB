# -*- coding: utf-8 -*-

"""
    Simulator class
"""

# Author: Yoan Russac (yoan.russac@ens.fr)
# License: BSD (3-clause)

# Importation
import numpy as np
from tqdm import tqdm
import time


class Simulator(object):
    """
    Simulator of stochastic games.
    param:
        - MAB: list, List of arms.
        - policies: list, List of policies to test.
        - K: int, Number of items (arms) in the pool.
        - d: int, Dimension of the problem
        - T: Number of steps in each round
    """
    def __init__(self, env, policies, k, d, steps, verbose, param):
        """"
        global init function, initializing all the internal parameters that are fixed
        """
        self.policies = policies
        self.env = env
        self.steps = steps
        self.d = d
        self.k = k
        self.verbose = verbose
        self.param = param
        self.L = 1
        # self.s_method = self.mab.s_method

        # if 'L' in self.param:
        #    self.L = self.param['L']

    def run_exp(self, steps, n_mc, q, seed_tot, t_saved=None):
        """
        Runs an experiment with steps points and n_mc Monte Carlo repetition
        param:
            - steps: Number of steps for an experiment (int)
            - n_mc: Total number of Monte Carlo experiment (int)
            - q: Quantile (int). ex: q=5%
            - t_saved: Trajectory of points saved to store fewer than 'steps'-points on a trajectory.
                        (numpy array ndim = 1)
        """
        random_state = np.random.RandomState(seed_tot)
        seeds = random_state.randint(1, 312414, n_mc)  # seed for all experiment on

        if t_saved is None:
            t_saved = [i for i in range(steps)]  # if t_saved is not given the entire trajectory is saved

        cum_regret = dict()
        cum_best_action = dict()
        n_sub = np.size(t_saved)  # Number of points saved for each trajectory
        avg_regret = dict()
        avg_cum_best_action = dict()
        q_regret = dict()
        q_b_a = dict()
        Q_b_a = dict()
        up_q_regret = dict()
        timedic = dict()
        rewards = dict()
        best_action_selected = dict()

        for policy in self.policies:
            name = policy.__str__()
            cum_regret[name] = np.zeros((n_mc, n_sub))
            cum_best_action[name] = np.zeros((n_mc, n_sub))
            timedic[name] = 0

        # run n_mc independent simulations
        for nExp in tqdm(range(n_mc)):
            if self.verbose:
                print('--------')
                print('Experiment number: ' + str(nExp))
                print('--------')

            # Re-initialization part
            state = np.random.RandomState(seeds[nExp])
            self.env.state = state
            self.env.re_init()

            for i, policy in enumerate(self.policies):
                policy.re_init()
                policy.r_s = self.env.state
                name = policy.__str__() #mettre str(policy) non ?
                rewards[name] = np.zeros(steps)
                best_action_selected[name] = np.zeros(steps)

            optimal_rewards = np.zeros(steps)

            for t in range(steps):
                self.env.get_action_set()  # Receiving the action set for the round
                idx_best_arm, instant_best_reward = self.env.get_best_arm()  # Best action for all the policies
                optimal_rewards[t] = instant_best_reward
                noise = state.normal(scale=self.env.std_noise)  # centered noise with std_noise standard deviation

                for i, policy in enumerate(self.policies):
                    name = policy.__str__()
                    time_init = time.time()
                    idx_a_t = policy.choose_a(self.env.a_list)  # idx of chosen arm
                    round_reward = self.env.play(idx_a_t, noise)  # reward obtained by playing the arm
                    policy.update(round_reward)
                    expected_reward_round = np.dot(policy.last_action, self.env.theta)
                    rewards[name][t] = expected_reward_round
                    best_action_selected[name][t] = int(idx_a_t == idx_best_arm)
                    timedic[name] += time.time() - time_init

            for policy in self.policies:
                name = policy.__str__()
                cum_regret[name][nExp, :] = np.cumsum(optimal_rewards - rewards[name])[t_saved]
                cum_best_action[name][nExp, :] = np.cumsum(best_action_selected[name])

        for policy in self.policies:
            name = policy.__str__()
            cum_reg = cum_regret[name]
            cum_b_a = cum_best_action[name]
            avg_cum_best_action[name] = np.mean(cum_b_a, 0)
            avg_regret[name] = np.mean(cum_reg, 0)
            q_regret[name] = np.percentile(cum_reg, q, 0)
            q_b_a[name] = np.percentile(cum_b_a, q, 0)
            Q_b_a[name] = np.percentile(cum_b_a, 100 - q, 0)
            up_q_regret[name] = np.percentile(cum_reg, 100 - q, 0)

        print("--- Data built ---")
        return avg_regret, q_regret, up_q_regret, timedic, avg_cum_best_action, q_b_a, Q_b_a, cum_regret
