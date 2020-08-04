import random
import numpy as np


class EnvLinear:
    """d : dimension of the model
    type : drift or abrupt
    seed : fix seed of random generator"""

    def __init__(self, d, k, state=np.random.RandomState(10), rate=1, sigma_noise=1, env_type="drift"):
        self.d = d #dimension of the problem
        self.k = k  # number of actions at each step
        self.type = env_type #type of non-stationnarity
        self.state = state #some random state
        drw = state.normal(np.zeros(d, ))
        self.theta = drw / np.sqrt(np.sum(drw ** 2))  #theta star
        self.a_list = [] #list of actions
        self.theta_list = [] #list of real theta values
        self.std_noise = sigma_noise  # standard deviation of the noise
        self.inner_clock = 0 #timesteps
        self.rate = rate
        self.next_change = int(state.exponential(self.rate))

    def __str__(self):
        return f" dim: {self.d} theta: {self.theta}"  # à changer mais bien pratique

    def get_action_set(self):
        a_list = []
        for i in range(self.k):
            drw = self.state.normal(np.zeros(self.d, ))
            arr = drw / np.sqrt(np.sum(drw ** 2))
            a_list.append(arr)
        self.a_list = a_list
        return a_list

    def get_canonical_action_set(self):
        a_list = []
        for i in range(self.d):
            arr = np.zeros(self.d)
            arr[i] = 1
            a_list.append(arr)
        self.a_list = a_list
        return a_list

    def update_theta(self):  # TODO : changer le nom
        if self.type == "stationnary":
            pass
        elif self.type == "drift":
            new_val = self.theta + 0.05 * self.state.normal(np.zeros(self.d, ))  # passer en paramètre
            self.theta = new_val / np.sqrt(np.sum(new_val ** 2))
        elif self.type == "abrupt":
            if self.inner_clock == self.next_change:
                drw = self.state.normal(np.zeros(self.d, ))
                self.theta = drw / np.sqrt(np.sum(drw ** 2))
                self.next_change = int(self.state.exponential(self.rate))
                print(f"ENV : Abrupt change triggered, stationnary for {self.next_change} steps")
                self.inner_clock = 0
            else:
                self.inner_clock += 1

    def get_best_choice(self):
        """
        Return the indices of the best arm and the best_reward associated
        """
        all_expected_rewards = [np.dot(self.a_list[i], self.theta) for i in range(len(self.a_list))]
        idx_best_arm = np.argmax(all_expected_rewards)
        best_reward = all_expected_rewards[idx_best_arm]
        return idx_best_arm, best_reward

    def get_reward(self, idx_action): #TODO : s'assurer que bruit ajouté côté policy
        reward = np.dot(self.a_list[idx_action], self.theta) #noise will be added in policies
        return reward

    def update_env(self):
        self.theta_list.append(self.theta)
        self.update_theta()

    def re_init(self):
        drw = self.state.normal(np.zeros(self.d, ))
        self.theta = drw / np.sqrt(np.sum(drw ** 2))  # a changer ?
        self.a_list = []
        self.theta_list = []
        self.inner_clock = 0
        self.next_change = int(self.state.exponential(self.rate))
