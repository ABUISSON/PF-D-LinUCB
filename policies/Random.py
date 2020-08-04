import numpy as np


class RandomPolicy:
    def __init__(self, d, seed_alg=1, name=''):
        self.d = d
        self.name = name
        self.last_action = np.zeros(self.d)
        self.r_s = np.random.RandomState(seed_alg)

    def choose_a(self, a_list):
        randm_idx = self.r_s.randint(len(a_list))
        self.last_action = a_list[randm_idx]
        return randm_idx

    def update(self, reward):
        pass

    def __str__(self):
        return 'Random' + self.name

    def re_init(self):
        pass
