import numpy as np
from policies.SW_LinUCB import SWLinUCB
from policies.D_LinUCB import DLinUCB

class BOB:
    """
    bandits over bandits policy class
    """
    def __init__(self, d, T, H, n_experts, type="SW", param={}, name="", seed_alg=2):
        """
        - d: dimension of the action vectors
        - T: time horizon of the experiment
        - H: size of a block
        - n_experts: number of experts
        - type: "SW" or "D" for SW-LinUCB or D-LinUCB as experts
        - name: additional suffix when comparing several policies (optional)
        - seed_alg: used for reproducibility, in particular when using parallelization
        - param:
            - delta: probability of theta in the confidence bound
            - alpha: tuning the exploration parameter
            - lambda_: regularization parameter
            - s: constant such that L2 norm of theta smaller than s
            for SW type :
                - tau: sliding window parameter
            for D type :
                - gamma: discount parameter
            - name: additional suffix when comparing several policies (optional)
            - sm: Should Sherman-Morisson formula be used for inverting matrices ?
            - sigma_noise: square root of the variance of the noise
            - verbose: To print information
        """

        assert type in ["SW","D"], 'choose either SW or D as BOB type'

        self.d = d
        self.T = T
        self.H = H
        self.type = type
        self.experts = []
        self.param = param
        # if n_experts=='auto':
        #     n_experts = np.ceil((2/3) * np.log2(2*param['s']*T**(3/2)
        #     /np.log(T)**(3/2))) + 1
        #     self.n_experts = int(n_experts)
        # else:
        #     self.n_experts = n_experts

        #creating experts
        if type == "SW":
            if param['tau'] != 'auto':
                assert self.n_experts == len(param['tau']), "Number of experts should match number of parameters"
                for i in range(len(param['tau'])):
                    if self.param['tau'][i] <= self.H:
                        args = {"d":self.d, "delta":self.param["delta"],
                         "alpha":self.param["alpha"],
                        "lambda_":self.param['lambda_'], "s":self.param['s'],
                        "tau":self.param['tau'][i], "name":"SW - " + str(self.param['tau'][i]),
                        "sm":self.param['sm'], "sigma_noise":self.param['sigma_noise'],
                        "verbose":self.param['verbose']}
                        self.experts.append(SWLinUCB(**args))
            else:
                pavement = self.get_pavement()
                for i in range(len(pavement)):
                    if pavement[i] <= self.H:
                        args = {"d":self.d, "delta":self.param["delta"],
                         "alpha":self.param["alpha"],
                        "lambda_":self.param['lambda_'], "s":self.param['s'],
                        "tau":pavement[i], "name":"SW - " + str(pavement[i]),
                        "sm":self.param['sm'], "sigma_noise":self.param['sigma_noise'],
                        "verbose":self.param['verbose']}
                        self.experts.append(SWLinUCB(**args))
                if H not in pavement:
                    args = {"d":self.d, "delta":self.param["delta"],
                     "alpha":self.param["alpha"],
                    "lambda_":self.param['lambda_'], "s":self.param['s'],
                    "tau":H, "name":"SW - " + str(H),
                    "sm":self.param['sm'], "sigma_noise":self.param['sigma_noise'],
                    "verbose":self.param['verbose']}
                    self.experts.append(SWLinUCB(**args))

        elif type == "D":
            assert self.n_experts == len(param['gamma']), "Number of experts should match number of parameters"
            for i in range(self.n_experts):
                args = {"d":self.d, "delta":self.param["delta"],
                 "alpha":self.param["alpha"],
                "lambda_":self.param['lambda_'], "s":self.param['s'],
                "gamma":self.param['gamma'][i], "name":"SW - " + str(i),
                "sm":self.param['sm'], "sigma_noise":self.param['sigma_noise'],
                "verbose":self.param['verbose']}
                self.experts.append(SWLinUCB(**args))

        self.n_experts = len(self.experts)
        self.master_weights = np.ones(self.n_experts) #vector of s_i^(gamma_j) in our paper
        self.master_p_history = [] #for vizualisation purpose
        self.block_reward = 0
        self.expert_choice_history = []


        self.alpha = np.minimum(1, np.sqrt(self.n_experts * np.log(self.n_experts)/
        ((np.exp(1)-1) * np.ceil(self.T/self.H))))

    def choose_expert(self):
        """
        Choice of expert at the beginning of each block
        """
        self.master_p = (1-self.alpha)*self.master_weights/np.sum(self.master_weights) \
        + self.alpha/self.n_experts
        self.master_p_history.append(self.master_p)
        expert_idx = np.random.choice(self.n_experts, replace=True, p=self.master_p)
        self.last_expert_idx = expert_idx
        self.expert_choice_history.append(self.experts[expert_idx].name)
        return self.experts[expert_idx]

    def update_master_weights(self):
        """
        Update the underlying weights of the EXP3
        """
        change = np.exp(self.alpha/(self.n_experts*self.master_p[self.last_expert_idx]) *
        (1/2 + (self.block_reward/(2*self.H + 4*self.param['sigma_noise']
        *np.sqrt(self.H*np.log(self.T/np.sqrt(self.H)))))))
        self.master_weights[self.last_expert_idx] = \
        self.master_weights[self.last_expert_idx] * change

    def select_arm(self,t,arms):
        if t % self.H != 0:
            return self.current_expert.select_arm(arms)
        else:
            if hasattr(self, 'last_expert_idx'): #We do not want to update at first step
                self.update_master_weights()
            self.current_expert = self.choose_expert()
            self.block_reward = 0
            self.current_expert.re_init()
            return self.current_expert.select_arm(arms)

    def update(self, action, reward):
        self.current_expert.update(action, reward)
        self.block_reward += reward

    def show_info(self):
        """
        """
        print("Proba master: ", self.master_p)
        print("Choice : ", expert_idx)
        print("Current expert: ", self.last_expert_idx)
        print("block reward: ", self.block_reward)
        print("normalized block reward: ", self.block_reward/(2*self.H + 4*self.param['sigma_noise']
        *np.sqrt(self.H*np.log(self.T/np.sqrt(self.H)))))
        print("weights: ", self.master_weights)
        print("\n")

    def get_pavement(self):
        n_experts = int(np.ceil((2/3) * np.log2(2*self.param['s']*self.T**(3/2)
        /np.log(self.T)**(3/2))) + 1)
        gamma_pavement = [1 - (1/2) * np.log(self.T)
        /(self.T*(2*self.param['s'])**(2/3)) * 2**i for i in range(n_experts)]
        if self.type == "SW":
            return [int(1/(1-gamma)) for gamma in gamma_pavement]
