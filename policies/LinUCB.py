import numpy as np
from numpy.linalg import pinv


class LinUCB:
    def __init__(self, d, param={}, name="", seed_alg=2, delta=0.05):
        """
        param:
            - d: dimension of the action vectors
            - name: additional suffix when comparing several policies (optional)
            - seed_alg: used for reproducibility, in particular when using parallelization
            - param: dic containing specific information
                - alpha: tuning the exploration parameter
                - L: upper-bound for the L2-norm of the actions
                - lambda_: regularization parameter
                - s: constant such that L2 norm of theta smaller than s
                - sigma_noise: square root of the variance of the noise
                - sm: Should Sherman-Morisson formula be used for inverting matrices ?
                - verbose: To print information
            - delta: probability of theta not in the confidence ellipsoid
        """
        # immediate attributes from the constructor
        self.d = d
        self.delta = delta
        self.name = name  # optional practical when comparing several LinUCB policies
        self.r_s = np.random.RandomState(seed_alg)

        # param attributes
        if 'alpha' in param:
            self.alpha = param['alpha']
        else:
            self.alpha = 1

        if 'L' in param:
            self.L = param['L']
        else:
            self.L = 1

        if 'lambda_' in param:
            self.lambda_ = param['lambda_']
        else:
            self.lambda_ = 1

        if 's' in param:
            self.s = param['s']
        else:
            self.s = 1

        if 'sigma_noise' in param:
            self.sigma_noise = param['sigma_noise']
        else:
            self.sigma_noise = 1

        if 'sm' in param:
            self.sm = param['sm']
        else:
            self.sm = False

        if 'verbose' in param:
            self.verbose = param['verbose']
        else:
            self.verbose = False

        # build attributes
        self.c_delta = 2 * np.log(1 / self.delta)
        self.const1 = np.sqrt(self.lambda_) * self.s

        # attributes for the re-init
        self.t = 0
        self.last_action = np.zeros(self.d)
        self.hat_theta = np.zeros(self.d)
        self.cov = self.lambda_ * np.identity(self.d)
        self.invcov = 1 / self.lambda_ * np.identity(self.d)
        self.b = np.zeros(self.d)

    def choose_a(self, a_list):
        """
        Selecting an arm according to the LinUCB policy
        param:
            - arms : list of arms
        Output:
        -------
        chosen_arm : index of the pulled arm
        """
        assert type(a_list) == list, 'List of arms as input required'
        kt = len(a_list)
        ucb_s = np.zeros(kt)  # upper-confidence bounds for every action
        beta_t = self.const1 + self.sigma_noise * \
                 np.sqrt(self.c_delta + self.d * np.log(1 + self.t * self.L ** 2 / (self.lambda_ * self.d)))
        for (i, a) in enumerate(a_list):
            invcov_a = np.inner(self.invcov, a.T)
            ucb_s[i] = np.dot(self.hat_theta, a) + self.alpha * beta_t * np.sqrt(np.dot(a, invcov_a))
        mixer = self.r_s.rand(ucb_s.size)  # Shuffle to avoid always pulling the same arm when ties
        ucb_indices = list(np.lexsort((mixer, ucb_s)))  # Sort the indices
        idx_chosen_arm = ucb_indices[-1]
        self.last_action = a_list[idx_chosen_arm]
        if self.verbose:
            # Sanity checks
            print('--- t:', self.t)
            print('--- beta_t:', beta_t)
            print('--- theta_hat: ', self.hat_theta)
            print('--- Design Matrix:', self.cov)
            print('--- b matrix:', self.b)
            print('--- UCBs:', ucb_s)
            print('--- Chosen arm:', idx_chosen_arm)
            print('--- self.last_action:', self.last_action)
        return idx_chosen_arm

    def update(self, reward):
        """
        Updating the main parameters for the model
        param:
            - reward: Reward used for updating
        Output:
        -------
        Nothing, but the class instances are updated
        """
        features = self.last_action
        aat = np.outer(features, features.T)
        self.cov = self.cov + aat
        self.b = self.b + reward * features
        if not self.sm:
            self.invcov = pinv(self.cov)
        else:
            a = features[:, np.newaxis]
            const = 1 / (1 + np.dot(features, np.inner(self.invcov, features)))
            const2 = np.matmul(self.invcov, a)
            self.invcov = self.invcov - const * np.matmul(const2, const2.T)
        self.hat_theta = np.inner(self.invcov, self.b)
        if self.verbose:
            print('AAt:', aat)
            print('Policy was updated at time t= ' + str(self.t))
            print('Reward received =', reward)
        self.t += 1

    def re_init(self):
        """
        Re-init function to reinitialize the statistics while keeping the same hyper-parameters
        """
        self.t = 0
        self.hat_theta = np.zeros(self.d)
        self.cov = self.lambda_ * np.identity(self.d)
        self.invcov = 1 / self.lambda_ * np.identity(self.d)
        self.b = np.zeros(self.d)
        if self.verbose:
            print('Parameters of the policy reinitialized')
            print('Design Matrix after init: ', self.cov)

    def __str__(self):
        return 'LinUCB' + self.name

    @staticmethod
    def id():
        return 'LinUCB'
