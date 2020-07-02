import numpy as np
import random
from utils.distributions import bernoulli
import math


# ----------------------------------------------------------------------------------------------------------------------
class Recombination(object):
    def __init__(self):
        pass

    def recombination(self, x):
        pass


# ----------------------------------------------------------------------------------------------------------------------
class EvolutionStrategyRecombination(Recombination):
    def __init__(self, type='es', bounds=(-np.infty, np.infty), params=None,
                 y_real=None, fun=None):
        super().__init__()
        self.type = type
        self.bounds = bounds
        assert (0. <= params['SIGMA'] <= 3.), 'SIGMA must be in [0, 3]'
        # assert type in [], 'type must be one in {}'

        self.params = params

        self.SIGMA = params['SIGMA']
        self.ALPHA = params['ALPHA']
        self.BETA = params['BETA']
        self.update = params['update']
        self.alpha_matrix = np.zeros((140, 140))
        self.y_real = y_real
        self.pop_size = params['pop_size']

        self.sigma_list = np.zeros((140, 1))
        for i in range(self.sigma_list.shape[0]):
            self.sigma_list[i] = self.SIGMA

        self.bias_coe = np.random.uniform(-1, 1, 4)

        self.fun = fun

    def recombination(self, x):
        # discrete recombination
        if self.type in ['es_a1', 'es_a2', 'es_a3', 'es_a4', 'es_a5', 'es_a6', 'es_a7', 'es_a8']:

            offsprings = []

            for i in range(140):

                a = random.randint(0, x.shape[0]-1)
                b = random.randint(0, x.shape[0]-1)
                c = random.randint(0, x.shape[0]-1)

                p1 = x[a]
                p2 = x[b]
                p3 = x[c]

                parents = [p1, p2, p3]
                offspring = []

                for z in range(4):
                    rand = random.randint(0, 2)
                    offspring.append(parents[rand][z])

                offsprings.append(offspring)

            offsprings = np.asarray(offsprings)

            return offsprings

        # intermediate recombination
        elif self.type in ['es_b1', 'es_b2', 'es_b3', 'es_b4', 'es_b5', 'es_b6', 'es_b7', 'es_b8']:

            offsprings = []

            for i in range(140):

                a = random.randint(0, x.shape[0]-1)
                b = random.randint(0, x.shape[0]-1)
                c = random.randint(0, x.shape[0]-1)

                p1 = x[a]
                p2 = x[b]
                p3 = x[c]

                offspring = []

                for z in range(4):
                    param = (p1[z] + p2[z] + p3[z]) / 3
                    offspring.append(param)

                offsprings.append(offspring)

            offsprings = np.asarray(offsprings)

            return offsprings

    def mutation(self, x, c=None, cs=None):
        # uncorrelated mutation with one sigma
        if self.type in ['es_a1', 'es_a2', 'es_b1', 'es_b2']:

            iter = 5

            if c % iter == 0 and c != 0:
                if cs/c > 1/5:
                    self.SIGMA = self.SIGMA / self.update
                elif cs/c < 1/5:
                    self.SIGMA = self.SIGMA * self.update

            # learning rate
            t = 1 / math.sqrt(x.shape[0])

            self.SIGMA = self.SIGMA * math.exp(t * np.random.normal(0, 1))

            z = self.SIGMA * np.random.normal(0, 1, (x.shape[0], 4))
            x_new = x + z

            x_new = np.clip(x_new, self.bounds[0], self.bounds[1])
            return x_new

        # uncorrelated mutation with N sigmas
        elif self.type in ['es_a3', 'es_a4', 'es_b3', 'es_b4']:
            iter = 5

            if c % iter == 0 and c != 0:
                if cs / c > 1 / 5:
                    self.sigma_list = self.sigma_list / self.update
                elif cs / c < 1 / 5:
                    self.sigma_list = self.sigma_list * self.update

            x_new = np.zeros((x.shape[0], 4))

            t1 = 1 / math.sqrt(2 * math.sqrt(x.shape[0]))
            t2 = 1 / math.sqrt(2 * x.shape[0])

            for i in range(4):
                self.sigma_list[i] = self.sigma_list[i] * math.exp((t1 * np.random.normal(0, 1)) +
                                                                   (t2 * np.random.normal(0, 1)))

            for i in range(x.shape[0]):
                z = np.zeros(4)
                for j in range(4):
                    z[j] = self.sigma_list[j] * np.random.normal(0, 1)
                x_new[i] = x[i] + z

            x_new = np.clip(x_new, self.bounds[0], self.bounds[1])
            return x_new

        # Correlated mutation
        elif self.type in ['es_a5', 'es_a6', 'es_b5', 'es_b6']:
            iter = 5

            if c % iter == 0 and c != 0:
                if cs / c > 1 / 5:
                    self.SIGMA = self.SIGMA / self.update
                elif cs / c < 1 / 5:
                    self.SIGMA = self.SIGMA * self.update

            x_new = np.zeros((x.shape[0], 4))

            t1 = 1 / math.sqrt(2 * math.sqrt(x.shape[0]))
            t2 = 1 / math.sqrt(2 * x.shape[0])

            self.ALPHA = self.ALPHA + self.BETA * np.random.normal(0, 1)

            R_xy = np.array(((np.cos(self.ALPHA), -np.sin(self.ALPHA), 0, 0),
                             (np.sin(self.ALPHA), np.cos(self.ALPHA), 0, 0),
                             (0, 0, 1, 0),
                             (0, 0, 0, 1)))

            R_xz = np.array(((np.cos(self.ALPHA), 0, -np.sin(self.ALPHA), 0),
                             (0, 1, 0, 0),
                             (np.sin(self.ALPHA), 0, np.cos(self.ALPHA), 0),
                             (0, 0, 0, 1)))

            R_xw = np.array(((np.cos(self.ALPHA), 0, 0, -np.sin(self.ALPHA)),
                             (0, 1, 0, 0),
                             (0, 0, 1, 0),
                             (np.sin(self.ALPHA), 0, 0, np.cos(self.ALPHA))))

            R_yz = np.array(((1, 0, 0, 0),
                             (0, np.cos(self.ALPHA), -np.sin(self.ALPHA), 0),
                             (0, np.sin(self.ALPHA), np.cos(self.ALPHA), 0),
                             (0, 0, 0, 1)))

            R_yw = np.array(((1, 0, 0, 0),
                             (0, np.cos(self.ALPHA), 0, -np.sin(self.ALPHA)),
                             (0, 0, 1, 0),
                             (0, np.sin(self.ALPHA), 0, np.cos(self.ALPHA))))

            R_zw = np.array(((1, 0, 0, 0),
                             (0, 1, 0, 0),
                             (0, 0, np.cos(self.ALPHA), -np.sin(self.ALPHA)),
                             (0, 0, np.sin(self.ALPHA), np.cos(self.ALPHA))))

            R = np.linalg.multi_dot([R_xy, R_xz, R_xw, R_yz, R_yw, R_zw])

            for i in range(4):
                self.sigma_list[i] = self.sigma_list[i] * math.exp((t1 * np.random.normal(0, 1)) +
                                                                   (t2 * np.random.normal(0, 1)))

            for i in range(x.shape[0]):
                z = np.zeros(4)
                for j in range(4):
                    z[j] = self.sigma_list[j] * np.random.normal(0, 1)

                z = np.dot(z, R)
                x_new[i] = x[i] + z

            x_new = np.clip(x_new, self.bounds[0], self.bounds[1])

            return x_new

        # BMO
        elif self.type in ['es_a7', 'es_a8', 'es_b7', 'es_b8']:
            iter = 5

            if c % iter == 0 and c != 0:
                if cs / c > 1 / 5:
                    self.sigma_list = self.sigma_list / self.update
                elif cs / c < 1 / 5:
                    self.sigma_list = self.sigma_list * self.update

            x_new = np.zeros((x.shape[0], 4))

            t1 = 1 / math.sqrt(2 * math.sqrt(x.shape[0]))
            t2 = 1 / math.sqrt(2 * x.shape[0])

            for i in range(4):
                self.sigma_list[i] = self.sigma_list[i] * math.exp((t1 * np.random.normal(0, 1)) +
                                                                   (t2 * np.random.normal(0, 1)))

            for i in range(4):
                self.bias_coe[i] = self.bias_coe[i] + 0.1 * np.random.normal(0, 1)

            for i in range(x.shape[0]):
                z = np.zeros(4)
                for j in range(4):
                    z[j] = self.sigma_list[j] * np.random.normal(self.bias_coe[j], 1)
                x_new[i] = x[i] + z

            x_new = np.clip(x_new, self.bounds[0], self.bounds[1])
            return x_new

# ----------------------------------------------------------------------------------------------------------------------
class DifferentialRecombination(Recombination):
    def __init__(self, type='de', bounds=(-np.infty, np.infty), params=None):
        super().__init__()
        self.type = type
        self.bounds = bounds

        assert (0. <= params['F'] <= 2.), 'F must be in [0, 2]'
        assert (0. < params['CR'] <= 1.), 'CR must be in (0, 1]'
        assert type in ['de', 'ade', 'revde', 'dex3'], 'type must be one in {de, dex3, ade, revde}'

        self.F = params['F']
        self.CR = params['CR']

    def recombination(self, x):
        indices_1 = np.arange(x.shape[0])
        # take first parent
        x_1 = x[indices_1]
        # assign second parent (ensure)
        indices_2 = np.random.permutation(x.shape[0])
        x_2 = x_1[indices_2]
        # assign third parent
        indices_3 = np.random.permutation(x.shape[0])
        x_3 = x_2[indices_3]

        if self.type == 'de':
            # differential mutation
            y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])

            # uniform crossover
            if self.CR < 1.:
                p_1 = bernoulli(self.CR, y_1.shape)
                y_1 = p_1 * y_1 + (1. - p_1) * x_1

            return (y_1), (indices_1, indices_2, indices_3)

        elif self.type == 'revde':
            y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])
            y_2 = np.clip(x_2 + self.F * (x_3 - y_1), self.bounds[0], self.bounds[1])
            y_3 = np.clip(x_3 + self.F * (y_1 - y_2), self.bounds[0], self.bounds[1])

            # uniform crossover
            if self.CR < 1.:
                p_1 = bernoulli(self.CR, y_1.shape)
                p_2 = bernoulli(self.CR, y_2.shape)
                p_3 = bernoulli(self.CR, y_3.shape)
                y_1 = p_1 * y_1 + (1. - p_1) * x_1
                y_2 = p_2 * y_2 + (1. - p_2) * x_2
                y_3 = p_3 * y_3 + (1. - p_3) * x_3

            return (y_1, y_2, y_3), (indices_1, indices_2, indices_3)

        elif self.type == 'ade':
            y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])
            y_2 = np.clip(x_2 + self.F * (x_3 - x_1), self.bounds[0], self.bounds[1])
            y_3 = np.clip(x_3 + self.F * (x_1 - x_2), self.bounds[0], self.bounds[1])

            # uniform crossover
            if self.CR < 1.:
                p_1 = bernoulli(self.CR, y_1.shape)
                p_2 = bernoulli(self.CR, y_2.shape)
                p_3 = bernoulli(self.CR, y_3.shape)
                y_1 = p_1 * y_1 + (1. - p_1) * x_1
                y_2 = p_2 * y_2 + (1. - p_2) * x_2
                y_3 = p_3 * y_3 + (1. - p_3) * x_3

            return (y_1, y_2, y_3), (indices_1, indices_2, indices_3)

        if self.type == 'dex3':
            # y1
            y_1 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])

            # uniform crossover
            if self.CR < 1.:
                p_1 = bernoulli(self.CR, y_1.shape)
                y_1 = p_1 * y_1 + (1. - p_1) * x_1

            # y2
            indices_1p = np.arange(x.shape[0])
            # take first parent
            x_1 = x[indices_1p]
            # assign second parent (ensure)
            indices_2p = np.random.permutation(x.shape[0])
            x_2 = x_1[indices_2p]
            # assign third parent
            indices_3p = np.random.permutation(x.shape[0])
            x_3 = x_2[indices_3p]

            y_2 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])

            # uniform crossover
            if self.CR < 1.:
                p_2 = bernoulli(self.CR, y_2.shape)
                y_2 = p_2 * y_2 + (1. - p_2) * x_1

            # y3
            indices_1p = np.arange(x.shape[0])
            # take first parent
            x_1 = x[indices_1p]
            # assign second parent (ensure)
            indices_2p = np.random.permutation(x.shape[0])
            x_2 = x_1[indices_2p]
            # assign third parent
            indices_3p = np.random.permutation(x.shape[0])
            x_3 = x_2[indices_3p]

            y_3 = np.clip(x_1 + self.F * (x_2 - x_3), self.bounds[0], self.bounds[1])

            # uniform crossover
            if self.CR < 1.:
                p_3 = bernoulli(self.CR, y_3.shape)
                y_3 = p_3 * y_3 + (1. - p_3) * x_1

            return (y_1, y_2, y_3), (indices_1, indices_2, indices_3)
        else:
            raise ValueError('Wrong name of the differential mutation!')
