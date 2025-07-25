import numpy as np
import random
import math

from random import shuffle, randint
from collections import defaultdict
from itertools import combinations
from math import sqrt, log
from scipy.special import binom

from mycode.utils.montecarlo import MonteCarlo
from mycode.utils.sampler import SampleWithoutReplacement
from mycode.utils.timer import timer_decorator

class StratifiedExact(MonteCarlo):

    def __init__() -> None:
        pass

    def reset(self):
        pass

    def run(self):
        pass 

class StratifiedNeymanExact(MonteCarlo):

    def __init__(self, n, m, V, logging=False) -> None:
        super().__init__(n, m, V, logging)  

        # Initialize the game with the number of players
        self.V = self.V(n)

        # Calculate the worth of an empty coalition
        self.v_empty = self.V.run([])      
        
        # Initialize the Shapley value for every player
        self.sh_i = {i:0 for i in range(self.n)}
        
        # Initialize the Shapley value, variance and counter for every player & position
        # BE AWARE: This is a dictionary of dictionaries instead of a plain dictionary
        self.sh_il = defaultdict(lambda: defaultdict(float))
        self.var_il = defaultdict(lambda: defaultdict(float))
        self.m_il = defaultdict(lambda: defaultdict(int))
        self.sample_without_stratas = defaultdict(lambda: defaultdict(list))

        # Initialize an alpha for distribution between the first and second stage
        self.alpha = 0.5

        # log the setup if needed
        self._log_start()

    def reset(self):
        pass

    def _sample_without_replacement(self, i, l, m, extra=False):
        """
        Generate a random sample without replacement from the set of integers 0, 1, ..., n-1.
        The sample will have size k and m samples will be generated.
        """
        # Initialize the sample
        temp_v = []

        if extra:
            available_indices = list(set(range(int(binom(self.n - 1, l)))) - set(self.sample_without_stratas[i][l]))
            sample = np.random.choice(available_indices, m, replace=False)
        else:
            sample = np.random.choice(range(int(binom(self.n - 1, l))), m, replace=False)

        for s in sample:
            permutation = np.array(SampleWithoutReplacement.get_permutation_from_index(n=self.n-1, k=l, index=s))
            permutation = list(np.where(permutation == i, self.n-1, permutation))
            temp_v.append(self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l]))
            self.evals += 1

        if not extra:
            self.sample_without_stratas[i][l] = list(sample)
            variance = np.var(temp_v, ddof=1) * (1 - len(sample) / binom(self.n - 1, l))
        else:
            self.sample_without_stratas[i][l] += list(sample)
            variance = None

        phi = np.sum(temp_v)
        del temp_v

        return phi, variance 
    
    def _sample_with_replacement(self, i, l, m, extra=False):

        temp_v = []
        permutation = list(range(self.n))
        permutation.remove(i)

        for _ in range(m):

            shuffle(permutation)
            temp_v.append(self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l]))
            self.evals += 1
        
        if not extra:
            variance = np.var(temp_v, ddof=1)
        else:
            variance = None

        phi = np.sum(temp_v)
        del temp_v

        return phi, variance

    @timer_decorator
    def run(self):

        # Compute the amount of samples for the first phase per player and position
        m_p = int(self.m / self.n**2)
        
        # Set the amount of iterations 
        self.ll = 0

        # As long as amount of samples his higher then the possible amount of samples
        # we need to adjust the amount of samples
        while m_p >= binom(self.n - 1, self.ll):
            self.ll += 1
            m_rem = int(self.m - self.n * 2 * np.sum([binom(self.n - 1, t) for t in range(self.ll)]))
            m_p = m_rem / (self.n * (self.n - 2 * self.ll))

        # Set the amount of samples for the first phase
        m_1 = math.floor(m_p * 0.5)

        self.summed_variance = 0

        # Calculate the outer strata exact
        for i in range(self.n):

            permutation = list(range(self.n))
            permutation.remove(i)
            
            # Iterate trough the strata
            for l in range(self.n):
                
                # Calculate the outer strata close to 0 or n exact
                if l < self.ll or l >= self.n - self.ll:

                    samples = list(combinations(permutation, l))
                    for sample in samples:
                        self.sh_il[i][l] += (self.V.run(list(sample)[0:l] + [i]) - self.V.run(list(sample)[0:l]))
                        self.m_il[i][l] += 1
                        self.evals += 1
                    self.var_il[i][l] = 0

                # Calculate the inner strata
                else:

                    self.m_il[i][l] = m_1
                    # check for sampling without replacement
                    if m_p >= 0.05 * binom(self.n - 1, l):

                        # sample without replacement
                        self.sh_il[i][l], self.var_il[i][l] = self._sample_without_replacement(i, l, m_1)

                    else:
                        
                        # sample with replacement
                        self.sh_il[i][l], self.var_il[i][l] = self._sample_with_replacement(i, l, m_1)

                        
                # Sum all variances
                self.summed_variance += self.var_il[i][l]

        # Calculate the amount of samples for the second phase per 
        # player and position based on the variance
        summed_m = 0
        
        # Compute the amount of samples for the second phase per player and position based on the variance
        for i in range(self.n):
            for l in range(self.n):

                if l < self.ll and l >= self.n - self.ll:
                    summed_m += self.m_il[i][l]
                    continue

                self.m_il[i][l] += (self.var_il[i][l] / self.summed_variance * (self.m - self.evals))
                self.m_il[i][l] = int(round(self.m_il[i][l]))

                if self.m_il[i][l] >= binom(self.n - 1, l):
                    self.m_il[i][l] = int(binom(self.n - 1, l))

                summed_m += self.m_il[i][l]

        flatten_variances = [
            ((i_, l_), value) for i_, l_dict in self.var_il.items()
            for l_, value in l_dict.items() if value > 0
        ]
        
        flatten_variances = sorted(flatten_variances, key=lambda x: x[1], reverse=True)

        indexer = 0
        while summed_m < self.m:

            (i, l), _ = flatten_variances[indexer % len(flatten_variances)] 
            

            if self.m_il[i][l] >= binom(self.n - 1, l):
                
                # remove from flatten variances
                flatten_variances.remove(((i, l), _))

                if len(flatten_variances) == 0:
                    # TODO 
                    # self.logger.warning('No more strata to add samples to')
                    break
                else:
                    continue

            self.m_il[i][l] += 1
            summed_m += 1
            
            if len(flatten_variances) <= 0:
                # TODO
                # self.logger.warning('No more strata to add samples to')
                break

            indexer += 1
        
        # print(self.evals, np.sum([self.m_il[i][l] for l in range(self.n) for i in range(self.n)]))
        for i in range(self.n):
            for l in range(self.n):

                if l >= self.ll and l < self.n - self.ll:

                    if len(self.sample_without_stratas[i][l]) > 0:
                        
                        # sample without replacement
                        phi, _ = self._sample_without_replacement(i, l, self.m_il[i][l] - m_1, extra=True)
                        self.sh_il[i][l] += phi

                    else:
                        # sample with replacement
                        phi, _ = self._sample_with_replacement(i, l, self.m_il[i][l] - m_1, extra=True)
                        self.sh_il[i][l] += phi

                # Compute the values for the strata
                self.sh_il[i][l] = self.sh_il[i][l] / self.m_il[i][l]

            # Compute the Shapley value
            self.sh_i[i] = np.mean(list(self.sh_il[i].values()))

        self._log_end()

        return self.sh_i

    @timer_decorator
    def more(self, m) -> dict:

        new_m = {i: {l: 0 for l in range(self.n)} for i in range(self.n)}
        summed_m = 0


        for i in range(self.n):
            for l in range(self.n):

                self.sh_il[i][l] = self.sh_il[i][l] * self.m_il[i][l]

                if l < self.ll or l >= self.n - self.ll:
                    continue

                new_m[i][l] += int(round((self.var_il[i][l] / self.summed_variance) * m))
                
                if new_m[i][l] + self.m_il[i][l] >= binom(self.n - 1, l):
                    new_m[i][l] = int(binom(self.n - 1, l) - self.m_il[i][l])

                summed_m += new_m[i][l]

        # print(new_m[14])
        flatten_variances = [
            ((i_, l_), value) for i_, l_dict in self.var_il.items()
            for l_, value in l_dict.items() if value > 0
        ]

        flatten_variances = sorted(flatten_variances, key=lambda x: x[1], reverse=True)

        indexer = 0
        # print(summed_m, m)
        while summed_m < m:

            (i, l), _ = flatten_variances[indexer % len(flatten_variances)]

            if new_m[i][l] + self.m_il[i][l] >= binom(self.n - 1, l):
                flatten_variances.remove(((i, l), _))

            else:
                new_m[i][l] += 1
                summed_m += 1
                indexer += 1

            if len(flatten_variances) <= 0:
                break

        for i in range(self.n):
            for l in range(self.n):

                if l >= self.ll and l < self.n - self.ll:

                    if len(self.sample_without_stratas[i][l]) > 0:
                        
                        # sample without replacement
                        phi, _ = self._sample_without_replacement(i, l, new_m[i][l], extra=True)

                    else:

                        # sample with replacement
                        phi, _ = self._sample_with_replacement(i, l, new_m[i][l], extra=True)

                    self.sh_il[i][l] += phi
                    self.m_il[i][l] += new_m[i][l]

                # Compute the values for the strata
                self.sh_il[i][l] = self.sh_il[i][l] / self.m_il[i][l]

            # Compute the Shapley value
            self.sh_i[i] = np.mean(list(self.sh_il[i].values()))

        return self.sh_i

class StratifiedBernsteinExact(MonteCarlo):
    
    def __init__(self, n, m, V, logging=False) -> None:
        super().__init__(n, m, V, logging)

        # Initialize the game with the number of players
        self.V = self.V(n)

        # Calculate the worth of an empty coalition
        self.v_empty = self.V.run([])      
        
        # Initialize the Shapley value for every player
        self.sh_i = {i:0 for i in range(self.n)}
        
        # Initialize the Shapley value, variance and counter for every player & position
        # BE AWARE: This is a dictionary of dictionaries instead of a plain dictionary
        self.sh_il = {i:{l:0 for l in range(self.n)} for i in range(self.n)}
        self.var_il = {i:{l:0 for l in range(self.n)} for i in range(self.n)}
        self.m_il = {i:{l:0 for l in range(self.n)} for i in range(self.n)}
        self.max_il = {i:{l:0 for l in range(self.n)} for i in range(self.n)}
        self.epsilon = {}
        self.sample_without_stratas = {i:{l:[] for l in range(self.n)} for i in range(self.n)}

        # Initialize the alpha for distribution between the first and second stage
        self.alpha = 0.5

        self._log_start()

    def reset():
        pass

    def _bernstein(self, delta, maxi, var, n):
        a = np.sqrt((2 * var * np.log(2/delta)) / n)
        b = (7 * maxi * np.log(2/delta)) / ((3 * n-1))
        return a + b
    
    def _calc_variance(self, var_0, mean_0, addit, new_n):
        return ((new_n-2)/(new_n-1))*var_0+(1/new_n)*((addit-mean_0)**2)

    @timer_decorator
    def run(self):

        # Compute the amount of samples for the first phase per player and position
        samples_per_player = int(self.m / self.n**2)
        
        # Set the amount of iterations 
        ll = 0

        # Set the total amount of variance
        self.summed_variance = 0

        # As long as amount of samples his higher then the possible amount of samples
        # we need to adjust the amount of samples
        while samples_per_player >= binom(self.n - 1, ll):
            ll += 1
            m_remaining = int(self.m - self.n * 2 * np.sum([binom(self.n - 1, t) for t in range(ll)]))
            samples_per_player = m_remaining / (self.n * (self.n - 2 * ll))

        # Set the amount of samples for the first phase
        m_phase1 = math.floor(samples_per_player * 0.5)

        # Calculate the outer strata exact
        for i in range(self.n):

            permutation = list(range(self.n))
            permutation.remove(i)

            # Calculate the outer strata close to 0 exact
            for l in range(0, ll):

                samples = list(combinations(permutation, l))
                for sample in samples:
                    self.sh_il[i][l] += (1/len(samples)) * (self.V.run(list(sample)[0:l] + [i]) - self.V.run(list(sample)[0:l]))
                    self.evals += 1
                self.var_il[i][l] = 0

            # Calculate the outer strata close to n exact
            for l in range(self.n - ll, self.n):

                samples = list(combinations(permutation, l))
                for sample in samples:
                    self.sh_il[i][l] += (1/len(samples)) * (self.V.run(list(sample)[0:l] + [i]) - self.V.run(list(sample)[0:l]))
                    self.evals += 1
                self.var_il[i][l] = 0

            # Calculate the inner strata
            for l in range(ll, self.n - ll):
                
                # Empty list for the marginals
                temp_v_list = []  

                # # If potentially more than 5% of the samples will be sampled sampling without 
                # # replacement will be performed. Otherwise sampling with replacement will be performed. 
                # if samples_per_player >= 0.05 * binom(self.n - 1, l):
                    
                #     # Select a set of permutations
                #     indexed_permutations = np.random.choice(range(int(binom(self.n - 1, l))), m_phase1, replace=False)
                #     for index in indexed_permutations:
                #         permutation = np.array(SampleWithoutReplacement.get_permutation_from_index(n=self.n-1, k=l, index=index))
                #         permutation = list(np.where(permutation == i, self.n-1, permutation))
                #         temp_v_list.append((self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l])))
                #         self.evals += 1

                #     # Save the evaluated indexed permutations
                #     self.sample_without_stratas[i][l] = list(indexed_permutations)

                #     # Calculate the variance variables
                #     self.var_il[i][l] = np.var(temp_v_list, ddof=1) * (1 - len(temp_v_list) / binom(self.n - 1, l))

                # else:
                
                permutation = list(range(self.n))
                permutation.remove(i)

                for _ in range(m_phase1):

                    shuffle(permutation)
                    temp_v_list.append((self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l])))
                    self.evals += 1

                # Calculate the Shapley value and variance for the inner strata
                self.var_il[i][l] = np.var(temp_v_list, ddof=1)

                # Calculate the variance variables
                self.summed_variance += self.var_il[i][l]

                # Calculate the Shapley value, the max and the amount of 
                # samples for the inner strata
                self.sh_il[i][l] = np.mean(temp_v_list)
                self.max_il[i][l] = np.max(temp_v_list)
                self.m_il[i][l] = m_phase1

        # Comput the theoretical error for every strata
        for i in range(self.n):
            for l in range(ll, self.n - ll):

                self.epsilon[(i, l)] = self._bernstein(
                    delta=0.01, 
                    maxi=self.max_il[i][l], 
                    var=self.var_il[i][l], 
                    n=self.m_il[i][l]) - self._bernstein(delta=0.01, 
                                                         maxi=self.max_il[i][l], 
                                                         var=self.var_il[i][l], 
                                                         n=self.m_il[i][l] + 1)

        # Compute the amount of samples for the second phase per player and position based on 
        # the error of the Bernstein inequality        
        while self.evals < self.m:
            
            # Get the stratum with the highest theoretical error according to Bernstein
            max_red = max(self.epsilon, key=self.epsilon.get)
            i, l = max_red
            i, l = int(i), int(l)

            # if len(self.sample_without_stratas[i][l]) > 0:
                
            #     # Select a random sample from the non-evaluated set
            #     remaining_samples = list(set(range(int(binom(self.n - 1, l)))) - set(self.sample_without_stratas[i][l]))

            #     # If there exist still samples, take one
            #     if len(remaining_samples) > 0:
            #         selected_sample = np.random.choice(list(remaining_samples))
            #     else:
            #         break

            #     # Add the sample to the evaluated set.
            #     self.sample_without_stratas[i][l].append(selected_sample)
                
            #     # Get the permutation responding to the selected index
            #     permutation = np.array(SampleWithoutReplacement.get_permutation_from_index(
            #         n=self.n-1,
            #         k=l,
            #         index=selected_sample
            #     ))
            #     permutation = list(np.where(permutation == i, self.n-1, permutation))

            #     # Calculate the marginal contribution
            #     marginal = self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l])
            #     self.evals += 1

            #     # Adjust all the values
            #     self.m_il[i][l] += 1
            #     self.sh_il[i][l] = (self.sh_il[i][l] * (self.m_il[i][l] - 1) + marginal) / self.m_il[i][l]

            #     # Calculate the new variance and adjust for the sampling without replacement
            #     #TODO test variance.. comparison
            #     var0 = self.var_il[i][l] / (1 - (len(self.sample_without_stratas[i][l]) - 1) / binom(self.n - 1, l))
            #     var1 = self._calc_variance(var0, self.sh_il[i][l], marginal, self.m_il[i][l]) 
            #     self.var_il[i][l] = var1 * (1 - (len(self.sample_without_stratas[i][l]) / binom(self.n - 1, l)))

            #     # If the calculated marginal is larger than the maximum, adjust the maximum
            #     if marginal > self.max_il[i][l]:
            #         self.max_il[i][l] = marginal

            #     # Adjust the theoretical error and adjust for the maximum amount
            #     # of samples that can be drawm from a stratum
            #     if self.m_il[i][l] >= int(binom(self.n - 1, l)):
            #         self.epsilon[(i, l)] = 0
            #     else:
            #         self.epsilon[(i, l)] = self._bernstein(
            #             delta=0.01, 
            #             maxi=self.max_il[i][l], 
            #             var=self.var_il[i][l], 
            #             n=self.m_il[i][l]) - self._bernstein(delta=0.01, 
            #                                                     maxi=self.max_il[i][l], 
            #                                                     var=self.var_il[i][l], 
            #                                                     n=self.m_il[i][l] + 1)            

            # else:

            # Calculate the Shapley value for the stratum
            permutation = list(range(self.n))
            permutation.remove(i)
            shuffle(permutation)
            marginal = self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l])
            self.evals += 1

            # Adjust all the values
            self.m_il[i][l] += 1
            self.sh_il[i][l] = (self.sh_il[i][l] * (self.m_il[i][l] - 1) + marginal) / self.m_il[i][l]
            self.var_il[i][l] = self._calc_variance(self.var_il[i][l], self.sh_il[i][l], marginal, self.m_il[i][l])

            # If the calculated marginal is larger than the maximum, adjust the maximum
            if marginal > self.max_il[i][l]:
                self.max_il[i][l] = marginal

            # Adjust the theoretical error and adjust for the maximum amount
            # of samples that can be drawm from a stratum
            if self.m_il[i][l] >= binom(self.n - 1, l):
                self.epsilon[(i, l)] = 0
            else:
                self.epsilon[(i, l)] = self._bernstein(
                    delta=0.01, 
                    maxi=self.max_il[i][l], 
                    var=self.var_il[i][l], 
                    n=self.m_il[i][l]) - self._bernstein(delta=0.01, 
                                                            maxi=self.max_il[i][l], 
                                                            var=self.var_il[i][l], 
                                                            n=self.m_il[i][l] + 1)
        
        # Calculate the Shapley value for every player combining his strata
        for i in range(self.n):
            for l in range(self.n):
                self.sh_i[i] += self.sh_il[i][l] * (1.0/self.n)

        self._log_end()

        return self.sh_i
    
    @timer_decorator
    def more(self, m) -> dict:

        self.m += m
        while self.evals < self.m:
            
            # Get the stratum with the highest theoretical error according to Bernstein
            max_red = max(self.epsilon, key=self.epsilon.get)
            i, l = max_red
            i, l = int(i), int(l)

            if len(self.sample_without_stratas[i][l]) > 0:
                
                # Select a random sample from the non-evaluated set
                remaining_samples = list(set(range(int(binom(self.n - 1, l)))) - set(self.sample_without_stratas[i][l]))

                # If there exist still samples, take one
                if len(remaining_samples) > 0:
                    selected_sample = np.random.choice(list(remaining_samples))
                else:
                    break

                # Add the sample to the evaluated set.
                self.sample_without_stratas[i][l].append(selected_sample)
                
                # Get the permutation responding to the selected index
                permutation = np.array(SampleWithoutReplacement.get_permutation_from_index(
                    n=self.n-1,
                    k=l,
                    index=selected_sample
                ))
                permutation = list(np.where(permutation == i, self.n-1, permutation))

                # Calculate the marginal contribution
                marginal = self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l])
                self.evals += 1

                # Adjust all the values
                self.m_il[i][l] += 1
                self.sh_il[i][l] = (self.sh_il[i][l] * (self.m_il[i][l] - 1) + marginal) / self.m_il[i][l]

                # Calculate the new variance and adjust for the sampling without replacement
                #TODO test variance.. comparison
                var0 = self.var_il[i][l] / (1 - (len(self.sample_without_stratas[i][l]) - 1) / binom(self.n - 1, l))
                var1 = self._calc_variance(var0, self.sh_il[i][l], marginal, self.m_il[i][l]) 
                self.var_il[i][l] = var1 * (1 - (len(self.sample_without_stratas[i][l]) / binom(self.n - 1, l)))

                # If the calculated marginal is larger than the maximum, adjust the maximum
                if marginal > self.max_il[i][l]:
                    self.max_il[i][l] = marginal

                # Adjust the theoretical error and adjust for the maximum amount
                # of samples that can be drawm from a stratum
                if self.m_il[i][l] >= int(binom(self.n - 1, l)):
                    self.epsilon[(i, l)] = 0
                else:
                    self.epsilon[(i, l)] = self._bernstein(
                        delta=0.01, 
                        maxi=self.max_il[i][l], 
                        var=self.var_il[i][l], 
                        n=self.m_il[i][l]) - self._bernstein(delta=0.01, 
                                                                maxi=self.max_il[i][l], 
                                                                var=self.var_il[i][l], 
                                                                n=self.m_il[i][l] + 1)            

            else:

                # Calculate the Shapley value for the stratum
                permutation = list(range(self.n))
                permutation.remove(i)
                shuffle(permutation)
                marginal = self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l])
                self.evals += 1

                # Adjust all the values
                self.m_il[i][l] += 1
                self.sh_il[i][l] = (self.sh_il[i][l] * (self.m_il[i][l] - 1) + marginal) / self.m_il[i][l]
                self.var_il[i][l] = self._calc_variance(self.var_il[i][l], self.sh_il[i][l], marginal, self.m_il[i][l])

                # If the calculated marginal is larger than the maximum, adjust the maximum
                if marginal > self.max_il[i][l]:
                    self.max_il[i][l] = marginal

                # Adjust the theoretical error and adjust for the maximum amount
                # of samples that can be drawm from a stratum
                if self.m_il[i][l] >= binom(self.n - 1, l):
                    self.epsilon[(i, l)] = 0
                else:
                    self.epsilon[(i, l)] = self._bernstein(
                        delta=0.01, 
                        maxi=self.max_il[i][l], 
                        var=self.var_il[i][l], 
                        n=self.m_il[i][l]) - self._bernstein(delta=0.01, 
                                                                maxi=self.max_il[i][l], 
                                                                var=self.var_il[i][l], 
                                                                n=self.m_il[i][l] + 1)
        
        # Calculate the Shapley value for every player combining his strata
        for i in range(self.n):
            self.sh_i[i] = 0
            for l in range(self.n):
                self.sh_i[i] += self.sh_il[i][l] * (1.0/self.n)

        return self.sh_i

class ComplementaryExact(MonteCarlo):

    def __init__(self, n, m, V, logging=False) -> None:
        super().__init__(n, m, V, logging)  

        # Initialize the game with the number of players
        self.V = self.V(n)

        # Shapley value accumulators 
        self.sv = {i: 0 for i in range(self.n)}
        self.sv_ij = defaultdict(lambda: defaultdict(float))
        self.m_ij = defaultdict(lambda: defaultdict(int))

        self._log_start()

    def reset(self):
        self.sv = {i: 0 for i in range(self.n)}
        self.sv_ij = defaultdict(lambda: defaultdict(float))
        self.m_ij = defaultdict(lambda: defaultdict(int))

    def run(self) -> dict:

        u = self.V.run(list(range(self.n)))
        self.evals += 1
        for player in range(self.n):
            self.sv_ij[player][self.n] += u
            self.m_ij[player][self.n] += 1

        used = 1
        ll = 1
        rem_strata = math.ceil((self.n-1)/2)

        while (self.m - used) / rem_strata >= binom(self.n, ll):
            used += binom(self.n, ll)
            rem_strata -= 1
            ll += 1

        for j in range(1, ll):

            for sample in combinations(list(range(self.n)), j):
                
                sample_a = list(sample)
                sample_b = list(set(range(self.n)) - set(sample_a))

                u = self.V.run(sample_a) - self.V.run(sample_b)
                self.evals += 1

                for i in sample_a:
                    self.sv_ij[i][j] += u
                    self.m_ij[i][j] += 1
                
                for i in sample_b:
                    self.sv_ij[i][self.n - j] -= u
                    self.m_ij[i][self.n - j] += 1

        # Main loop for Monte Carlo iterations
        for k in range(self.m - self.evals):

            # Step 3: Generate a random permutation of players
            permutation = list(range(self.n))
            shuffle(permutation)

            # Step 4: Randomly select an index i from the permutation
            j = randint(ll, self.n - ll)  # Note: randint is inclusive on both ends

            # Step 5: Define coalition S with the first i players in the permutation
            S = [permutation[i] for i in range(j)]

            # Step 6: N\S is the complement of S
            N_minus_S = [permutation[i] for i in range(j, self.n)]

            # Step 7: Compute the complementary contribution
            u = self.V.run(S) - self.V.run(N_minus_S)
            self.evals += 1

            # Step 8-10: Add contribution to members of S
            # for i in S 
            for i in range(j):
                self.sv_ij[permutation[i]][j] += u
                self.m_ij[permutation[i]][j] += 1

            # Step 11-13: Subtract contribution from members of N\S
            # for i in N_minus_S
            for i in range(j, self.n):
                self.sv_ij[permutation[i]][self.n - j] -= u
                self.m_ij[permutation[i]][self.n - j] += 1

        # Step 14-15: Normalize the cumulative sums to get the expected Shapley value
        for i in range(self.n):
            self.sv[i] = sum(self.sv_ij[i][j] / self.m_ij[i][j] if self.m_ij[i][j] > 0 else 0 for j in range(1, self.n+1)) / self.n

        self._log_end()        

        return self.sv

class ComplementaryNeymanExact(MonteCarlo):

    def __init__(self, n, m, V, logging=False, init_m=50) -> None:
        super().__init__(n, m, V, logging)  

        # Initialize the game with the number of players
        self.V = self.V(n)
        self.init_m = init_m #TODO

        # Shapley value accumulators 
        self.sv = {i: 0 for i in range(self.n)}
        self.sv_ij = defaultdict(lambda: defaultdict(list))
        self.m_ij = defaultdict(lambda: defaultdict(int))
        self.var = defaultdict(lambda: defaultdict(float))
        self.sv_ij_float = defaultdict(lambda: defaultdict(float))
        self.fraction = defaultdict(float)

        self._log_start()
           

    def reset(self):
        self.sv = {i: 0 for i in range(self.n)}
        self.sv_ij = defaultdict(lambda: defaultdict(list))
        self.m_ij = defaultdict(lambda: defaultdict(int))
        self.var = defaultdict(lambda: defaultdict(float))
        self.sv_ij_float = defaultdict(lambda: defaultdict(float))

    @timer_decorator
    def run(self) -> dict:

        u = self.V.run(list(range(self.n)))
        self.evals += 1
        for player in range(self.n):
            self.sv_ij_float[player][self.n] += u
            self.m_ij[player][self.n] += 1

        used = 1
        ll = 1
        rem_strata = math.ceil((self.n-1)/2)
        trigger = False

        while (self.m - used) / rem_strata >= binom(self.n, ll):
            used += binom(self.n, ll)
            rem_strata -= 1
            ll += 1
            if rem_strata == 0:
                trigger = True
                break

        for j in range(1, ll):

            for sample in combinations(list(range(self.n)), j):
                
                sample_a = list(sample)
                sample_b = list(set(range(self.n)) - set(sample_a))

                u = self.V.run(sample_a) - self.V.run(sample_b)
                self.evals += 1

                for i in sample_a:
                    self.sv_ij_float[i][j] += u
                    self.m_ij[i][j] += 1
                
                for i in sample_b:
                    self.sv_ij_float[i][self.n - j] -= u
                    self.m_ij[i][self.n - j] += 1

        if trigger:
            for i in range(self.n):
                self.sv[i] = sum(self.sv_ij_float[i][j] / self.m_ij[i][j] if self.m_ij[i][j] > 0 else 0 for j in range(1, self.n+1)) / self.n
            self._log_end()
            return self.sv

        for i in range(self.n):
            for j in range(ll, self.n - ll + 1):
                for k in range(self.init_m):

                    if self.m_ij[i][j] >= self.init_m:
                        continue
                    
                    # Step 3: Generate a random permutation of players
                    permutation = list(range(self.n))
                    shuffle(permutation)

                    # Step 5: Define coalition S with the first i players in the permutation
                    S = permutation[:j]

                    # Step 6: N\S is the complement of S
                    N_minus_S = permutation[j:]

                    # If needed add the player under investigation to the first coalition
                    if self.m_ij[i][j] < 2 and i in N_minus_S:
                        player_to_switch = S[0]

                        S.remove(player_to_switch)
                        S.append(i)

                        N_minus_S.remove(i)
                        N_minus_S.append(player_to_switch)

                    # Step 7: Compute the complementary contribution
                    u = self.V.run(S) - self.V.run(N_minus_S)
                    self.evals += 1

                    # Step 8-10: Add contribution to members of S
                    for i_ in S:
                        # player_index = permutation[i]
                        self.sv_ij[i_][j].append(u)
                        self.m_ij[i_][j] += 1

                    # Step 11-13: Subtract contribution from members of N\S
                    for i_ in N_minus_S:
                        # player_index = permutation[i]
                        self.sv_ij[i_][self.n - j].append(-u)
                        self.m_ij[i_][self.n - j] += 1


        sigma_j = np.zeros(self.n)
        sigma_n_j = np.zeros(self.n)
        var_sum = 0

        for i in range(self.n):
            for j in range(ll, self.n - ll + 1):
                if len(self.sv_ij[i][j]) < 2:
                    print('error', self.sv_ij[i][j], i, j)
                self.var[i][j] = np.var(self.sv_ij[i][j], ddof=1)

        for j in range(math.ceil(self.n / 2) - 1, self.n - ll + 1):
            for i in range(self.n):
                sigma_j[j] += self.var[i][j] / (j + 1)
                if self.n - j - 2 < 0:
                    sigma_n_j[j] += 0
                else:
                    sigma_n_j[j] += self.var[i][self.n - j - 2] / (self.n - j - 1)
            var_sum += np.sqrt(sigma_j[j] + sigma_n_j[j])
        
        remaining_m = self.m - self.evals
        allocated_m = np.zeros(self.n)

        for j in range(math.ceil(self.n / 2) - 1, self.n - ll + 1):
            self.fraction[j] =  (np.sqrt(sigma_j[j] + sigma_n_j[j]) / var_sum)
            allocated_m[j] = max(0, round(remaining_m * self.fraction[j]))

        ### Add while loop to add remaining samples
        for i in range(self.n):
            for j in range(ll, self.n - ll + 1):
                self.sv_ij_float[i][j] = np.sum(self.sv_ij[i][j])
        
        for j in range(ll, self.n - ll + 1):
            for _ in range(int(allocated_m[j])):
                permutation = list(range(self.n))
                shuffle(permutation)
                S = [permutation[i] for i in range(j)]
                N_minus_S = [permutation[i] for i in range(j, self.n)]
                
                u = self.V.run(S) - self.V.run(N_minus_S)
                self.evals += 1

                # Step 8-10: Add contribution to members of S
                for i in range(j):
                    player_index = permutation[i]
                    self.sv_ij_float[player_index][j] += u
                    self.m_ij[player_index][j] += 1

                # Step 11-13: Subtract contribution from members of N\S
                for i in range(j, self.n):
                    player_index = permutation[i]
                    self.sv_ij_float[player_index][self.n - j] -= u
                    self.m_ij[player_index][self.n - j] += 1


        # Step 14-15: Normalize the cumulative sums to get the expected Shapley value
        for i in range(self.n):
            self.sv[i] = sum(self.sv_ij_float[i][j] / self.m_ij[i][j] if self.m_ij[i][j] > 0 else 0 for j in range(1, self.n+1)) / self.n

        self._log_end()
        return self.sv
    
    @timer_decorator
    def more(self, m):

        for i in range(self.n):
            for j in self.fraction.keys():
                
                allocated_m = int(round(m * self.fraction[j]))
                for _ in range(allocated_m):
                    permutation = list(range(self.n))
                    shuffle(permutation)
                    S = [permutation[i] for i in range(j)]
                    N_minus_S = [permutation[i] for i in range(j, self.n)]
                    
                    u = self.V.run(S) - self.V.run(N_minus_S)
                    self.evals += 1

                    # Step 8-10: Add contribution to members of S
                    for i in range(j):
                        player_index = permutation[i]
                        self.sv_ij_float[player_index][j] += u
                        self.m_ij[player_index][j] += 1

                    # Step 11-13: Subtract contribution from members of N\S
                    for i in range(j, self.n):
                        player_index = permutation[i]
                        self.sv_ij_float[player_index][self.n - j] -= u
                        self.m_ij[player_index][self.n - j] += 1

        for i in range(self.n):
            self.sv[i] = sum(self.sv_ij_float[i][j] / self.m_ij[i][j] if self.m_ij[i][j] > 0 else 0 for j in range(1, self.n+1)) / self.n

        return self.sv

class ComplementaryBernsteinExact(MonteCarlo):

    def __init__(self, n, m, V, logging=False, init_m=30) -> None:
        super().__init__(n, m, V, logging)  

        # Initialize the game with the number of players
        self.V = self.V(n)

        # Shapley value accumulators 
        self.sv = {i: 0 for i in range(self.n)}
        self.sv_ij = defaultdict(lambda: defaultdict(list))
        self.m_ij = defaultdict(lambda: defaultdict(int))
        self.var = defaultdict(lambda: defaultdict(float))
        self.sv_ij_float = defaultdict(lambda: defaultdict(float))

        self.init_m = init_m

        self.thelta = 5
        self.r = 10
        self.flag = False

        self._log_start()
           

    def reset(self):
        self.sv = {i: 0 for i in range(self.n)}
        self.sv_ij = defaultdict(lambda: defaultdict(list))
        self.m_ij = defaultdict(lambda: defaultdict(int))
        self.var = defaultdict(lambda: defaultdict(float))
        self.sv_ij_float = defaultdict(lambda: defaultdict(float))

    def _calculate_var(self, i, j):
        
        if self.flag:
            # This block is executed when flag is True
            self.var[i][j] = np.var(self.sv_ij[i][j], ddof=1)
            self.var[i][j] = (
                sqrt(2 * self.var[i][j] * log(2 / self.thelta) / len(self.sv_ij[i][j])) +
                7 * log(2 / self.thelta) / (3 * (len(self.sv_ij[i][j]) - 1))
            )
        else:
            # This block is executed when flag is False
            self.var[i][j] = np.var(self.sv_ij[i][j])
            self.var[i][j] = (
                sqrt(2 * self.var[i][j] * self.thelta / len(self.sv_ij[i][j])) +
                3 * self.r * self.thelta / len(self.sv_ij[i][j])
            )

    def _test_select_coaltions(self, ll):
        perm = list(range(self.n))
        shuffle(perm)
        j = random.randint(ll, self.n - ll)
        _ = np.nan
        return _, (perm, j)
    

    def _select_coaltions(self, ll):

        max_score = 0
        
        idx = np.arange(self.n)

        for breakp in range(math.ceil(self.n / 2) - 1, self.n - ll + 1):

            np.random.shuffle(idx)
            
            l, fl = [], []

            for pos in range(breakp):

                temp = self.var[idx[pos]][breakp] - self.var[idx[pos]][self.n - breakp]
                l.append(temp)
                fl.append(-temp)

            r, fr = [], []

            for pos in range(breakp, self.n):

                temp = self.var[idx[pos]][self.n - breakp] - self.var[idx[pos]][breakp]
                r.append(temp)
                fr.append(-temp)
            
            sli = np.argsort(fl)
            slr = np.argsort(fr)

            score = 0
            p = 0

            while p < breakp and p < self.n - breakp and score + l[sli[p]] + r[slr[p]] < score:
                score += l[sli[p]] + r[slr[p]]
                idx[sli[p]], idx[breakp + slr[p]] = idx[breakp + slr[p]], idx[sli[p]]
                p += 1

            score = 0
            for pos in range(breakp):
                score += self.var[idx[pos]][breakp]

            for pos in range(breakp, self.n):
                score += self.var[idx[pos]][self.n - breakp]
        
            if score > max_score:
                max_score = score
                max_perm = idx
                max_breakp = breakp

        return max_score, (max_perm, max_breakp)

    @timer_decorator
    def run(self) -> dict:

        u = self.V.run(list(range(self.n)))
        self.evals += 1
        for player in range(self.n):
            self.sv_ij_float[player][self.n] += u
            self.m_ij[player][self.n] += 1

        used = 1
        ll = 1
        rem_strata = math.ceil((self.n-1)/2)
        trigger = False

        while (self.m - used) / rem_strata >= binom(self.n, ll):
            used += binom(self.n, ll)
            rem_strata -= 1
            ll += 1
            if rem_strata == 0:
                trigger = True
                break

        for j in range(1, ll):

            for sample in combinations(list(range(self.n)), j):
                
                sample_a = list(sample)
                sample_b = list(set(range(self.n)) - set(sample_a))

                u = self.V.run(sample_a) - self.V.run(sample_b)
                self.evals += 1

                for i in sample_a:
                    self.sv_ij_float[i][j] += u
                    self.m_ij[i][j] += 1
                
                for i in sample_b:
                    self.sv_ij_float[i][self.n - j] -= u
                    self.m_ij[i][self.n - j] += 1

        if trigger:
            for i in range(self.n):
                self.sv[i] = sum(self.sv_ij_float[i][j] / self.m_ij[i][j] if self.m_ij[i][j] > 0 else 0 for j in range(1, self.n+1)) / self.n
            self._log_end()
            return self.sv

        for i in range(self.n):
            for j in range(ll, self.n - ll + 1):
                for k in range(self.init_m):

                    if self.m_ij[i][j] >= self.init_m:
                        continue
                    
                    # Step 3: Generate a random permutation of players
                    permutation = list(range(self.n))
                    shuffle(permutation)

                    # Step 5: Define coalition S with the first i players in the permutation
                    S = permutation[:j]

                    # Step 6: N\S is the complement of S
                    N_minus_S = permutation[j:]

                    # If needed add the player under investigation to the first coalition
                    if self.m_ij[i][j] < 2 and i in N_minus_S:
                        player_to_switch = S[0]

                        S.remove(player_to_switch)
                        S.append(i)

                        N_minus_S.remove(i)
                        N_minus_S.append(player_to_switch)

                    # Step 7: Compute the complementary contribution
                    u = self.V.run(S) - self.V.run(N_minus_S)
                    self.evals += 1

                    # Step 8-10: Add contribution to members of S
                    for i_ in S:
                        # player_index = permutation[i]
                        self.sv_ij[i_][j].append(u)
                        self.m_ij[i_][j] += 1

                    # Step 11-13: Subtract contribution from members of N\S
                    for i_ in N_minus_S:
                        # player_index = permutation[i]
                        self.sv_ij[i_][self.n - j].append(-u)
                        self.m_ij[i_][self.n - j] += 1


        ## PU IT HERE
        for i in range(self.n):
            for j in range(ll, self.n - ll + 1):
                self._calculate_var(i, j)
        
        for _ in range(self.m - self.evals):
            score_, (perm, j) = self._select_coaltions(ll)

            # print(perm, j)

            # Step 5: Define coalition S with the first i players in the permutation
            S = perm[:j]

            # Step 6: N\S is the complement of S
            N_minus_S = perm[j:]

            u = self.V.run(S) - self.V.run(N_minus_S)
            self.evals += 1

            for i in S:
                self.sv_ij[i][j].append(u)
                self.m_ij[i][j] += 1
                self._calculate_var(i, j)
            
            for i in N_minus_S:
                self.sv_ij[i][self.n - j].append(-u)
                self.m_ij[i][self.n - j] += 1
                self._calculate_var(i, self.n - j)


        for i in range(self.n):
            for j in range(ll, self.n - ll + 1):
                self.sv_ij_float[i][j] = np.sum(self.sv_ij[i][j])

        # Step 14-15: Normalize the cumulative sums to get the expected Shapley value
        for i in range(self.n):
            self.sv[i] = sum(self.sv_ij_float[i][j] / self.m_ij[i][j] if self.m_ij[i][j] > 0 else 0 for j in range(1, self.n+1)) / self.n

        self._log_end()

        return self.sv   

    @timer_decorator
    def more(self, m):
        pass
                



