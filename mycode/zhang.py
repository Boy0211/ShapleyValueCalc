import numpy as np
from random import shuffle, randint
from scipy.special import binom, comb
import math
from collections import defaultdict
from itertools import combinations
import time

from mycode.utils.montecarlo import MonteCarlo
from mycode.utils.timer import timer_decorator

class Complementary(MonteCarlo):

    def __init__(self, n, m, V, logging=False) -> None:
        super().__init__(n, m, V, logging)  

        # Initialize the game with the number of players
        self.V = self.V(n)

        # Shapley value accumulators 
        self.sv = {i: 0 for i in range(self.n)}
        self.sv_ij = defaultdict(lambda: defaultdict(list))
        # self.sv_ij = defaultdict(lambda: defaultdict(float))
        # self.sv_ij = np.zeros((self.n, self.n + 1))
        # self.m_ij = defaultdict(lambda: defaultdict(int))
        # self.m_ij = np.zeros((self.n, self.n + 1))

        self._log_start()

    def reset(self):
        self.sv = {i: 0 for i in range(self.n)}
        self.sv_ij = defaultdict(lambda: defaultdict(float))
        self.m_ij = defaultdict(lambda: defaultdict(int))

    @timer_decorator
    def run(self) -> dict:
        
        permutation = list(range(self.n))

        # Main loop for Monte Carlo iterations
        for k in range(self.m):

            # Step 3: Generate a random permutation of players
            # permutation = list(range(self.n))
            shuffle(permutation)

            # Step 4: Randomly select an index i from the permutation
            j = randint(1, self.n)  # Note: randint is inclusive on both ends

            # Step 5: Define coalition S with the first i players in the permutation
            S = permutation[:j] # [permutation[i] for i in range(j)]

            # Step 6: N\S is the complement of S
            N_minus_S = permutation[j:]  #[permutation[i] for i in range(j, self.n)]

            # Step 7: Compute the complementary contribution
            u = self.V.run(S) - self.V.run(N_minus_S)
            self.evals += 1

            # Step 8-10: Add contribution to members of S
            # for i in range(j):
            #     # player_index = permutation[i]
            #     self.sv_ij[permutation[i]][j] += u
            #     self.m_ij[permutation[i]][j] += 1

            for i in S:
                self.sv_ij[i][j].append(u)
                # self.m_ij[permutation[i]][j] += 1                

            # Step 11-13: Subtract contribution from members of N\S
            # for i in range(j, self.n):
            #     # player_index = 
            #     self.sv_ij[permutation[i]][self.n - j] -= u
            #     self.m_ij[permutation[i]][self.n - j] += 1

            for i in N_minus_S:
                # player_index = 
                self.sv_ij[i][self.n - j].append(-u)
                # self.m_ij[permutation[i]][self.n - j] += 1

        # Step 14-15: Normalize the cumulative sums to get the expected Shapley value
        # for i in range(self.n):
        #     # self.sv[i] = sum(self.sv_ij[i][j] / self.m_ij[i][j] if self.m_ij[i][j] > 0 else 0 for j in range(1, self.n+1)) / self.n
        #     self.sv[i] = sum(sum(self.sv_ij[i][j]) / len(self.sv_ij[i][j]) if self.sv_ij[i][j] else 0 for j in range(1, self.n+1)) / self.n

        for i in range(self.n):
            total = 0.0
            for j in range(1, self.n+1):
                if self.sv_ij[i][j]:
                    total += sum(self.sv_ij[i][j]) / len(self.sv_ij[i][j])
            self.sv[i] = total / self.n

        self._log_end()


        # Step 16: Return the Shapley values
        return self.sv
    
    @timer_decorator
    def more(self, m) -> dict:

        for k in range(m):

            # Step 3: Generate a random permutation of players
            permutation = list(range(self.n))
            shuffle(permutation)

            # Step 4: Randomly select an index i from the permutation
            j = randint(1, self.n)

            # Step 5: Define coalition S with the first i players in the permutation
            S = permutation[:j]
            N_minus_S = permutation[j:]

            # Step 6: Compute the complementary contribution
            u = self.V.run(S) - self.V.run(N_minus_S)
            self.evals += 1

            # Step 8-10: Add contribution to members of S
            for i in S:
                self.sv_ij[i][j] += u
                self.m_ij[i][j] += 1

            # Step 11-13: Subtract contribution from members of N\S
            for i in N_minus_S:
                self.sv_ij[i][self.n - j] -= u
                self.m_ij[i][self.n - j] += 1

        # Step 14-15: Normalize the cumulative sums to get the expected Shapley value
        for i in range(self.n):
            self.sv[i] = sum(self.sv_ij[i][j] / self.m_ij[i][j] if self.m_ij[i][j] > 0 else 0 for j in range(1, self.n+1)) / self.n

        return self.sv

    
class ComplementaryNeyman(MonteCarlo):

    def __init__(self, n, m, V, logging=False, init_m=50) -> None:
        super().__init__(n, m, V, logging)  

        # Initialize the game with the number of players
        self.V = self.V(n)

        # Shapley value accumulators 
        self.sv = {i:0 for i in range(self.n)}
        self.sv_ij = defaultdict(lambda: defaultdict(list))
        self.sv_ij_sum = defaultdict(lambda: defaultdict(float))
        self.m_ij = defaultdict(lambda: defaultdict(int))
        self.var = defaultdict(lambda: defaultdict(float))
        self.init_m = init_m
        self.coef = [comb(n - 1, _) for _ in range(n)]
        self.fraction = np.zeros(self.n)

        self._log_start()
           
    def reset(self):
        self.sv = {i: 0 for i in range(self.n)}
        self.sv_ij = defaultdict(lambda: defaultdict(list))
        self.m_ij = defaultdict(lambda: defaultdict(int))
        self.var = defaultdict(lambda: defaultdict(float))
        self.sv_ij_sum = defaultdict(lambda: defaultdict(float))

    @timer_decorator
    def run(self) -> dict:

        for i in range(self.n):
            permutation = list(range(self.n))
            permutation.pop(i)

            for j in range(self.n):

                if len(self.sv_ij[i][j]) >= self.init_m:
                    continue
                
                if len(self.sv_ij[i][j]) >= self.coef[j]:
                    continue

                shuffle(permutation)
                S = permutation[:j] + [i]
                N_minus_S = permutation[j:]

                u = self.V.run(S) - self.V.run(N_minus_S)
                self.evals += 1
                self.sv_ij[i][j].append(u)

                for i_ in S:
                    self.sv_ij[i_][j].append(u)
                
                for i_ in N_minus_S:
                    self.sv_ij[i_][len(N_minus_S) - 1].append(-u)

        del permutation

        for i in range(self.n):
            for j in range(self.n):
                if j != 0 and j != self.n - 1:
                    self.var[i][j] = np.var(self.sv_ij[i][j], ddof=1)
                else:
                    self.var[i][j] = 0
                self.m_ij[i][j] = len(self.sv_ij[i][j])
                # self.sv_ij[i][j] = np.sum(self.sv_ij[i][j])
                self.sv_ij_sum[i][j] = np.sum(self.sv_ij[i][j])   

        total_var = 0
        sigma_j = np.zeros(self.n)
        sigma_n_j = np.zeros(self.n)

        for j in range(math.ceil(self.n / 2) - 1, self.n):
            for i in range(self.n):
                sigma_j[j] += self.var[i][j] / (j + 1)
                if self.n - j - 2 < 0:
                    sigma_n_j[j] += 0
                else:
                    sigma_n_j[j] += self.var[i][self.n - j - 2] / (self.n - j - 1)
            total_var += np.sqrt(sigma_j[j] + sigma_n_j[j])

        remaining_m = self.m - self.evals
        allocated_m = np.zeros(self.n)

        for j in range(math.ceil(self.n / 2) - 1, self.n):

            self.fraction[j] = (np.sqrt(sigma_j[j] + sigma_n_j[j]) / total_var)
            allocated_m[j] = max(0, 
                                 round(remaining_m * self.fraction[j]))


        permutation = list(range(self.n))
        for j in range(self.n):
            for _ in range(int(allocated_m[j])):
                shuffle(permutation)
                
                S = permutation[:j + 1]
                N_minus_S = permutation[j + 1:]

                u = self.V.run(S) - self.V.run(N_minus_S)
                self.evals += 1

                for i in S:
                    self.sv_ij_sum[i][j] += u
                    self.m_ij[i][j] += 1
                
                for i in N_minus_S:
                    self.sv_ij_sum[i][len(N_minus_S) - 1] -= u
                    self.m_ij[i][len(N_minus_S) - 1] += 1


        for i in range(self.n):
            for j in range(self.n):
                if self.m_ij[i][j] == 0:
                    self.sv[i] += 0
                else:
                    self.sv[i] += (self.sv_ij_sum[i][j] / self.m_ij[i][j])
            self.sv[i] /= self.n

        self._log_end()

        return self.sv
    
    @timer_decorator
    def more(self, m) -> dict:
        
        permutation = list(range(self.n))
        for j in range(self.n):
    
            for _ in range(int(m * self.fraction[j])):
                shuffle(permutation)

                S = permutation[:j + 1]
                N_minus_S = permutation[j + 1:]

                u = self.V.run(S) - self.V.run(N_minus_S)
                self.evals += 1

                for i in S:
                    self.sv_ij_sum[i][j] += u
                    self.m_ij[i][j] += 1

                for i in N_minus_S:
                    self.sv_ij_sum[i][len(N_minus_S) - 1] -= u
                    self.m_ij[i][len(N_minus_S) - 1] += 1

        for i in range(self.n):
            for j in range(self.n):
                if self.m_ij[i][j] == 0:
                    self.sv[i] += 0
                else:
                    self.sv[i] += (self.sv_ij_sum[i][j] / self.m_ij[i][j])
            self.sv[i] /= self.n

        return self.sv

 
class ComplementaryBernstein(MonteCarlo):

    def __init__(self, n, m, V, logging=False) -> None:
        super().__init__(n, m, V, logging)  

        # Initialize the game with the number of players
        self.V = self.V(n)

        # Shapley value accumulators 
        self.sv = {i: 0 for i in range(self.n)}
        self.sv_ij = defaultdict(lambda: defaultdict(list))
        self.m_ij = defaultdict(lambda: defaultdict(int))
        self.var = defaultdict(lambda: defaultdict(float))
        self.sv_ij_float = defaultdict(lambda: defaultdict(float))

        self.thelta = 5
        self.r = 10
        self.flag = False
        self.m_init = 30
        self.coef = [binom(n - 1, _) for _ in range(n + 1)]

        self._log_start()
           

    def reset(self):
        self.sv = {i: 0 for i in range(self.n)}
        self.sv_ij = defaultdict(lambda: defaultdict(list))
        self.m_ij = defaultdict(lambda: defaultdict(int))
        self.var = defaultdict(lambda: defaultdict(float))
        self.sv_ij_float = defaultdict(lambda: defaultdict(float))

    def  _calculate_var(self, i, j) -> None:

        if self.flag:
            # This block is executed when flag is True
            self.var[i][j] = np.var(self.sv_ij[i][j], ddof=1)
            self.var[i][j] = (
                math.sqrt(2 * self.var[i][j] * math.log(2 / self.thelta) / len(self.sv_ij[i][j])) +
                7 * math.log(2 / self.thelta) / (3 * (len(self.sv_ij[i][j]) - 1))
            )
        else:
            # This block is executed when flag is False
            self.var[i][j] = np.var(self.sv_ij[i][j])
            self.var[i][j] = (
                math.sqrt(2 * self.var[i][j] * self.thelta / len(self.sv_ij[i][j])) +
                3 * self.r * self.thelta / len(self.sv_ij[i][j])
            )

    def _select_coalition(self):

        max_score = 0
        permutation = np.arange(self.n)

        for breakp in range(math.ceil(self.n / 2), self.n):

            np.random.shuffle(permutation)
                
            l, fl = [], []

            for pos in range(breakp):

                temp = self.var[permutation[pos]][breakp] - self.var[permutation[pos]][self.n - breakp]
                l.append(temp)
                fl.append(-temp)

            r, fr = [], []

            for pos in range(breakp, self.n):

                temp = self.var[permutation[pos]][self.n - breakp] - self.var[permutation[pos]][breakp]
                r.append(temp)
                fr.append(-temp)
            
            sli = np.argsort(fl)
            slr = np.argsort(fr)

            score = 0
            p = 0

            while p < breakp and p < self.n - breakp and score + l[sli[p]] + r[slr[p]] < score:
                score += l[sli[p]] + r[slr[p]]
                permutation[sli[p]], permutation[breakp + slr[p]] = permutation[breakp + slr[p]], permutation[sli[p]]
                p += 1

            score = 0
            for pos in range(breakp):
                score += self.var[permutation[pos]][breakp]

            for pos in range(breakp, self.n):
                score += self.var[permutation[pos]][self.n - breakp]
        
            if score > max_score:
                max_score = score
                max_perm = permutation
                max_breakp = breakp

        return max_score, (max_perm, max_breakp)


    @timer_decorator
    def run(self) -> dict:

        # While self.sv_ij[i][j] < self.m_init keep on adding samples
        while True:

            # break statement
            # if all code below is excecuted without adding any more samples
            # than break the while loop
            break_evals = self.evals

            for i in range(self.n):
                permutation = [_ for _ in range(self.n) if _ != i] # create a list of all players except i

                for j in range(self.n):

                    if self.m_ij[i][j + 1] >= self.m_init or self.m_ij[i][j + 1] >= self.coef[j]:
                        continue

                    np.random.shuffle(permutation)
                    S = permutation[:j] + [i] 
                    N_minus_S = permutation[j:]

                    u = self.V.run(S) - self.V.run(N_minus_S)
                    self.evals += 1

                    for i_ in S:
                        self.sv_ij[i_][j + 1].append(u) # Hence, the second element of the nested dict represents
                        self.m_ij[i_][j + 1] += 1       # the number of samples in u1 or u2

                    for i_ in N_minus_S:
                        self.sv_ij[i_][self.n - j - 1].append(-u)  # The +1 and -1 are used to adjust for the fact that
                        self.m_ij[i_][self.n - j - 1] += 1        # we iteratively add the player i to u1

            # Break the while true loop if there where no adjustments
            # in the last iteration. This means that for all i and j 
            # the number of samples is equal to m_init.
            if self.evals == break_evals:
                break

        for i in range(self.n):
            for j in range(1, self.n + 1):
                self._calculate_var(i, j)

        for _ in range(self.m - self.evals):

            score_, (perm, j) = self._select_coalition() # This function returns a permutation with a breakpoint

            S = perm[:j]
            N_minus_S = perm[j:]

            u = self.V.run(S) - self.V.run(N_minus_S)
            self.evals += 1

            for i_ in S:
                self.sv_ij[i_][j].append(u)
                self.m_ij[i_][j] += 1
                self._calculate_var(i_, j)

            for i_ in N_minus_S:
                self.sv_ij[i_][self.n - j].append(-u)
                self.m_ij[i_][self.n - j] += 1
                self._calculate_var(i_, self.n - j)

        for i in range(self.n):
            for j in range(1, self.n + 1):
                self.sv_ij_float[i][j] = np.sum(self.sv_ij[i][j])           

        for i in range(self.n):
            self.sv[i] = np.sum(self.sv_ij_float[i][j] / self.m_ij[i][j] if self.m_ij[i][j] > 0 else 0 for j in range(1, self.n+1)) / self.n

        self._log_end()

        return self.sv
    
    @timer_decorator
    def more(self, m):
        print('Not implemented')
        pass