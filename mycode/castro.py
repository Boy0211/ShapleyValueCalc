import numpy as np
import random
from collections import defaultdict
import math

from mycode.utils.montecarlo import MonteCarlo
from mycode.utils.timer import timer_decorator

class Stratified(MonteCarlo):

    def __init__(self, n, m, V, logging=False) -> None:
        super().__init__(n, m, V, logging)  

        # Initialize the game with the number of players
        self.V = self.V(n)

        # Calculate the worth of an empty coalition
        self.v_empty = self.V.run([])      
        
        # Initialize the Shapley value and counter
        self.sh = {i:0 for i in range(self.n)}
        
        # BE AWARE: This is a dictionary of dictionaries instead of a plain dictionary
        self.m_p_pos = {i:{l:int(self.m / self.n**2) for l in range(self.n)} for i in range(self.n)}

        # BE AWARE: This is a dictionary of dictionaries
        self.sh_pos = defaultdict(lambda: defaultdict(float))

        self._log_start()
              
    def reset(self):
        pass
    
    @timer_decorator
    def run(self):
        
        for i in range(self.n):

            permutation = list(range(self.n))
            permutation.remove(i)

            for l in range(self.n):
                for _ in range(self.m_p_pos[i][l]):
                    
                    random.shuffle(permutation)
                    self.sh_pos[i][l] += (self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l])) * (1.0/self.m_p_pos[i][l])
                    self.evals += 1

            self.sh[i] = np.mean(list(self.sh_pos[i].values())) 

        self._log_end()

        return self.sh
    
    @timer_decorator
    def more(self, m) -> dict:

        m = int(m / (self.n**2))

        for i in range(self.n):

            permutation = list(range(self.n))
            permutation.remove(i)

            for l in range(self.n):

                self.sh_pos[i][l] *= self.m_p_pos[i][l]
                self.m_p_pos[i][l] += m

                for _ in range(m):

                    random.shuffle(permutation)
                    self.sh_pos[i][l] += (self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l]))
                    self.evals += 1
                
                self.sh_pos[i][l] /= self.m_p_pos[i][l]

            self.sh[i] = np.mean(list(self.sh_pos[i].values()))

        return self.sh
    

class StratifiedNeyman(MonteCarlo):
    
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
        self.fraction = {i:{l:0 for l in range(self.n)} for i in range(self.n)}

        # Initialize an alpha for distribution between the first and second stage
        self.alpha = 0.5

        self._log_start()

    def reset(self):
        pass
    
    @timer_decorator
    def run(self):

        # Set the total amount of variance
        summed_variance = 0

        # Compute the amount of samples for the first phase per player and position
        m_phase1 = int((self.m / self.n**2) * 0.5)

        for i in range(self.n):
            
            permutation = list(range(self.n))
            permutation.remove(i)

            for l in range(self.n):

                temp_phase1_list = []
                for _ in range(m_phase1):
                    
                    random.shuffle(permutation)
                    temp_phase1_list.append(self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l]))
                    self.evals += 1
                
                # Calculate the Shapley value and the variance
                self.sh_il[i][l] = np.sum(temp_phase1_list)
                self.var_il[i][l] = np.var(temp_phase1_list, ddof=1)
                del temp_phase1_list

                # Add the variance to the total variance
                summed_variance += self.var_il[i][l]

        remaining_samples = self.m - self.evals

        # Calculate the total amount of variance and set the overallocated_samples on 0
        for i in range(self.n):
            for l in range(self.n):
                self.fraction[i][l] = self.var_il[i][l] / summed_variance
                self.m_il[i][l] = round(self.fraction[i][l] * remaining_samples)

        if self.logging:
            self.logger.info(f"Stratified Neymand, first phase finished, starting second phase.")

        for i in range(self.n):

            permutation = list(range(self.n))
            permutation.remove(i)

            for l in range(self.n):

                for _ in range(self.m_il[i][l]):
                    random.shuffle(permutation)
                    self.sh_il[i][l] += (self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l]))
                    self.evals += 1

                self.m_il[i][l] += m_phase1
                self.sh_il[i][l] = self.sh_il[i][l] / self.m_il[i][l]

            self.sh_i[i] = np.mean(list(self.sh_il[i].values()))
    
        self._log_end()

        return self.sh_i
    
    @timer_decorator
    def more(self, m) -> dict:

        new_m_il = {i:{l:round(self.fraction[i][l] * m) for l in range(self.n)} for i in range(self.n)}

        for i in range(self.n):

            permutation = list(range(self.n))
            permutation.remove(i)

            for l in range(self.n):

                self.sh_il[i][l] *= self.m_il[i][l]
                self.m_il[i][l] += new_m_il[i][l]

                for _ in range(new_m_il[i][l]):
                    
                    random.shuffle(permutation)
                    self.sh_il[i][l] += (self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l]))
                    self.evals += 1

                self.sh_il[i][l] /= self.m_il[i][l]

            self.sh_i[i] = np.mean(list(self.sh_il[i].values()))
        
        return self.sh_i
        


# def run(self):
        
#         if self.debug:
#             print("Method: StratifiedNeumann has been launched\n")

#         # Set the total amount of variance
#         summed_variance = 0

#         # Compute the amount of samples for the first phase per player and position
#         m_phase1 = int((self.m / self.n**2) * 0.5)

#         for i in range(self.n):
            
#             permutation = list(range(self.n))
#             permutation.remove(i)

#             for l in range(self.n):

#                 temp_phase1_list = []
#                 for _ in range(m_phase1):
                    
#                     random.shuffle(permutation)
#                     temp_phase1_list.append(self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l]))
#                     self.evals += 1
                
#                 # Calculate the Shapley value and the variance
#                 self.sh_il[i][l] = np.mean(temp_phase1_list)
#                 self.var_il[i][l] = np.var(temp_phase1_list, ddof=1)

#                 # Add the variance to the total variance
#                 summed_variance += self.var_il[i][l]

#         # Calculate the total amount of variance and set the overallocated_samples on 0
#         overallocated_samples = 0
#         remaining_strata = []
        
#         # Compute the amount of samples for the second phase per player and position based on the variance
#         for i in range(self.n):
#             for l in range(self.n):
#                 self.m_il[i][l] = math.ceil(self.var_il[i][l] / summed_variance * self.m) - m_phase1

#                 # sum the amount of overallocated samples
#                 if self.m_il[i][l] < 0:
#                     overallocated_samples -= self.m_il[i][l]
#                     self.m_il[i][l] = int(0)

#                 # save the strata which will receive more samples
#                 elif self.m_il[i][l] > 0:
#                     remaining_strata.append((i, l))

#         # Deduct the overallocated samples
#         full_set = sum([self.m_il[i][l] for i, l in remaining_strata])
#         for i, l in remaining_strata:
#             self.m_il[i][l] -= overallocated_samples * (self.m_il[i][l] / full_set)
#             self.m_il[i][l] = int(round(self.m_il[i][l]))

#         # Add trigger for debugging
#         if self.debug:
#             trigger = True 
        
#         # Add any samples that were left over due to rounding
#         while sum([sum([self.m_il[i][l] for l in range(self.n)]) for i in range(self.n)]) < m_phase1 * self.n**2:

#             if self.debug and trigger:
#                 print(f'{((m_phase1* self.n**2) - sum([sum([self.m_il[i][l] for l in range(self.n)]) for i in range(self.n)])) / self.m} samples are randomly added')
#                 trigger = False

#             i, l = random.choice(remaining_strata)
#             self.m_il[i][l] += 1

#         # Add any samples that were overused due to rounding
#         while sum([sum([self.m_il[i][l] for l in range(self.n)]) for i in range(self.n)]) > m_phase1 * self.n**2:

#             if self.debug and trigger:
#                 print(f'{sum([sum([self.m_il[i][l] for l in range(self.n)]) for i in range(self.n)]) - (m_phase1 * self.n**2)} samples are randomly removed')
#                 trigger = False

#             i, l = random.choice(remaining_strata)
#             self.m_il[i][l] -= 1

#         # Set up the value for the second phase 
#         self.sh_il = {i:{l:self.sh_il[i][l] * m_phase1 for l in range(self.n)} for i in range(self.n)}

#         for i in range(self.n):

#             permutation = list(range(self.n))
#             permutation.remove(i)

#             for l in range(self.n):

#                 for _ in range(self.m_il[i][l]):
#                     random.shuffle(permutation)
#                     self.sh_il[i][l] += (self.V.run(permutation[0:l] + [i]) - self.V.run(permutation[0:l]))
#                     self.evals += 1
                
#                 self.sh_i[i] += self.sh_il[i][l] / (m_phase1 + self.m_il[i][l])
            
#             self.sh_i[i] /= self.n
        
#         if self.debug:
#             print('Number of evals done: ', self.evals / self.m * 100, '%', '\n')

#         return self.sh_i