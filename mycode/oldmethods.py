import numpy as np
from random import shuffle
from scipy.stats import qmc
import math
from tqdm import trange

from mycode.utils.montecarlo import MonteCarlo
from mycode.utils.timer import timer_decorator


class Simple(MonteCarlo):

    def __init__(self, n, m, V, logging=False) -> None:
        super().__init__(n, m, V, logging)
        
        # Initialize the game with the number of players
        self.V = self.V(n)

        # Calculate the worth of an empty coalition
        self.v_empty = self.V.run([])

        # Initialize the Shapley value and counter
        self.sh = {i:0 for i in range(self.n)}
        self.m_p = {i:0 for i in range(self.n)}
        self.sh_player = {i:0 for i in range(self.n)}

        self._log_start()

    def reset(self):
        self.m_p = {i:0 for i in range(self.n)}
        self.sh_player = {i:0 for i in range(self.n)}
    
    @timer_decorator
    def run(self) -> dict:


        permutation = list(range(self.n))
        for _ in trange(int(self.m / self.n)):
            shuffle(permutation)
            v_previous = self.v_empty

            for player_index in range(len(permutation)):
                player = permutation[player_index]
                new = self.V.run(permutation[:player_index + 1])
                self.sh_player[player] += new - v_previous
                self.m_p[player] += 1
                v_previous = new
                self.evals += 1

        for key, value in self.sh_player.items():
            self.sh[key] = value / self.m_p[key]

        self._log_end()

        return self.sh
    
    @timer_decorator
    def more(self, m):

        self._check_multiple_of_n2(m)

        permutation = list(range(self.n))
        for _ in range(int(m / self.n)):
            shuffle(permutation)
            v_previous = self.v_empty

            for player_index in range(len(permutation)):
                player = permutation[player_index]
                self.sh_player[player] += self.V.run(permutation[:player_index + 1]) - v_previous
                self.m_p[player] += 1
                v_previous = self.V.run(permutation[:player_index + 1])
                self.evals += 1

        for key, value in self.sh_player.items():
            self.sh[key] = value / self.m_p[key]

        return self.sh
        
class Antithetic(MonteCarlo):

    def __init__(self, n, m, V, logging=False) -> None:
        super().__init__(n, m, V, logging)  

        # Initialize the game with the number of players
        self.V = self.V(n)

        # Calculate the worth of an empty coalition
        self.v_empty = self.V.run([])      
        
        # Initialize the Shapley value and counter
        self.sh = {i:0 for i in range(self.n)}
        self.m_p = {i:0 for i in range(self.n)}
        self.sh_player = {i:0 for i in range(self.n)}

        self._log_start()

    def reset(self):
        self.sh = {i:0 for i in range(self.n)}
        self.m_p = {i:0 for i in range(self.n)}

    @timer_decorator
    def run(self) -> dict:

        flipped = False
        permutation = list(range(self.n))
        for _ in range(int(self.m / self.n)):

            if flipped:
                shuffle(permutation)
                flipped = False
            else:
                permutation.reverse()
                flipped = True
            
            v_previous = self.v_empty

            for player_index in range(len(permutation)):
                player = permutation[player_index]
                self.sh_player[player] += self.V.run(permutation[:player_index + 1]) - v_previous
                self.m_p[player] += 1
                v_previous = self.V.run(permutation[:player_index + 1])
                self.evals += 1

        for key, value in self.sh_player.items():
            self.sh[key] = value / self.m_p[key]


        self._log_end()

        return self.sh
    
    @timer_decorator
    def more(self, m):

        self._check_multiple_of_n2(m)

        flipped = False
        permutation = list(range(self.n))
        for _ in range(int(m / self.n)):

            if flipped:
                shuffle(permutation)
                flipped = False
            else:
                permutation.reverse()
                flipped = True
            
            v_previous = self.v_empty

            for player_index in range(len(permutation)):
                player = permutation[player_index]
                self.sh_player[player] += self.V.run(permutation[:player_index + 1]) - v_previous
                self.m_p[player] += 1
                v_previous = self.V.run(permutation[:player_index + 1])
                self.evals += 1

        for key, value in self.sh_player.items():
            self.sh[key] = value / self.m_p[key]

        return self.sh
    
class StratifiedMaleki(MonteCarlo):

    def __init__(self, n, m, V, logging=False) -> None:
        super().__init__(n, m, V, logging)  

        # Initialize the game with the number of players
        self.V = self.V(n)

        # Calculate the worth of an empty coalition
        self.v_empty = self.V.run([])      
        
        # Initialize the Shapley value and counter
        self.sh = {i:0 for i in range(self.n)}
        self.sh_ik = {i:{k:0 for k in range(self.n)} for i in range(self.n)}
        self.m_k = {k:0 for k in range(self.n)}

        self._log_start()

    def reset(self):
        pass

    def _calculate_m_k_star(self, m, k, n):
        # Calculate numerator: m * (k+1)^(2/3)
        numerator = m * (k + 1) ** (2/3)
        
        # Calculate denominator: sum from j=0 to n-1 of (j+1)^(2/3)
        denominator = sum((j + 1) ** (2/3) for j in range(n))
        
        # Return the final result
        return numerator / denominator

    @timer_decorator
    def run(self) -> dict:
        
        summed = 0
        for k in range(self.n):
            self.m_k[k] = int(self._calculate_m_k_star(self.m/self.n, k, self.n))
            summed += (self.m_k[k])

        xyz = 0
        while summed < self.m/self.n:
            self.m_k[xyz] += 1
            summed += 1
            xyz = (xyz + 1) % self.n

        for i in range(self.n):
            
            permutation = list(range(self.n))
            permutation.remove(i)
            
            for k in range(self.n):
                for _ in range(int(self.m_k[k])):

                    shuffle(permutation)

                    self.sh_ik[i][k] += (self.V.run(permutation[:k] + [i]) - self.V.run(permutation[:k])) * (1.0/self.m_k[k])
                    self.evals += 1

            self.sh[i] = np.mean(list(self.sh_ik[i].values()))

        self._log_end()

        return self.sh
    
    @timer_decorator
    def more(self, m):
        
        m_k = {k:0 for k in range(self.n)}
        summed = 0
        for k in range(self.n):
            m_k[k] = int(self._calculate_m_k_star(m/self.n, k, self.n))
            summed += (m_k[k])

        xyz = 0
        while summed < m/self.n:
            m_k[xyz] += 1
            summed += 1
            xyz = (xyz + 1) % self.n
        
        for i in range(self.n):
            
            permutation = list(range(self.n))
            permutation.remove(i)

            for k in range(self.n):

                self.sh_ik[i][k] *= self.m_k[k]
                self.m_k[k] += m_k[k]

                for _ in range(int(m_k[k])):

                    shuffle(permutation)
                    self.sh_ik[i][k] += (self.V.run(permutation[:k] + [i]) - self.V.run(permutation[:k]))
                    self.evals += 1

                self.sh_ik[i][k] /= self.m_k[k]

            self.sh[i] = np.mean(list(self.sh_ik[i].values()))
        
        return self.sh
    
class Structured(MonteCarlo):

    def __init__(self, n, m, V, logging=False) -> None:
        super().__init__(n, m, V, logging)

        # Initialize the game with the number of players
        self.V = self.V(n)

        # Calculate the worth of empty coalition
        self.v_empty = self.V.run([])

        # Initialize the shapley value dicts
        self.sh = {i:0 for i in range(self.n)}
        self.m_p = {i:0 for i in range(self.n)}
        self.sh_i = {i:0 for i in range(self.n)}

        self._log_start()

    def reset(self):
        pass

    def run(self) -> dict:

        # Randomize the order of positions that will be evaluated
        randomized_order = list(range(self.n))
        shuffle(randomized_order)
        
        # Divide the number of evaluations into groups equal to the number of players
        for _ in range(int(self.m / self.n)):

            permutation = list(range(self.n))
            shuffle(permutation)
            
            switch_pos = randomized_order[_ % self.n]

            for i in range(self.n):

                # Find the index of the player
                index = permutation.index(i)

                # Switch positions
                permutation[switch_pos], permutation[index] = permutation[index], permutation[switch_pos]

                # Calculate the Shapley value
                self.sh_i[i] += self.V.run(permutation[:switch_pos + 1]) - self.V.run(permutation[:switch_pos])
                self.m_p[i] += 1
                self.evals += 1

                # Switch back the positions
                permutation[switch_pos], permutation[index] = permutation[index], permutation[switch_pos]

        for key, value in self.sh_i.items():
            self.sh[key] = value / self.m_p[key]  

        self._log_end()
        return self.sh

    def more() -> dict:
        pass


class Orthogonal(MonteCarlo):

    def __init__(self, n, m, V, logging=False) -> None:
        super().__init__(n, m, V, logging)

        # Initialize the game with the number of players
        self.V = self.V(n)

        # Calculate the worth of empty coalition
        self.v_empty = self.V.run([])

        # Initialize the shapley value dicts
        self.sh = {i:0 for i in range(self.n)}
        self.m_p = {i:0 for i in range(self.n)}
        self.sh_i = {i:0 for i in range(self.n)}

        self._log_start()

    def reset(self):
        pass

    def _argsort_to_permutation(self, x):
        """
        Return a permutation that sorts the indices of x in ascending order.
        """
        return np.argsort(x)

    def _generate_projection_matrix(self, d):
        """
        Generate the projection matrix U for d dimensions. The result is a (d-1) x d matrix.
        """

        U = np.zeros((d - 1, d))
        for i in range(d - 1):
            for j in range(d):
                if i < j:
                    U[i, j] = 1
            U[i] -= (i + 1) / d
        # Normalise the rows from U
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        return U

    def _orthogonal_samples(self, k, d):
        """
        Implements Algorithm 3: Generate k = 2(d-1) well-distributed permutations from S_{d-2}.
        Parameters:
            k (int): The number of permutations to generate (must be 2(d-1)).
            d (int): Dimension of the space (length of permutations).
        Returns:
            np.array: A matrix (k x d) of permutations.
        """
        assert k == 2 * (d - 1), "The number of permutations k must be equal to 2(d-1)."

        # Step 1: Generate a matrix of normally distributed random numbers
        X = np.random.normal(0, 1, (k, d - 1))  # Each row now has dimension d-1
        
        # Step 2: Instantiate an empty matrix Y to store the permutations
        Y = np.zeros((k, d), dtype=np.int32)
        
        # Step 3: Gram-Schmidt process
        for i in range(k):
            for j in range(i):
                # Project X[i] orthogonally onto X[j]
                X[i] -= np.dot(X[i], X[j]) * X[j]
            
            # Normalize X[i] to a unit vector
            X[i] /= np.linalg.norm(X[i])
        
            # Step 4: Define the projection matrix U
            U = self._generate_projection_matrix(d)
        
            # Generate two permutations (original and negative)
            Y[i] = self._argsort_to_permutation(U.T @ X[i])
            if i + 1 < k:
                Y[i + 1] = self._argsort_to_permutation(U.T @ -X[i])
        
        return Y

    def run(self) -> dict:
        
        k = 2 * (self.n - 1)

        for _ in range(int(self.m / self.n / k)):
            for permutation in self._orthogonal_samples(k, self.n):
                v_previous = self.v_empty
                
                for player_index in range(len(permutation)):
                    player = permutation[player_index]
                    self.sh_i[player] += self.V.run(permutation[:player_index + 1]) - v_previous
                    self.m_p[player] += 1
                    v_previous = self.V.run(permutation[:player_index + 1])
                    self.evals += 1

        for key, value in self.sh_i.items():
            self.sh[key] = value / self.m_p[key]

        self._log_end()

        return self.sh
    
class Sobol(MonteCarlo):

    def __init__(self, n, m, V, logging=False) -> None:
        super().__init__(n, m, V, logging)

        # Initialize the game with the number of players
        self.V = self.V(n)

        # Calculate the worth of empty coalition
        self.v_empty = self.V.run([])

        # Initialize the shapley value dicts
        self.sh = {i:0 for i in range(self.n)}
        self.m_p = {i:0 for i in range(self.n)}
        self.sh_i = {i:0 for i in range(self.n)}

        # Initialize the Sobol sequence
        self.sampler = qmc.Sobol(d = self.n, scramble=True)

        self._log_start()

    def reset(self):
        pass

    def _sobol_permutation(self, m):
        """
        Generate a permutation of the first n integers using the Sobol sequence.
        """
        if np.log2(m) % 1 != 0:
            y = math.ceil(np.log2(m))
            sobol_point = self.sampler.random_base2(y)
            random_indices = np.random.choice(sobol_point.shape[0], m, replace=False)
            sobol_point = sobol_point[random_indices]

        else:
            y = math.ceil(np.log2(m))
            sobol_point = self.sampler.random_base2(y)

        permutations = [np.argsort(point) for point in sobol_point]

        return permutations
    
    def run(self) -> dict:

        for permutation in self._sobol_permutation(int(self.m / self.n)):
            v_previous = self.v_empty
            
            for player_index in range(len(permutation)):
                player = permutation[player_index]
                self.sh_i[player] += self.V.run(permutation[:player_index + 1]) - v_previous
                self.m_p[player] += 1
                v_previous = self.V.run(permutation[:player_index + 1])
                self.evals += 1

        for key, value in self.sh_i.items():
            self.sh[key] = value / self.m_p[key]

        self._log_end()

        return self.sh




    
    
