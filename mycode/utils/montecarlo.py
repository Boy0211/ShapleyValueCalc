from mycode.utils.logger import setup_logger

class MonteCarlo():

    def __init__(self, n, m, V, logging=False) -> None:
        """
        Approximation class for the Shapley value. 
        
        Parameters:
        n (list):           List representing the players
        m (int):            Total number of evaluations              
        V (function):       Function to calculate the worth of a coalition based on a game
        logging (bool):     Logger flag
        """

        self.n = n
        # if m is fixed, get the number of m from the the temp_fixedm.txt file
        if m == 'fixed':   
            self.m = int(open('temp_fixedm.txt', 'r').read().strip())
        else:
            self.m = m * (n**2)
        self.V = V
        self.evals = 0
        self.logging = logging

        if self.logging == True:
            self.logger = setup_logger()

    def _log_start(self):
        if self.logging:
            self.logger.info(f"Monte Carlo: n={self.n}, m={self.m/self.n**2}, V={self.V.name}")

    def _log_end(self):
        if self.logging:
            self.logger.info(f"Evaluations = {self.evals}, {round(self.evals/self.m*100)}%")
            if self.evals > self.m:
                self.logger.warning("Warning: Number of evaluations exceeded the limit.")
            if self.evals < self.m:
                self.logger.warning("Warning: Number of evaluations is less than expected.")

    def _check_multiple_of_n2(self, extra_m):
        if extra_m % self.n**2 != 0:
            raise ValueError(f"Number of evaluations must be a multiple of n**2, Got {extra_m / self.n**2}")