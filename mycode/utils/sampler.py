from math import factorial
import random

class SampleWithoutReplacement():

    def __init__(self) -> None:
        pass

    def get_index_from_permutation(combination: tuple, n: int) -> int:
        """
        Return the index of combination (length == k)
        The combination argument should be a sorted sequence (i | i∈{0…n-1})
        """

        k = len(combination)
        index= 0
        item_in_check= 0
        n-= 1 # to simplify subsequent calculations
        for offset, item in enumerate(combination, 1):
            while item_in_check < item:
                index+= factorial(n-item_in_check)//factorial(k-offset)//factorial(n+offset-item_in_check-k)
                item_in_check+= 1
            item_in_check+= 1
        return index

    def get_permutation_from_index(n: int, k: int, index: int) -> tuple:
        """
        Select the 'index'th combination of k over n
        Result is a tuple (i | i∈{0…n-1}) of length k

        Note that if index ≥ binomial_coefficient(n,k)
        then the result is almost always invalid
        """

        result= []
        for item, n in enumerate(range(n, -1, -1)):
            pivot= factorial(n-1)//factorial(k-1)//factorial(n-k)
            if index < pivot:
                result.append(item)
                k-= 1
                if k <= 0: break
            else:
                index-= pivot
        return tuple(result)

    def calc_variance(var_0, mean_0, addit, new_n) -> float:
        return ((new_n-2)/(new_n-1))*var_0+(1/new_n)*((addit-mean_0)**2)