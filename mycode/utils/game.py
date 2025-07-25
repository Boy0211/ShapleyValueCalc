import numpy as np
from math import floor
from abc import ABC, abstractmethod
import json

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

import networkx as nx
import pandas as pd

class Games():

    def __init__(self) -> None:
        pass

    def airportgame(self, N):
        return AirportGame(N)

    def votinggame(self, N):
        return VotingGame(N)

    def complexdivisiongame(self, N):
        return ComplexDivisionGame(N)

    def shoesgame(self, N):
        return ShoesGame(N)
    
    def mstgame(self, N):
        return MSTGame(N)
    
    def datavaluationgame(self, N):
        return DataValuationGame(N)
    
    def featureevaluationgame(self, N):
        return FeatureEvaluationGame(N)
    
    def networkedgame(self, N):
        return NetworkedGame(N)

class Game(ABC):

    def __init__(self, n, name, debug=False) -> None:
        self.n = n
        self.name = name
        self.debug = False
        
    def __str__(self) -> str:
        return self.name

class AirportGame(Game):
    """
    This is the Airport Game, where the value of a coalition is 
    the maximum weight of a player in the coalition. This game
    is used with a fixed amount of players and weights.
    """
    
    def __init__(self, n) -> None:
        
        # Get the number of players of the underlying class
        super().__init__(n, 'airportgame')



        # set the weights of the game
        if self.n == 15:
            self.W = [1, 1, 2, 2, 2, 3, 4, 5, 5, 5, 7, 8, 8, 8, 10]
        elif self.n <= 100:
            self.W = self._get_weights_small(self.n)
        elif (self.n > 100) and (self.n <= 200):
            self.W = self._get_weights_large(self.n)

    def _get_weights_small(self, num_players:int) -> np.array:
        """
        num_players (int): Number of players

        returns (np.array): Weights of the players

        Function returns the weights of the players in the airport game
        """
        if num_players < 10:
            raise ValueError('The number of players must be at least 10.')
        elif num_players > 100:
            raise ValueError('The number of players must be at most 100.')

        # Number of times a weight is added to the game
        x = [8, 12, 6, 14, 8, 9, 13, 10, 10, 10]

        # Based on the number of players, remove a certain number of weights
        # in a systematic way
        x.reverse()
        for i in range(100 - num_players):
            if x[i % 10] == 0:
                j = i + 1
            else:
                j = i
            x[j % 10] -= 1
        x.reverse()

        N = np.array([1] * x[0] + [2] * x[1] + [3] * x[2] + [4] * x[3] + 
                    [5] * x[4] + [6] * x[5] + [7] * x[6] + [8] * x[7] + 
                    [9] * x[8]  + [10] * x[9])
        
        return N
    
    def _get_weights_large(self, num_players:int) -> np.array:

        if num_players < 101:
            raise ValueError('The number of players must be at least 101.')
        elif num_players > 200:
            raise ValueError('The number of players must be at most 200.')

        # Number of times a weight is added to the game
        x = [16, 24, 16, 28, 16, 24, 18, 20, 20, 18]

        # Based on the number of players, remove a certain number of weights
        # in a systematic way
        x.reverse()
        for i in range(200 - num_players):
            if x[i % 10] == 0:
                j = i + 1
            else:
                j = i
            x[j % 10] -= 1
        x.reverse()

        N = np.array([1] * x[0] + [2] * x[1] + [3] * x[2] + [4] * x[3] + 
                    [5] * x[4] + [6] * x[5] + [7] * x[6] + [8] * x[7] + 
                    [9] * x[8]  + [10] * x[9])
        
        return N 

    def run(self, S):

        if len(S) < 1:
            return 0 
        elif len(S) == 1:
            return self.W[S[0]]
        else:
            return np.max([self.W[i] for i in S])
    
class VotingGame(Game):
    """
    This is the voting game, where the value of a coalition is 1 if the sum of the weights
    of the players in the coalition is greater than half of the total sum of the weights of the
    players in the game. Otherwise, the value is 0.
    """

    def __init__(self, n) -> None:

        # Get the number of players of the underlying class
        super().__init__(n, 'votinggame')

        # set the weights of the game
        if self.n == 15:
            self.W = [1, 3, 3, 6, 12, 16, 17, 19, 19, 19, 21, 22, 23, 24, 29]
        elif self.n == 51:
            self.W = [3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 6, 
                      6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 
                      12, 12, 12, 13, 13, 14, 17, 17, 21, 25, 26, 26, 27, 41, 
                      45]
            self.W.reverse()
        elif (self.n != 15) and (self.n != 51) and (self.n >= 20) and (self.n <= 100) and (self.n % 10 == 0):
            self.W = self._get_weights_players(num_players=n)
        else:
            raise ValueError("This game is not implemented for this number of players.")

        self.totalW = np.sum(self.W)

    def _get_weights_players(self, num_players:int) -> np.array:

        df = pd.read_csv(f'data/votinggame/{num_players}.csv')
        W = df['Weight'].tolist()

        return W


    def run(self, S):

        if np.sum([self.W[int(i)] for i in S]) > (self.totalW/2):
            return 1
        else:
            return 0
    
class ComplexDivisionGame(Game):
    """
    This is the Complex Division Game, where the value of a coalition is the square of the sum
    of the weights of the players in the coalition, divided by 2. However, the value is the difference
    between the value and the integer value of the value.
    """

    def __init__(self, n) -> None:

        # Get the number of players of the underlying class
        super().__init__(n, 'complexdivisiongame')

        # set the weights of the game
        self.W = [45, 41, 27, 26, 25, 21, 13, 13, 12, 12, 11, 11, 10, 10, 10]

        # check if the length of the weights is equal to the amount of players
        if self.n != len(self.W):
            raise ValueError("N must be equal to the length of self.w.")

    def run(self, S):

        first = np.sum([self.W[i]/50 for i in S])**2
        second = int(first)

        # Good code should be #TODO
        # summation = np.sum([self.W[i]/50 for i in S])
        # first = summation**2
        # second = int(summation)**2

        return first - second
    
class ShoesGame(Game):
    """
    This is the Shoes Game, where the value of a coalition is the minimum number of 
    lefties and righties in the coalition. The weights of the players are based on 
    the position of the player in the coalition.
    """

    def __init__(self, n) -> None:

        # Get the number of players of the underlying class
        super().__init__(n, 'shoesgame')
        
        # check if the value of 'n' is even
        if self.n % 2 != 0:
            raise ValueError("The value of 'n' must be even.")
        
        # make a list of weights
        self.W_left = [0] * int(0.5 * self.n)
        self.W_right = [1] * (self.n - len(self.W_left))
        self.W = self.W_left + self.W_right

    def run(self, S):
      
        shoes = [self.W[i] for i in S]
        l = shoes.count(0)
        r = shoes.count(1)
        return np.min([l, r])

class MSTGame(Game):
    """
    This is the Minimum Spanning Tree (MST) Game, where the value of a coalition 
    is the total cost of the minimum spanning tree that spans all the players 
    in the coalition plus the special source node `0`.
    
    In this game, nodes are arranged in a cyclic manner where adjacent nodes 
    have a cost of 1, and nodes can connect to the special node `0` with a 
    cost of n + 1.
    """

    def __init__(self, n) -> None:
        # Initialize based on the underlying class
        super().__init__(n, 'mstgame')

        self.adj_cost = 1          # Cost to connect adjacent nodes
        self.source_cost = self.n + 1 # Cost to connect any player to the source node `0`

    def run(self, S):
        """
        Evaluate the coalition S based on the minimum spanning tree cost involving node 0.
        
        Parameters:
            - S (list or set): A list or set of player indices forming a coalition S (excluding `0`).

        Returns:
            - value (int or float): The minimum cost of the spanning tree of coalition S union with node `0`.
        """

        if len(S) == 0:
            return 0
        
        if len(S) == 1:
            return self.source_cost
        
        if len(S) == self.n:
            return self.n * 2
        
        S = sorted(S)  # Sort players in the coalition

        cost = self.source_cost
        for index in range(1, len(S)):

            # if next to each other add the adjacent cost
            if (S[index] - S[index - 1] == 1):
                cost += self.adj_cost

            # if not adjacent, add the source cost
            else:
                cost += self.source_cost

        if ((S[index] == self.n - 1) and (S[0] == 0)):
            cost += self.adj_cost
            cost -= self.source_cost

        return cost

class DataValuationGame(Game):
    """
    This is a data valuation game, where the value of a coaltion is the accuracy
    of a Support Vector Classifier (SVC) trained on the coalition. The dataset
    used in this game is the Breast Cancer dataset from sklearn.
    """

    def __init__(self, n, t_size=50) -> None:

        # Get the number of players of the underlying class
        super().__init__(n, 'datavaluationgame')

        # import the training dataset
        data = load_breast_cancer()

        # pick the same 100 data points for training
        with open('data/datavaluationgame/train.txt', 'r') as f:
            fl = f.read().splitlines()
            train_indices = [int(''.join(filter(str.isalnum, x))) for x in fl[0].split(',')]

        self.X_train = data.data[train_indices]
        self.y_train = data.target[train_indices]

        # pick t_size random datapoints for testing, ensuring they are not in the training set
        with open('data/datavaluationgame/test.txt', 'r') as f:
            fl = f.read().splitlines()
            test_indices = [int(''.join(filter(str.isalnum, x))) for x in fl[0].split(',')]

        self.X_test = data.data[test_indices]
        self.y_test = data.target[test_indices]

        # Create the classifier
        self.clf = SVC()

    def run(self, S):

        if len(S) == 0:
            return 0

        # Get the training data based on the coalition
        X_train = self.X_train[S, :]
        y_train = self.y_train[S]

        if len(np.unique(y_train)) < 2:
            return 0

        # Fit the classifier
        self.clf.fit(X_train, y_train)

        # Predict the values
        y_pred = self.clf.predict(self.X_test)

        # Return the accuracy
        return accuracy_score(self.y_test, y_pred)
    
class FeatureEvaluationGame(Game):
    """
    This is a feature evaluation game, where the value of a coalition is the accuracy
    of a Support Vector Classifier (SVC) trained on the coalition. The dataset
    used in this game is the Breast Cancer dataset from sklearn.
    """

    def __init__(self, n) -> None:

        # Get the number of players of the underlying class
        super().__init__(n, 'featureevaluationgame')

        # import the training dataset
        data = load_breast_cancer()

        # Create a trainingset of 569 samples and a test size of 0.1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.data, data.target, 
                                                                                test_size=0.1, random_state=42)
        
        # Create the classifier
        self.clf = SVC()

    def run(self, S):

        if len(S) == 0:
            return 0

        # Select a subset of features
        X_train = self.X_train[:, S]

        # Fit the classifier
        self.clf.fit(X_train, self.y_train)

        # Predict the values
        y_pred = self.clf.predict(self.X_test[:, S])

        # Return the accuracy
        return accuracy_score(self.y_test, y_pred)
    
class NetworkedGame(Game):
    """
    This is a networked game, currently the game can be anything. Described in the 
    self.game() function. The calculation method is spllited into Myerson and Shapley,
    where Myerson takes the interconnected componetns of a subgraph into account and 
    Shapley does not. 
    """

    def __init__(self, n: int) -> None:
        with open('temp_graph.txt', 'r') as f:
            d = json.load(f)

        self.typecalc = d['typecalc']
        self.gamename = d['gamename']
        super().__init__(n, 'networkedgame_' + d['graph'] + '_' + d['typecalc'] + '_' + str(d['gamename']))

        if d['graph'] == 'Krebbs':

            # Load in the Krebbs network
            df = pd.read_csv('data/networks/krebbs/krebbs/edges.csv')
            self.G = nx.from_pandas_edgelist(df, 'source', 'target')

        elif d['graph'] == 'Zerkani':

            # Load in the Zerkani network
            df = pd.read_csv('data/networks/zerkani/zerkani.csv')
            self.G = nx.from_pandas_edgelist(df, 'Source', 'Target', edge_attr=True)

        elif d['graph'] == 'Random':
            fh = open('data/networks/random/random.edgelist', 'rb')
            self.G = nx.read_edgelist(fh, nodetype=int)

        elif d['graph'] == 'Small-world':
            fh = open('data/networks/smallworld/smallworld.edgelist', 'rb')
            self.G = nx.read_edgelist(fh, nodetype=int)

        elif d['graph'] == 'Scale-free':
            fh = open('data/networks/scalefree/scalefree.edgelist', 'rb')
            self.G = nx.read_edgelist(fh, nodetype=int)

    def run(self, S):

        # If no graph return 0
        if len(S) == 0:
            return 0

        # Shapley or Myerson
        if self.typecalc == 'myerson':

            # return the sum of the connected components
            return sum([self.game(c) for c in nx.connected_components(self.G.subgraph(S))])
        else:

            # return game if connected otherwise 0
            return self.game(set(S)) if nx.is_connected(self.G.subgraph(S)) else 0
        
    def game(self, S):
        Sg = self.G.subgraph(S)

        num_nodes = len(Sg.nodes)
        # num_edges = len(Sg.edges)

        return num_nodes**2 - num_nodes





        
