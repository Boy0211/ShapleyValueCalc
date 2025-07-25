from mycode.oldmethods import Simple, Antithetic, Orthogonal, StratifiedMaleki, Sobol, Structured
from mycode.zhang import Complementary, ComplementaryNeyman, ComplementaryBernstein
from mycode.castro import Stratified, StratifiedNeyman
from mycode.newmethods import StratifiedExact, StratifiedNeymanExact, StratifiedBernsteinExact, \
ComplementaryExact, ComplementaryNeymanExact, ComplementaryBernsteinExact


class Approx():

    ### Old Methods ###
    def simple(self, n, m, V, logging=False):
        return Simple(n, m, V, logging)
    
    def antithetic(self, n, m, V, logging=False):
        return Antithetic(n, m, V, logging)
    
    def orthogonal(self, n, m, V, logging=False):
        return Orthogonal(n, m, V, logging)
    
    def sobol(self, n, m, V, logging=False):
        return Sobol(n, m, V, logging)
    
    def stratified_maleki(self, n, m, V, logging=False):
        return StratifiedMaleki(n, m, V, logging)
    
    def structured(self, n, m, V, logging=False):
        return Structured(n, m, V, logging)
    
    ### Zhang Methods ###
    def complementary(self, n, m, V, logging=False):
        return Complementary(n, m, V, logging)

    def complementary_neyman(self, n, m, V, logging=False):
        return ComplementaryNeyman(n, m, V, logging)
    
    def complementary_bernstein(self, n, m, V, logging=False):
        return ComplementaryBernstein(n, m, V, logging)
    
    ### Castro Methods ###
    def stratified(self, n, m, V, logging=False):
        return Stratified(n, m, V, logging)
    
    def stratified_neyman(self, n, m, V, logging=False):
        return StratifiedNeyman(n, m, V, logging)
    
    ### (New) Exact methods combining complementary ###
    def complementary_exact(self, n, m, V, logging=False):
        return ComplementaryExact(n, m, V, logging)
    
    def complementary_neyman_exact(self, n, m, V, logging=False):
        return ComplementaryNeymanExact(n, m, V, logging)
    
    def complementary_bernstein_exact(self, n, m, V, logging=False):
        return ComplementaryBernsteinExact(n, m, V, logging)
    
    ### (New) Exact methods combining stratified ###
    def stratified_exact(self, n, m, V, logging=False):
        return StratifiedExact(n, m, V, logging)
    
    def stratified_neyman_exact(self, n, m, V, logging=False):
        return StratifiedNeymanExact(n, m, V, logging)
    
    def stratified_bernstein_exact(self, n, m, V, logging=False):
        return StratifiedBernsteinExact(n, m, V, logging)
    

    
