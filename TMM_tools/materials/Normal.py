from TMM_tools.materials.base_material import *
from numpy import sqrt

class Normal_Material(Basic_Material):
    def __init__(self,refractiveindex=1, mu=1):
        super().__init__()
        self._refractiveindex=refractiveindex
        self._epsilon= refractiveindex ** 2
        self._mu=mu
    def get_n(self,f):
        return self._refractiveindex
    def get_epsilon(self,f):
        return self._epsilon
    def get_mu(self,f):
        return self._mu
    @property
    def n(self):
        return self._refractiveindex
    @n.setter
    def n(self,refractive):
        self._refractiveindex=refractive
        self._epsilon=refractive**2
    @property
    def epsilon(self):
        return self._epsilon
    @epsilon.setter
    def epsilon(self,epsilon):
        self._epsilon=epsilon
        self._refractiveindex=sqrt(epsilon)
    @property
    def mu(self):
        return self._mu
    @mu.setter
    def mu(self,mu):
        self._mu=mu
    def __mul__(self,other):
        times=int(other)
        layers=[]
        for _ in range(times):
            layers.append(Normal_Material())
        return layers

if __name__ == '__main__':
    A=Normal_Material(refractiveindex=2)
    print(A.n)
    A.n=3
    print(A.n,A.epsilon)