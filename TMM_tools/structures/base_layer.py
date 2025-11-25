from abc import ABC, abstractmethod
class Basic_Layer(ABC):
    def __init__(self,thickness):
        self._thickness=thickness
    @abstractmethod
    def get_transfer_matrix(self, f, theta, mode="TE"):
        ...
    @abstractmethod
    def thickness(self):
        ...
if __name__ == '__main__':
    ...