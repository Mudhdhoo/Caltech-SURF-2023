import numpy as np
from Bhatt_Calculator import Bhatt_Calculator

class Reconstructor(Bhatt_Calculator):

    def __init__(self, algorithm, threshold, Bhatt_MC, max_sparsity, batch_size, sigma, method, verbose) -> None:
        super().__init__(threshold, Bhatt_MC, max_sparsity, batch_size, sigma, method, verbose)

    def cheap_reconstruction(self):
        pass


