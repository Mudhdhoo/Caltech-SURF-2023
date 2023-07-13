from random import random
import numpy as np

# Make this class private? Add verbose function?
class Bhatt_Calculator:
    def __init__(self,threshold,Bhatt_MC,max_sparsity,batch_size,sigma,method,verbose) -> None:
        self.threshold = threshold
        self.Bhatt_MC = Bhatt_MC
        self.max_sparsity = max_sparsity
        self.batch_size = batch_size
        self.sigma = sigma
        self.method = method
        self.verbose = verbose

    def Bhatt(self, J, u):
        """
        Calculates the Bhattacharyya coefficient.
        """
        n, q = J.shape
        u_vec = u.reshape(n,1)
        Z0, W0 = self.Z0_calculator(self.Bhatt_MC, q)
        indices = np.random.randint(0,int(n), Z0.shape)     # Pick random indicies for Monte-Carlo sampling
        val = self.bhatt_integrand(J, u_vec, Z0, indices)

    def bhatt_integrand(self,J, u_vec, Z0, indices):
        """
        Computes either h(j,u,x,z) or f(j,u,z) for Monte-Carlo Sampling
        """
        MC, q = Z0.shape
        n = J.shape[0]
        nindex = len(indices)
        P1_Z,P2_Z = self.Pcalculator_sparse2(J, u_vec, Z0, indices)     # Calculate P1 and P2

    def Z0_calculator(self, MC_iterations, q):
        if self.method == 'random':
            Z0 = np.random.randn(MC_iterations,q)
            W0 = np.ones([1,MC_iterations])/MC_iterations

        return Z0, W0

    def Pcalculator_sparse2(self, J, u_vec, Z0, indices):
        """
        Computes the probability distributions P1 and P2.
        """
        # Setup
        n, q = J.shape
        MC = Z0.shape[0]
        nindex = len(indices)
        C = 1/np.sqrt(self.sigma) * (2*np.pi)**(-q/2)
        logC = np.log(C)

        if q == 1:
            threshold_val = self.sigma * np.sqrt(2 * np.log(C/self.threshold))
        else:
            threshold_val = 2 * self.sigma**2 * np.log(C/self.threshold)

        inv2sigma2 = -0.5 * self.sigma**(-2)
        sum0 = sum(1-u_vec)     # Denominator P2
        sum1 = n - sum0         # Denominator P1

        # Compute sparsity pattern
        ivec = np.zeros([self.max_sparsity,1])
        jvec = np.zeros([self.max_sparsity,1])
        count = 0

        # Batch processing
        num_batches = int(np.ceil(nindex / self.batch_size))

        if q > 1:
            J_1nq =J.reshape([1, n, q])
        else:
            J_1nq = J.T
        
        perm = np.random.permutation(nindex)

        for b in range(1,num_batches+1):
            # Get batch indices
            I_start = (b-1) * self.batch_size 
            I_end = min(b * self.batch_size, nindex)
            this_batch_size = I_end - I_start 

            # Compute norm squared for the batch
            if q == 1:
                perm_batch_indices = perm[I_start:I_end]
                randomize_ind = [int(indices[i]) for i in perm_batch_indices]
                J_I_minus_J = np.array([J[i] for i in randomize_ind]) - J_1nq  #this_batch_size x n
                KER_above_threshold = J_I_minus_J < threshold_val and J_I_minus_J > - threshold_val
                
                print(KER_above_threshold)
            # else?                                                                                                                                                                                    