import numpy as np
import scipy.sparse as sp
from numpy.matlib import repmat

# Make this class private? Add verbose function?
class Bhatt_Calculator:
    """
    Class for calculating the Bhattacharyya coefficient. Parent class of Segmenter and Reconstructor.

    ------------ Parameters ------------
    threshold: float
    Threshold above which kernel values are determined to be significant in computing P1 and P2.

    Bhatt_MC: int
        Number of Monte-Carlo iteration for calculating the Bhattacharyya coefficient.

    max_sparsity: int
        Maximum entries in the sparse matrix when computing P1 and P2.

    batch_size: int
        Size of batches for batch processing when computing P1 and P2.

    sigma: float
        Standarddeviation of the Bhattacharyya kernel K(z).

    method: str
        Method for computing the Gaussian integrals. 'random' or 'quadrature'.

    verbose: bool
        Write runtime information to screen if True, otherwise write nothing.
    """
    
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
        val = np.mean(np.sum(self.bhatt_integrand(J, u_vec, Z0, indices) * W0, 1),0)        # Sample h(J,u,x,Z) and take the mean to approximate integral

        if self.verbose:
            print('Bhattacharyya Coefficient: ' + str(val))
        
        return val
 
    def bhatt_integrand(self,J, u_vec, Z0, indices):
        """
        Computes f(j,u,z) for Monte-Carlo Samplings 
        """
        MC, q = Z0.shape
        n = J.shape[0]
        nindex = len(indices)
        P1_Z, P2_Z = self.Pcalculator_sparse2(J, u_vec, Z0, indices)     # Calculate P1 and P2
        B_KER_Z0 = ((2*np.pi)**(q/2) * self.sigma**(1/2) * np.exp(0.5*np.sum(Z0**2,1))).reshape(-1,1) #MC x 1
        B_KER_Z_Jx = repmat(B_KER_Z0, 1, nindex).T      # nindex x MC

        # evaluate integrand pointwise
        fout = B_KER_Z_Jx * np.sqrt(P1_Z * P2_Z)    # max(P1_Z*P2_Z, 0)?

        return fout

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
        sum0 = np.sum(1-u_vec)     # Denominator P2
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
            I_end = min(b * self.batch_size, nindex) - 1
            this_batch_size = I_end - I_start + 1

            # Compute norm squared for the batch
            if q == 1:
                perm_batch_indices = perm[I_start:I_end+1]
                randomize_ind = np.array([int(indices[i]) for i in perm_batch_indices])

                J_I_minus_J = np.array([J[i] for i in randomize_ind]) - J_1nq  #this_batch_size x n
                KER_above_threshold1 = J_I_minus_J < threshold_val
                KER_above_threshold2 = J_I_minus_J > - threshold_val
                KER_above_threshold = np.logical_and(KER_above_threshold1,KER_above_threshold2)
            #else?

            rows, cols = np.nonzero(KER_above_threshold)

            rows = np.expand_dims(rows[0:self.max_sparsity-count],1)
            cols = np.expand_dims(cols[0:self.max_sparsity-count],1)

            # Convert to global indices
            rows = rows + I_start - 0 ########### -1?
            l = len(rows)
            # Incrementally add to output arrays
            ivec[count+0:count+l] = perm[rows] 
            jvec[count+0:count+l] = cols

            # Endmatter
            count = count + l
            if count >= self.max_sparsity:
                if self.verbose:
                    print(f'Max sparsity reached at batch {b} of {num_batches}')
                break

        # Truncate if below max sparsity
        sparsity = min(count,self.max_sparsity)
        ivec = np.int64(ivec[0:sparsity])
        jvec = np.int64(jvec[0:sparsity])
        J_i = np.squeeze(J[indices[ivec],:],1)
        J_i = np.squeeze(J_i,1)
        J_j = np.squeeze(J[jvec,:],1)
        J_minus_J = J_i - J_j #J_minus_J = J[indices[ivec],:] - J[jvec,:]

        # Computing P1, P2
        S = sp.coo_matrix((np.ones(len(ivec)),(np.squeeze(ivec,1), np.arange(0,sparsity))), shape=[nindex, sparsity])   # nindex x sparsity
        if q > 1:
            pass # Add later
        else:
            Amat = np.exp(logC + inv2sigma2 * (J_minus_J + self.sigma * Z0.T)**2)
        A_ij_u_j_mat = Amat * np.squeeze(u_vec[jvec],1) #sparsity x MC
        S_A_ij_u_j_mat = S @ A_ij_u_j_mat #nindex x MC
        P1 = ((S@Amat) - S_A_ij_u_j_mat)/sum0 #nindex x MC
        P2 = S_A_ij_u_j_mat/sum1

        return P1, P2

    def find_nonzero(self, mat):
        """
        Returns the indices of non-zero elements of a matrix.
        """
        m,n = mat.shape
        rows = []
        cols= []
        for i in range(0,n):
            column = mat[:,i]
            for j, val in enumerate(column):
                if val != 0:
                    rows.append(j)
                    cols.append(i)

        return np.array(rows), np.array(cols)
