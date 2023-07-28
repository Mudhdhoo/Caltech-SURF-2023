from re import A
import numpy as np
import scipy.sparse as sp
from numpy.matlib import repmat
import time

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
        Z0 = np.array([[-0.630254567737951,
                        3.002049941761356,
                        0.811523935348309,
                        0.814732795773040,
                        -0.159896587337345,
                        -0.312704541017745,
                        0.814782078244699,
                        1.179967943037499,
                        1.459282139680454,
                        1.375259510665675,
                        1.092592061916827,
                        -0.024784250530783,
                        -0.026131419139398,
                        -0.879148941815737,
                        -0.134439026341600,
                        0.392561811482352,
                        -1.391248438007471,
                        -0.408388238239126,
                        -0.407598940833737,
                        0.093749542417616,
                        -0.326930871476909,
                        -0.323100211148631,
                        -1.308811412054652,
                        -1.158756576085709,
                        -1.294497636454334,
                        0.308368086409617,
                        0.432375611810973,
                        -1.873024877129877,
                        -1.123501499312582,
                        -0.369292987658857,
                        0.543207695793634,
                        0.209104562663669,
                        0.860536810080937,
                        -1.352695338915951,
                        -0.656084245108649,
                        -1.351046864732875,
                        0.089603499984597,
                        -0.814711163218137,
                        0.762269279829454,
                        -1.689402186625835,
                        0.080682696121877,
                        -2.162310352286902,
                        -0.909154477079780,
                        -0.078678618726943,
                        0.398146355933663,
                        0.332133703506583,
                        -1.757870525033633,
                        0.472557971350306,
                        0.799484467152829,
                        -0.798391311005358]]).T

        indices = np.random.randint(0,int(n), Z0.shape)     # Pick random indicies for Monte-Carlo sampling
        indices = np.array([[37,
                            6,
                            44,
                            26,
                            1,
                            8,
                            13,
                            2,
                            34,
                            2,
                            25,
                            24,
                            14,
                            29,
                            15,
                            46,
                            1,
                            28,
                            19,
                            19,
                            2,
                            34,
                            28,
                            38,
                            39,
                            13,
                            10,
                            18,
                            36,
                            7,
                            22,
                            1,
                            4,
                            35,
                            40,
                            45,
                            45,
                            31,
                            7,
                            4,
                            39,
                            37,
                            31,
                            36,
                            10,
                            39,
                            48,
                            21,
                            13,
                            24]]).T

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
                #perm_batch_indices = perm[I_start:I_end+1]
                #randomize_ind = np.array([int(indices[i]) for i in perm_batch_indices])
                randomize_ind1 = np.array([46,
39,
34,
31,
 7,
13,
13,
 1,
29,
45,
 8,
 2,
37,
45,
28,
19,
31,
 7,
 2,
21,
22,
10,
26,
39,
40,
 4,
37,
34,
 2,
 4,
39,
14,
24,
25,
15,
35,
48,
18,
 6,
13,
 1,
36,
38,
24,
 1,
36,
10,
28,
19,
44])
                J_I_minus_J = J[randomize_ind1-1] - J_1nq
                #J_I_minus_J = np.array([J[i] for i in randomize_ind]) - J_1nq  #this_batch_size x n
                KER_above_threshold1 = J_I_minus_J < threshold_val
                KER_above_threshold2 = J_I_minus_J > - threshold_val
                KER_above_threshold = np.logical_and(KER_above_threshold1,KER_above_threshold2)
            #else?

            ############# Construction Site #############
            #rows, cols = np.nonzero(KER_above_threshold)
            rows, cols = self.find_nonzero(KER_above_threshold)

            rows = np.expand_dims(rows[0:self.max_sparsity-count],1)
            cols = np.expand_dims(cols[0:self.max_sparsity-count],1)
            ##############################################

            # Convert to global indices
            rows = rows + I_start - 0 ########### -1?
            l = len(rows)
            # Incrementally add to output arrays
            ivec[count+0:count+l] = np.random.permutation(rows)
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
        #J_minus_J = J_i - J_j #J_minus_J = J[indices[ivec],:] - J[jvec,:]

        ################################
        J_minus_J = np.zeros(jvec.shape)
        ################################

        # Computing P1, P2
        S = sp.coo_matrix((np.ones(len(ivec)),(np.squeeze(ivec,1), np.arange(0,sparsity))), shape=[nindex, sparsity])   # nindex x sparsity
        if q > 1:
            pass # Add later
        else:
            Amat = np.exp(logC + inv2sigma2 * (J_minus_J + self.sigma * Z0.T)**2)
        A_ij_u_j_mat = Amat * np.squeeze(u_vec[jvec],1) #sparsity x MC
        S_A_ij_u_j_mat = S @ A_ij_u_j_mat #nindex x MC
        #P1 = ((S*Amat) - S_A_ij_u_j_mat)/sum0 #nindex x MC
        P1 = ((S@Amat) - S_A_ij_u_j_mat)/sum0 #nindex x MC
        P2 = S_A_ij_u_j_mat/sum1
       # print(P1)
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
