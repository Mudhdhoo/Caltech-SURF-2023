import numpy as np
from Image import Image
import matplotlib.pyplot as plt
from Bhatt_Calculator import Bhatt_Calculator
import time
from scipy.io import loadmat

class Segmenter(Bhatt_Calculator):
    """
    Segmeter class handling the image segmentation. Inherits from Bhatt_Calculator class.

    ------------ Parameters ------------
    image: Image
        Instance of the Image class.

    delta: int
        Parameter for stopping condition in the MBO scheme. The loop is terminated if
        sum(U_k+1) - sum(U_k) < delta.

    GL_epsilon: float
        Epsilon in Ginzburg-Landau.

    steps: int
        Steps in the semi-implicit Euler scheme for the diffusion.

    margin_proportion: float
        Sets margin so given proportion of pixels are included when computing x where V(x)
        is sufficiently near 1/2.

    maxiterations: int
        Maximum iterations in the Euler scheme (SDIE).

    grad_Bhatt_MC: int
        Number of Monte-Carlo iteration for calculating grad_u_Bhatt.

    Bhatt_MC: int
        Number of Monte-Carlo iteration for calculating the Bhattacharyya coefficient.

    sigma: float
        Standarddeviation of the Bhattacharyya kernel K(z).

    beta: float
        Parameter in the optimization scheme. Multiplier of B(J,u).

    gamma: float
        Parameter in the optimization scheme. Multiplier of GL_epsilon(u).

    momentum_u: float
        Parameter in the optimization scheme. Multiplier of ||u - u_n||_2^2.

    threshold_seg: float
        Threshold above which kernel values are determined to be significant in computing P1 and P2.

    max_sparsity_seg: int
        Maximum entries in the sparse matrix when computing P1 and P2.

    batch_size: int
        Size of batches for batch processing when computing P1 and P2.

    method: str
        Method for computing the Gaussian integrals. 'random' or 'quadrature'.

    dirac: bool
        Determines if Dirac or Gaussian is used.

    verbose: bool
        Write runtime information to screen if True, otherwise write nothing.
    """

    def __init__(self, image: Image, delta, GL_epsilon, steps, margin_proportion, maxiterations, 
                grad_Bhatt_MC, Bhatt_MC, sigma, beta, gamma, momentum_u, threshold_seg, max_sparsity_seg, batch_size, method, dirac, verbose) -> None:
        super().__init__(threshold_seg, Bhatt_MC, max_sparsity_seg, batch_size, sigma, method, verbose)    # Bhattacharyya paramters

        self.image = image   # The image to segment

        # Flags
        self.method = method
        self.dirac = dirac
        self.verbose = verbose

        # Segmentation Parameters
        #self.u0 = self.init_u0()     
       # self.u0 = loadmat('u0')['u0'] #########
        self.u0 = np.zeros([7,7]) 
        self.u0[1:6,1:6] =  1

        self.delta = delta;     # stopping condition
        self.GL_epsilon = GL_epsilon    # 4*1e-1; %1e0; %In Ginzburg--Landau
     
        self.steps = steps      # steps in the semi-implicit Euler scheme for the diffusion
        self.margin_proportion = margin_proportion      # sets margin so given proportion of pixels are included
        self.maxiterations = maxiterations      # Max Euler iterations
        self.grad_Bhatt_MC = grad_Bhatt_MC         # Monte Carlo iterations for grad_u Bhatt
        #self.Bhatt_MC = Bhatt_MC      # Monte Carlo iterations for Bhatt

        # Parameters of the optimisation scheme
        self.beta = beta
        self.gamma = gamma
        self.momentum_u = momentum_u

        # Plotting
        self.fig = plt.figure()
        plt.ion()

    def init_u0(self):
        """
        Creates the initial segmentation of the image.
        """
        M1, N1 = self.image.image_size[0], self.image.image_size[1] # 2D dimension of image
        circle_center = np.array([M1/2.5, N1/2])  
        circle_radius = N1/5 # Hardcoded atm, can change to be dynamic later
        phi0 = np.zeros([M1, N1])
        for i in range(0,M1):     # Iterate rows
            for j in range(0,N1):     # Iterate columns
                phi0[i][j] = circle_radius - np.sqrt(np.sum(([i, j]-circle_center)**2))
        u0 = np.heaviside(phi0, 0)

        return u0
    
    def segment(self):
        """
        Segments the given image using the modified MBO scheme.
        """
        u = self.__uupdate_MBO(self.u0)

        return u

    def __uupdate_MBO(self, u):
        """
        Implements the modified MBO scheme 
        """
        J = self.image.J
        u0 = u
        dt = self.GL_epsilon/self.gamma #       time for the diffusion
        a = self.gamma*self.GL_epsilon#     parameters in the diffusion
        b = 2*self.momentum_u #     parameters in the diffusion
        M1, N1 = self.image.image_size[0], self.image.image_size[1] # 2D dimension of image

        # Compute eigendecomposition
        eye_minus_adt_Lap_inv = np.linalg.inv(np.eye(M1) - a*(dt/self.steps)*self.__Lap1D_neu(M1))
        eye_minus_adt_Lap_inv = (eye_minus_adt_Lap_inv + eye_minus_adt_Lap_inv.T)/2
        sigma, phi = np.linalg.eig(eye_minus_adt_Lap_inv)

        #############################################
        temp = sigma[-1]
        sigma[-1] = sigma[-2]
        sigma[-2] = temp
        sigma = sigma.reshape(-1,1)

        phi[:,-2:] = np.flip(phi[:,-2:])
        #############################################

        self.__render(u)
        # Run iterations
        for i in range(0,self.maxiterations):
            u_old = u
            v = self.__force_diffuse(u,u0,a,b,dt,steps,phi,sigma);  # Forced diffusion

            #Gradient descent step in Bhatt
            margin = self.__margin_finder(v,margin_proportion)
            v_flattened = v.reshape(M1*N1, 1)
            indices = np.where((np.abs(v_flattened - 0.5)) <= margin)[0]     # Indicies where Bhatt coeff is calculated
            indices = np.expand_dims(indices,1)

            if self.beta > 0:     
                grad = self.__grad_Bhatt_u(J,u,indices,M1,N1)   
                v2 = v - dt * beta * grad # Can also add "method" argument later
            else:
                v2 = v

            # Thresholding
            u = np.heaviside(v2 - 0.5, 0)

            # Render the segmentation
            self.__render(u)

            # Stopping condition
            dist = np.sum(np.abs(u - u_old))
            if dist < self.delta:
                break

        return u

    def __Lap1D_neu(self, M):
        """
        Creates the Delta^(M) matrix that implements the 3 point stencil matrix
        with Neumann boundary conditions.
        """
        # Implement sparse matrices for speed up?

        # Create 3 point stencil matrix
        diagonal = -np.diag(np.ones(M)*2)
        upper_offset = np.diag(np.ones(M-1), k = 1)
        lower_offset = np.diag(np.ones(M-1), k = -1)
        lap1D = diagonal + upper_offset + lower_offset
        
        # Implement Neumann boundary conditions
        lap1D[0,0], lap1D[M-1,M-1] = -1, -1

        return lap1D

    def __force_diffuse(self,u,u0,a,b,dt,steps,phi,sigma):
        """
        Computes V_k(x), the diffusion of U_k(x). Semi-implicit Euler Scheme.
        Performance improvement by eigendecomposition.
        """
        v = u       # Initial condition
        #tau = dt/steps

        dt = dt/steps #######################

        # w = phi.T@v      

        # for _ in range(0, steps):
        #     w_new = sigma * ((1 - 2*a*tau - b*tau)*w + tau*(a*self.__shift_left(w) + a*self.__shift_right(w) + b*phi.T@u0))
        #     w = w_new
        # v = phi@w

        ####################################
        Phi_T_v = phi.T @ v
        bdt_Phi_T_u0 = b*dt*phi.T @ u0
        alpha = (1 - 2 * a * dt - b * dt)
        for _ in range(0, steps):
            Phi_T_v_1 = np.copy(Phi_T_v)
            Phi_T_v_2 = np.copy(Phi_T_v)
            Phi_T_w = alpha*Phi_T_v + dt * a *(self.__shift_left(Phi_T_v_1) + self.__shift_right(Phi_T_v_2)) + bdt_Phi_T_u0
            Phi_T_v = sigma * Phi_T_w

        v = phi @ Phi_T_v

        ####################################
        return v

    def __grad_Bhatt_u(self,J,u,indices,M1,N1):
        """
        Computes the first variation of the Bhattacharyya coefficient
        """
        # Compute 0.5*(V_1^-1 + V_2^-2)*B(J,u)
        n,q = J.shape
        V1 = np.sum(1-u)
        V2 = np.sum(u)

        s = 1/V2 - 1/V1
        is_index = np.zeros([n,1])
        is_index[indices] = 1       # a vector which is 1 at a desired index and 0 at the rest
        Z0, W0 = self.Z0_calculator(self.grad_Bhatt_MC,q)

        #######################################
        Z01 = np.array([[0.708056269489433],
                        [-0.185741243465675],
                        [0.881584314620342],
                        [-1.796442415931539],
                        [-1.131685703980258],
                        [-1.090086866765954],
                        [-1.145274943496607],
                        [2.393388102731935],
                        [1.730493141993269],
                        [-0.619657736298477]])
        #######################################

        grad =  is_index * 0.5 * s * self.Bhatt(J,u)    # 0.5*(V_1^-1 + V_2^-2)*B(J,u)

        # Compute integral of Q(z)  
        u = u.reshape(n,1)
        grad1 = Z0.shape[0]*np.sum(self.__B_grad_u_integrand(J, u, Z0, indices, V1, V2) * W0, 1) ########### Problem
        grad1 = np.expand_dims(grad1,1) 

        indices = np.squeeze(indices,1)
        grad[indices] = grad[indices] + grad1
        grad = grad.reshape(M1, N1)

        return grad

    def __B_grad_u_integrand(self, J, u_vec, Z0, indices, V1, V2):
        """
        Computes f(J,u,z) in the gradient.
        """
        MC, q = Z0.shape
        n = J.shape[0]
        nindex = len(indices)
        P1_Z, P2_Z = self.Pcalculator_sparse2(J, u_vec, Z0, indices)     # Calculate P1 and P2
        #fout = 0.5 * (1/V2 * np.sqrt( P1_Z/(P2_Z + 1e-8) ) - 1/V1 * np.sqrt( P2_Z / (P1_Z + 1e-8) ))
        fout = 0.5 * (1/V2 * np.sqrt( np.maximum(P1_Z/(P2_Z + 1e-8),0) ) - 1/V1 * np.sqrt( np.maximum(P2_Z / (P1_Z + 1e-8),0) ))

        return fout

    def __margin_finder(self, v, proportion):
        """
        Finds the x such that the number of pixels that has a distance to 0.5 less than x is almost
        equal to some proportion of the pixels, i.e find |sum( abs(v-0.5) < x,'all') - proportion * numel(v)| < 10.
        Using interval bisection
        """
        N = np.size(v)
        v = v.reshape(N,1) - 0.5
        phi = abs(v)
        MAX = 0.5
        MIN = 0
        for _ in range(1, 1000 + 1):
            margin = (MIN + MAX)/2
            val = sum(phi < margin) - proportion * N
            if abs(val) < 10:
                break
            if val > 0:
                MAX = margin
            else:       
                MIN = margin

        return margin

    def __shift_left(self, v):
        v[:,:-1] = v[:,1:]
        return v

    def __shift_right(self, v):
        v[:,1:] = v[:,:-1]
        return v

    def __render(self,u):
        """
        Live rendering of the segmentation.
        """ 
        segmentation = u*self.image.image
        #segmentation = u
        plt.imshow(segmentation)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()

if __name__ == '__main__':
    im = Image('rectangle')
    delta = 2
    GL_epsilon = 1
    steps = 100
    margin_proportion = 0.05
    maxiterations = 25
    grad_Bhatt_MC = 10
    Bhatt_MC = 50
    sigma = 1e-2
    beta = 2*1e2
    gamma = 2*2*1e-2
    momentum_u = 1e-7
    threshold_seg = 0.01    # TODO threshold_seg = min(threshold_seg,C);
    max_sparsity_seg = 62500 # TODO max_sparsity_seg = min(max_sparsity_seg,(M1*N1)^2);
    batch_size = 700
    method = 'random'
    dirac = 0
    verbose = True

    seg = Segmenter(im, delta, GL_epsilon, steps, margin_proportion, maxiterations, grad_Bhatt_MC, Bhatt_MC, sigma, beta, gamma, momentum_u, threshold_seg, max_sparsity_seg, batch_size, method, dirac, verbose)
    seg.segment()

    #u0 = seg.init_u0()
    # u0 = seg.u0
    # print(u0)
    # fig = plt.figure()
    # plt.imshow(u0)
    # plt.show()
