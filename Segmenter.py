import numpy as np
from Image import Image
import matplotlib.pyplot as plt
from Bhatt_Calculator import Bhatt_Calculator

class Segmenter(Bhatt_Calculator):
    """
    Segmeter class.
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
        self.u0 = self.init_u0()     
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

        # Parameters for computing P1, P2
       # self.threshold_seg = threshold_seg
       # self.max_sparsity_seg  = max_sparsity_seg # 1e7; %Maximum entries in sparse matrix
       # self.batch_size = batch_size # size of batches for batch processing

    def init_u0(self):
        """
        Creates the initial segmentation of the image.
        """
        M1, N1 = self.image.image_size[0], self.image.image_size[1] # 2D dimension of image
        circle_center = np.array([M1/2, N1/2])  
        circle_radius = N1/5    # Hardcoded atm, can change to be dynamic later
        phi0 = np.zeros([M1, N1])
        for i in range(0,M1):     # Iterate rows
            for j in range(0,N1):     # Iterate columns
                phi0[i][j] = circle_radius - np.sqrt(np.sum(([i, j]-circle_center)**2))
        u0 = np.heaviside(phi0, 0)
        return u0
    
    def segment(self):
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
                v2 = v - dt * beta * self.__grad_Bhatt_u(J,u,indices,M1,N1)  # Can also add "method" argument later
            else:
                v2 = v

            # Thresholding
            u = np.heaviside(v2 - 0.5, 0)

            # Stopping condition
            dist = np.sum(np.abs(u - u_old))
            if dist < self.delta:
                break
        print(u)
        return u

    def __Lap1D_neu(self, M):
        """
        Creates the Delta^(M) matrix that implements the 3 point stencil matrix
        with Neumann boundary conditions.
        """
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
        tau = dt/steps
        sigma = sigma.reshape([256,1])
        w = phi.T@v      

        for _ in range(0, steps):
            w_new = sigma * ((1 - 2*a*tau - b*tau)*w + tau*(a*self.__shift_left(w) + a*self.__shift_right(w) + b*phi.T@u0))
            w = w_new
        v = phi@w

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
        grad =  is_index * 0.5 * s * self.Bhatt(J,u)    # 0.5*(V_1^-1 + V_2^-2)*B(J,u)

        # Compute integral of Q(z)
        u = u.reshape(n,1)
        grad1 = Z0.shape[0]*np.sum(self.__B_grad_u_integrand(J, u, Z0, indices, V1, V2) * W0, 1)
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
        fout = 0.5 * (1/V2 * np.sqrt( P1_Z/(P2_Z + 1e-8) ) - 1/V1 * np.sqrt( P2_Z / (P1_Z + 1e-8) ))

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

if __name__ == '__main__':
    im = Image('heart')
    delta = 8
    GL_epsilon = 1e0
    steps = 10
    margin_proportion = 0.0225
    maxiterations = 50
    grad_Bhatt_MC = 10
    Bhatt_MC = 50
    sigma = 1e-2
    beta = 2*1e2
    gamma = 2*2*1e-2
    momentum_u = 1e-5
    threshold_seg = 0.25
    max_sparsity_seg = 2000000
    batch_size = 700
    method = 'random'
    dirac = 0
    verbose = True
    seg = Segmenter(im, delta, GL_epsilon, steps, margin_proportion, maxiterations, grad_Bhatt_MC, Bhatt_MC, sigma, beta, gamma, momentum_u, threshold_seg, max_sparsity_seg, batch_size, method, dirac, verbose)
    seg.segment()


