import numpy as np
from Image import Image
import matplotlib.pyplot as plt

class Segmenter:
    def __init__(self, image: Image, delta, GL_epsilon, steps, margin_proportion, maxiterations, 
                grad_Bhatt_MC, Bhatt_MC, beta, gamma, momentum_u, threshold_seg, max_sparsity_seg, batch_size) -> None:
        self.image = image   # The image to segment

        # Segmentation Parameters
        self.u0 = self.__init_u0()     
        self.delta = delta;     #stopping condition
        self.GL_epsilon = GL_epsilon    #4*1e-1; %1e0; %In Ginzburg--Landau
        self.steps = steps      # steps in the semi-implicit Euler scheme for the diffusion
        self.margin_proportion = margin_proportion      #0.05; %0.075 %sets margin so given proportion of pixels are included
        self.maxiterations = maxiterations      #Max Euler iterations
        self.grad_Bhatt_MC = grad_Bhatt_MC         #Monte Carlo iterations for grad_u Bhatt
        self.Bhatt_MC = Bhatt_MC      #Monte Carlo iterations for Bhatt

        # Parameters of the optimisation scheme
        self.beta = beta
        self.gamma = gamma
        self.momentum_u = momentum_u

        # Parameters for computing P1, P2
        self.threshold_seg = threshold_seg
        self.max_sparsity_seg  = max_sparsity_seg # 1e7; %Maximum entries in sparse matrix
        self.batch_size = batch_size # size of batches for batch processing

    def __init_u0(self):
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
        pass

    def __uupdate_MBO(self, verbose = False):
        """
        Implements the modified MBO scheme 
        """
        J = self.image.J
        u = self.u0
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
            v = self.__force_diffuse(u,self.u0,a,b,dt,steps,eye_minus_adt_Lap_inv,phi,sigma,verbose);  # Forced diffusion

            # Gradient descent step in Bhatt
            if self.beta > 0:
                v2 = v - dt * beta * self.__grad_Bhatt_u(J,u,indices,M1,N1,grad_Bhatt_MC,Bhatt_MC,self.threshold_seg, self.max_sparsity_seg,self.batch_size,sigma,verbose)  # Can also att "method" argument later
            else:
                v2 = v

            # Thresholding
            u = np.heaviside(v2 - 0.5, 0)

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
        # Create 3 point stencil matrix
        diagonal = -np.diag(np.ones(M)*2)
        upper_offset = np.diag(np.ones(M-1), k = 1)
        lower_offset = np.diag(np.ones(M-1), k = -1)
        lap1D = diagonal + upper_offset + lower_offset
        
        # Implement Neumann boundary conditions
        lap1D[0,0], lap1D[M-1,M-1] = -1, -1

        return lap1D

    def __force_diffuse(self):
        pass

    def __grad_Bhatt_u(J,u,indices,M1,N1,grad_Bhatt_MC,Bhatt_MC,threshold,max_sparsity,batch_size,sigma,method,verbose):
        pass

    def __margin_finder(self):
        pass

if __name__ == '__main__':
    im = Image('image')
    delta = 8
    GL_epsilon = 1e0
    steps = 10
    margin_proportion = 0.0225
    maxiterations = 50
    beta         = 2*1e2
    gamma        = 2*2*1e-2
    momentum_u   = 1e-5
    grad_Bhatt_MC = 10
    Bhatt_MC = 50
    seg = Segmenter(im, delta, GL_epsilon, steps, margin_proportion, maxiterations, grad_Bhatt_MC, Bhatt_MC, beta, gamma, momentum_u)
    seg.uupdate_MBO()


