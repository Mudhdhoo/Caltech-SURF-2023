from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import os
from Image import Image
from scipy.io import loadmat
from Bhatt_Calculator import Bhatt_Calculator
from Reconstructor import Reconstructor
from Parameters import *
from params import *
from utils import dice
plt.rcParams["font.family"] = "Times New Roman"

class Segmenter(Bhatt_Calculator):
    """
    Segmeter class handling the image segmentation. Inherits from Bhatt_Calculator class.

    ------------ Parameters ------------

    seg_params: Segmentation_Params
        Dataclass containing all the segmentation parameters:
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
                Parameter in the optimization scheme: beta * B(J,u).

            gamma: float
                Parameter in the optimization scheme: gamma * GL_epsilon(u).

            momentum_u: float
                Parameter in the optimization scheme: momentum_u * ||u - u_n||_2^2.

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
    def __init__(self, seg_params: Segmentation_Params) -> None:
        super().__init__(seg_params.threshold_seg, seg_params.Bhatt_MC, seg_params.max_sparsity_seg, seg_params.batch_size, seg_params.sigma, seg_params.method, seg_params.verbose)    # Bhattacharyya paramters

       # self.image = image   # The image to segment
        
        # Flags
        self.method = seg_params.method
        self.dirac = seg_params.dirac
        self.verbose = seg_params.verbose

        # Segmentation Parameters
       # self.u0 = self.init_u0()     

        # Triangle 
       # self.u0 = loadmat('u0')['u0'] #########

        # Rectangle
        #self.u0 = np.zeros([7,7]) 

        self.delta = seg_params.delta;     # stopping condition
        self.GL_epsilon = seg_params.GL_epsilon    # 4*1e-1; %1e0; %In Ginzburg--Landau
     
        self.steps = seg_params.steps      # steps in the semi-implicit Euler scheme for the diffusion
        self.margin_proportion = seg_params.margin_proportion      # sets margin so given proportion of pixels are included
        self.maxiterations = seg_params.maxiterations      # Max Euler iterations
        self.grad_Bhatt_MC = seg_params.grad_Bhatt_MC         # Monte Carlo iterations for grad_u Bhatt

        # Parameters of the optimisation scheme
        self.beta = seg_params.beta
        self.gamma = seg_params.gamma
        self.momentum_u = seg_params.momentum_u

        # Plotting
        self.dice_score = []
       # self.fig, self.axs = plt.subplots(2,2)
        self.fig = plt.figure()
        self.iteration = 0
        plt.ion()       # Turn matplotlib interactive mode on

    def init_u0(self, image:Image):
        """
        Creates the initial segmentation of the image.
        """
        M1, N1 = image.image_size[0], image.image_size[1] # 2D dimension of image
        circle_center = np.array([M1/2, N1/3.5])  
        circle_radius = N1/4.5 # Hardcoded atm, can change to be dynamic later
        phi0 = np.zeros([M1, N1])
        for i in range(0,M1):     # Iterate rows
            for j in range(0,N1):     # Iterate columns
                phi0[i][j] = circle_radius - np.sqrt(np.sum(([i, j]-circle_center)**2))
        u0 = np.heaviside(phi0, 0)

        return u0
    
    def segment(self, u0, image, plotting = True):
        """
        Segments the given image using the modified MBO scheme.
        """

        dice_score = dice(u0, image.ground_truth)
        self.dice_score.append(dice_score)
        if plotting:
            self.__render(u0, image,1)
            self.__render(u0, image,0)

        u = self.__uupdate_MBO(u0, image, plotting)
        print('Finished Segmentation')
    
        #fig = plt.figure(figsize = (5,3))
       # plt.imshow(u)
      #  plt.title('Final Reconstruction', fontweight = 'bold', fontsize = 20)
       # plt.axis('off')
       #plt.savefig(f'/Users/johncao/Documents/Caltech_SURF_2023/Poster/Poster_images/recon.eps', bbox_inches='tight')
      #  plt.xlabel('Iteration', fontsize = 15)
       # plt.ylabel('Dice Score', fontsize = 15)
       # plt.grid()
       # plt.show()
        
        return u

    def __uupdate_MBO(self, u, image:Image, plotting):
        """
        Implements the modified MBO scheme 
        """
        J = image.J
        u0 = u
        dt = self.GL_epsilon/self.gamma #       time for the diffusion
        a = self.gamma*self.GL_epsilon#     parameters in the diffusion
        b = 2*self.momentum_u #     parameters in the diffusion
        M1, N1 = image.image_size[0], image.image_size[1] # 2D dimension of image

        # Compute eigendecomposition
        eye_minus_adt_Lap_inv = np.linalg.inv(np.eye(M1) - a*(dt/self.steps)*self.__Lap1D_neu(M1))
        eye_minus_adt_Lap_inv = (eye_minus_adt_Lap_inv + eye_minus_adt_Lap_inv.T)/2
        sigma, phi = np.linalg.eig(eye_minus_adt_Lap_inv)
        sigma = sigma.reshape(-1,1)

        dice_score = dice(u, image.ground_truth)
        self.dice_score.append(dice_score)

        # Run iterations
        for i in range(0,self.maxiterations):
            print('iteration: '+ str(i))

            u_old = u
            v = self.__force_diffuse(u, u0, a, b, dt, self.steps, phi, sigma);  # Forced diffusion

            #Gradient descent step in Bhatt
            margin = self.__margin_finder(v, self.margin_proportion)
            v_flattened = v.reshape(M1*N1, 1)
            indices = np.where((np.abs(v_flattened - 0.5)) <= margin)[0]     # Indicies where Bhatt coeff is calculated
            indices = np.expand_dims(indices,1)

            if self.beta > 0:     
                grad = self.__grad_Bhatt_u(J,u,indices,M1,N1)   
                v2 = v - dt * self.beta * grad # Can also add "method" argument later
            else:
                v2 = v

            # Thresholding
            u = np.heaviside(v2 - 0.5, 0)

            #Calculate dice score
            dice_score = dice(u, image.ground_truth)
            self.dice_score.append(dice_score)

            # Render the segmentation
            if plotting:    
                self.__render(u, image, i)

            # Stopping condition
            dist = np.sum(np.abs(u - u_old))
            if dist <= self.delta:
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

    def __force_diffuse(self,u,u0,a,b,t,steps,phi,sigma):
        """
        Computes V_k(x), the diffusion of U_k(x). Semi-implicit Euler Scheme.
        Performance improvement by eigendecomposition.
        """
        v = u       # Initial condition
        dt = t/steps

        Phi_T_v = phi.T @ v
        bdt_Phi_T_u0 = b*dt*phi.T @ u0
        alpha = (1 - 2 * a * dt - b * dt)
        for _ in range(0, steps):
            Phi_T_v_1 = np.copy(Phi_T_v)
            Phi_T_v_2 = np.copy(Phi_T_v)
            Phi_T_w = alpha*Phi_T_v + dt * a *(self.__shift_left(Phi_T_v_1) + self.__shift_right(Phi_T_v_2)) + bdt_Phi_T_u0
            Phi_T_v = sigma * Phi_T_w
        v = phi @ Phi_T_v

        return v

    def __grad_Bhatt_u(self,J,u,indices,M1,N1):
        """
        Computes the first variation of the Bhattacharyya coefficient.
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
        fout = 0.5 * (1/V2 * np.sqrt( np.maximum(P1_Z/(P2_Z + 1e-8),0) ) - 1/V1 * np.sqrt( np.maximum(P2_Z / (P1_Z + 1e-8),0) ))

        return fout

    def __margin_finder(self, v, proportion):
        """
        Finds the x such that the number of pixels that has a distance to 0.5 less than x is almost
        equal to some proportion of the pixels, i.e find |sum( abs(v-0.5) < x,'all') - proportion * numel(v)| < 10.
        Using interval bisection.
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
        """
        Shift the column of a matrix left with replication padding.
        """
        v[:,:-1] = v[:,1:]
        return v

    def __shift_right(self, v):
        """
        Shift the column of a matrix right with replication padding.
        """
        v[:,1:] = v[:,:-1]
        return v

    def __render(self, u, image, iteration):
        """
        Live rendering of the segmentation.
        """ 
        # M, N = u.shape
        # mask = np.ones_like(u) - u      # Create mask for plotting transparent region
        # opaque_layer = np.zeros([M, N, 4])
        # opaque_layer[:,:,0:3] = 0
        # opaque_layer[:,:,3] = mask

        # self.axs[0,1].imshow(image.image)
        # self.axs[0,1].imshow(opaque_layer, alpha = 0.5)
        # self.axs[0,1].set_title('Segmentation')
    
        # self.axs[0,0].imshow(image.image)
        # self.axs[0,0].set_title('Reconstruction')

        # self.axs[1,0].plot(self.dice_score,'b')
        # self.axs[1,0].set_title('Dice Score')

        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

        # self.fig.tight_layout()
        # plt.show()

       # plt.imshow(image.image)
        rgb_u = np.zeros(image.image_size)
        rgb_u[:,:,0] = u
        rgb_u[:,:,1] = u
        rgb_u[:,:,2] = u
        plt.imshow(image.image[:,:,:]*rgb_u)
       # plt.imshow(image.image*u)
     #   plt.imshow(opaque_layer, alpha = 0.75)
        plt.axis('off')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()

if __name__ == '__main__':
    #im = loadmat(os.path.join('images','color_rect'))['color_rect']
    #u0 = loadmat(os.path.join('images','color_rect_u0'))['u0']

    seg = Segmenter(cow_params_seg)    
    im = loadmat(os.path.join('images','cow'))['cow']
    im = Image(im, build_y = False, scale = 1)
    u0 = seg.init_u0(im)                                                                   
    u = seg.segment(u0, im)

    # im = Image('heart', build_y = False)
    # seg = Segmenter(heart_params_seg)   
    # u0 = seg.init_u0(im)                                                                             
    # u = seg.segment(u0, im)