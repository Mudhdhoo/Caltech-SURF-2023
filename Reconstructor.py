import numpy as np
import matplotlib.pyplot as plt 
import os
from skimage.restoration import denoise_tv_chambolle
from Bhatt_Calculator import Bhatt_Calculator
from Image import Image
from Parameters import Reconstruction_Params
from scipy.io import loadmat
from scipy.sparse import diags, eye, kron
from params import *

class Reconstructor(Bhatt_Calculator):
    """
    Reconstructor class handling the image segmentation. Inherits from Bhatt_Calculator class.

    ------------ Parameters ------------

    recon_params: Reconstuction_Params
        Dataclass containing all the reconstruction parameter:
            momentum_Im: float
                Parameter of the optimization scheme: momentum_Im * ||I - I_n||^2_2

            alpha: float
                Parameter of the optimization scheme: alpha * ||T(I) - y||_F^2

            beta: float
                Parameter of the optimization scheme: beta * B(F(I), u_n)

            gfn_MC: int
                Number of Monte-Carlo samples to compute g.

            reg_a: float
                Hyperparameter for the Primal-Dual algorithm.

            reg_epsilon: float
                Hyperparameter for the Primal-Dual algorithm.

            threshold_gfn: float
                Threshold above which kernel values are determined to be significant in computing P1 and P2.

            max_sparsity_gfn: float
                Maximum entries in the sparse matrix when computing P1 and P2.

            method: str
                Method for computing the gaussian integrals. Either random or quadrature. Samples random points
                if "random". Samples from a Gauss-Hermite quadrature of "quadrature"'

            verbose: Bool
                Write runtime information to screen if True, otherwise write nothing.

    algorithm: str
        The algorithm used for cheap reconstruction.

    TV_weight: float
        Hyperparameter for the TV-denoising algorithm.

    """
    def __init__(self, recon_params: Reconstruction_Params, algorithm: str, TV_weight = 1) -> None:
        super().__init__(recon_params.threshold_gfn, 25, recon_params.max_sparsity_gfn, recon_params.batch_size, recon_params.sigma, recon_params.method, recon_params.verbose)
        self.momentum_Im = recon_params.momentum_im
        self.alpha = recon_params.alpha
        self.beta = recon_params.beta
        self.TV_weight = TV_weight
        self.gfn_MC = recon_params.gfn_MC
        self.reg_a = recon_params.reg_a
        self.reg_epsilon = recon_params.reg_epsilon
        self.y = None
        if algorithm not in ['TV', 'TGV', 'BM3D', 'none']:
            raise Exception('Reconstruction algorithm must be either "TV", "TGV", "BM3D" or "none".')
        self.algorithm = algorithm

    def reconstruct(self,image:Image, u):
        im = image.image
        self.y = image.y

        # Compute linearization
        g = self.gfn(image, u)
        tildeIm = im - 0.5*self.beta/self.momentum_Im * g

        im = self.Imupdate_linear(im, tildeIm)

        return im

    def cheap_reconstruction(self, y):
        """
        Cheap initial reconstruction of the image. 
        """
        is_rgb = True
        if len(y.shape) == 2:
            is_rgb = False

        if self.algorithm == 'TV':
            denoised_im = denoise_tv_chambolle(y, weight = self.TV_weight, multichannel = is_rgb)

        if self.algorithm == 'TGV':
            pass

        if self.algorithm == 'BM3D':
            pass

        if self.algorithm == 'none':
            pass

        return denoised_im
        
    def gfn(self, image:Image, u):

        M, N = image.image_size[0], image.image_size[1]
        J = image.J
        n, q = J.shape
        u_vec = u.reshape(-1,1)
        Z0, W0 = self.Z0_calculator(self.gfn_MC, q)
        fZ0 = self.B_grad_J_integrand(J, u_vec, Z0)
        gradJ = np.sum(fZ0 * W0.reshape([1, 1, len(W0[0])]), axis = 2)
        g = gradJ.reshape(image.image_size)

        return g

    def B_grad_J_integrand(self, J, u_vec, Z0):
        """
        Takes MC x q -> n x q x MC 
        Inputs: J - n x q array
                u_vec - n x 1 array
                Z0 - MC x q (random) array
                Bhatt_KER - function
                K - scalar, << n
        Outputs:fout - n x q x MC array
        """
        # Identify dimensions
        MC, q = Z0.shape
        n = len(u_vec)
        indices = np.arange(0, n).reshape(-1,1)

        # Evaluate distributions P1,P2, output is n x MC
        P1_Z, P2_Z = self.Pcalculator_sparse2(J, u_vec, Z0, indices)

        # Find indices where u[indices] == 0
        u_index_0 = np.where(u_vec == 0)[0]
        u_index_1 = np.where(u_vec == 1)[0]
        sum0 = np.sum(1 - u_vec)
        sum1 = n - sum0
        
        # Evaluate fout at each index
        if q == 1:
            # Greyscale
            fout_u0 = 0.5 * self.sigma**(-2) * (1/sum0) * np.sqrt(P2_Z[u_index_0,:] / (P1_Z[u_index_0,:]+1e-8)) * np.squeeze(np.tile(np.reshape(self.sigma*Z0,[1, MC, q]), [int(sum0), 1,1]))
            fout_u1 = 0.5 * self.sigma**(-2) * (1/sum1) * np.sqrt(P1_Z[u_index_1,:] / (P2_Z[u_index_1,:]+1e-8)) * np.squeeze(np.tile(np.reshape(self.sigma*Z0,[1, MC, q]), [int(sum1), 1,1]))
        else:
            # RGB
            z_minus_J_0 = np.squeeze(np.tile(np.reshape(self.sigma*Z0,[1, MC, q]), [int(sum0), 1,1]))
            z_minus_J_1 = np.squeeze(np.tile(np.reshape(self.sigma*Z0,[1, MC, q]), [int(sum1), 1,1]))
            sqrt_P2_div_P1 = np.sqrt(P2_Z[u_index_0,:] / (P1_Z[u_index_0,:]+1e-8))
            sqrt_P1_div_P2 = np.sqrt(P1_Z[u_index_1,:] / (P2_Z[u_index_1,:]+1e-8))

            sqrt_P2_div_P1_mult_z_minus_J = z_minus_J_0
            sqrt_P2_div_P1_mult_z_minus_J[:,:,0] = sqrt_P2_div_P1_mult_z_minus_J[:,:,0] * sqrt_P2_div_P1
            sqrt_P2_div_P1_mult_z_minus_J[:,:,1] = sqrt_P2_div_P1_mult_z_minus_J[:,:,1] * sqrt_P2_div_P1
            sqrt_P2_div_P1_mult_z_minus_J[:,:,2] = sqrt_P2_div_P1_mult_z_minus_J[:,:,2] * sqrt_P2_div_P1

            sqrt_P1_div_P2_mult_z_minus_J = z_minus_J_1
            sqrt_P1_div_P2_mult_z_minus_J[:,:,0] = sqrt_P1_div_P2_mult_z_minus_J[:,:,0] * sqrt_P1_div_P2
            sqrt_P1_div_P2_mult_z_minus_J[:,:,1] = sqrt_P1_div_P2_mult_z_minus_J[:,:,1] * sqrt_P1_div_P2
            sqrt_P1_div_P2_mult_z_minus_J[:,:,2] = sqrt_P1_div_P2_mult_z_minus_J[:,:,2] * sqrt_P1_div_P2

            fout_u0 = 0.5 * self.sigma**(-2) * (1/sum0) * sqrt_P2_div_P1_mult_z_minus_J
            fout_u1 = 0.5 * self.sigma**(-2) * (1/sum1) * sqrt_P1_div_P2_mult_z_minus_J 

        #Combine elements of fout
        if q == 1:
            fout = np.zeros([n, MC])
            fout[u_index_0, :] = fout_u0
            fout[u_index_1, :] = fout_u1
            fout = fout.reshape([n, q, MC])
        else:
            fout = np.zeros([n, MC, q])
            fout[u_index_0, :, :] = fout_u0
            fout[u_index_1, :, :] = fout_u1
            fout = fout.reshape([n, q, MC])

        return fout

    def Imupdate_linear(self, im_old, tilde_im):
        M, N = im_old.shape[0], im_old.shape[1]
        l = len(im_old.shape)
        if l == 2:
            D1, D2 = self.grad_forward(im_old)        
        else:
            D1, D2 = self.grad_forward(im_old[:,:,0])    
        
        im_new = self.primal_dual(im_old, 4*self.momentum_Im, tilde_im, D1, D2, M, N, l)

        return im_new

    def grad_forward(self,u):
        # Forward difference
        M,N = u.shape

        one = np.ones([M])
        one[-1] = 0         # Boundary conditions
        D1 = diags([-one, one], [0, 1], shape = [M,M])

        one = np.ones([N])
        one[-1] = 0  
        D2 = diags([-one, one], [0, 1], shape = [N,N])

        D1 = kron(eye(N), D1)
        D2 = kron(D2, eye(M))

        return D1, D2

    def primal_dual(self, x, mu, tildeIm, D1, D2, M, N, l):
        niter = 500
        theta = 1
        tol = 1e-2
        L = 8

        tau   = 0.99/L
        sigma0 = 0.99/(tau*L)
        gamma = 0.5*mu

        xhat   = x
        y      = self.K(x, D1, D2, M, N, l)
        xstar  = self.KS(y, D1, D2, M, N, l)
        res    = np.zeros([niter,1])

        for iter in range(0,niter):
            x_old     = x
            y_old     = y
            Kx_old    = self.K(x, D1, D2, M, N, l)
            xstar_old = xstar
            
            # DUAL PROBLEM
            Kx_hat = self.K(xhat, D1, D2, M, N, l)
            y      = self.proxFS(y + sigma0*Kx_hat, sigma0)
            # PRIMAL PROBLEM
            xstar = self.KS(y, D1, D2, M, N, l)
            x     = self.proxG(x-tau*xstar,tau, tildeIm)
            # EXTRAPOLATION
            xhat = x + theta * (x-x_old)

            # ACCELERATION
            theta = 1 / np.sqrt(1+2*gamma*tau)
            tau   = theta*tau
            sigma0 = sigma0/theta

            # primal residual
            p_res = (x_old-x)/tau - (xstar_old-xstar)
            p = np.sum(abs(p_res))
            # dual residual
            d_res = (y_old-y)/sigma0 - (Kx_old-Kx_hat)
            d = np.sum(abs(d_res))
            
            res[iter] = (np.sum(p)+np.sum(d)) / np.size(x)

            if res[iter] < tol:
                break
        
        return x
    
    def grad_channel(self,u, D1, D2, M, N):
        u = u.reshape(-1,1)
        return np.reshape(np.concatenate((D1@u, D2@u),1), [M, N, 2])
        #return np.reshape(np.concatenate((D1@u, D2@u),1), [2, M, N])

    def div_channel(self,v, D1, D2, M, N):
        return np.reshape(D1.T @ np.reshape(v[:,:,0], [M*N, 1]) + D2.T @ np.reshape(v[:,:,1], [M*N, 1]), [M, N])

    def K(self, u, D1, D2, M, N, l):
        if l > 2:
            return np.concatenate((self.grad_channel(u[:,:,0],D1,D2,M,N), self.grad_channel(u[:,:,0],D1,D2,M,N), self.grad_channel(u[:,:,0],D1,D2,M,N)), 2)
        
        return self.grad_channel(u, D1, D2, M, N)

    def KS(self, v, D1, D2, M, N, l):
        if l > 2:
            return np.stack((self.div_channel(v[:,:,0:2], D1, D2, M, N), self.div_channel(v[:,:,2:4], D1, D2, M, N), self.div_channel(v[:,:,4:6], D1, D2, M, N)),2)

        return self.div_channel(v[:,:,0:2], D1, D2, M, N)

    def proxFS(self, y, sigma0):
        return (y / (1 + sigma0*self.reg_epsilon)) / np.stack([np.maximum(1, self.norms(y / (1 + sigma0*self.reg_epsilon), 2, 2) / self.reg_a)]*y.shape[2], 2)

    def proxG(self, q, tau, tildeIm):
        return self.Ainv(2*self.alpha*self.Tadj(self.y) + 2*self.momentum_Im*tildeIm + q/tau, tau)

    def T(self, x):
        # If blur == 0
        return x
    
    def Tadj(self, x):
        # if blur == 0
        return x

    def Ainv(self, x, p):
        return x / (1/p + 2*self.alpha + 2*self.momentum_Im) 

    def norms(self, z, p, dir):
        if p == 1:
            y = np.sum(abs(z),dir)
        elif p == 'inf':
            y = np.max(z.reshape(-1,1))
        else:
            y = np.sum(z**p, dir)**(1/p)

        return y

if __name__ == '__main__':
    im = Image('cow')
    recon = Reconstructor(cow_params_recon, 'TV', TV_weight = 1)
    u = loadmat('u.mat')['u']
    rec_im = recon.cheap_reconstruction(im.image)
    im.update_image(rec_im) # Update the image

    new_im = recon.reconstruct(im, u)
   # fig, axs = plt.subplots(1,2)
    #axs[1].imshow(new_im)
    #axs[0].imshow(im.image)
    plt.imshow(new_im)
    plt.show()