import numpy as np
import matplotlib.pyplot as plt 
import os
from skimage.restoration import denoise_tv_chambolle
from Bhatt_Calculator import Bhatt_Calculator
from Image import Image
from Parameters import Reconstruction_Params
from scipy.io import loadmat


class Reconstructor(Bhatt_Calculator):
    def __init__(self, recon_params: Reconstruction_Params, algorithm: str, TV_weight = 1) -> None:
        super().__init__(recon_params.threshold_gfn, 25, recon_params.max_sparsity_gfn, recon_params.batch_size, recon_params.sigma, recon_params.method, recon_params.verbose)
        self.momentum_Im = recon_params.momentum_im
        self.alpha = recon_params.alpha
        self.beta = recon_params.beta
        self.TV_weight = TV_weight
        self.gfn_MC = recon_params.gfn_MC
        if algorithm not in ['TV', 'TGV', 'BM3D', 'none']:
            raise Exception('Reconstruction algorithm must be either "TV", "TGV", "BM3D" or "none".')
        self.algorithm = algorithm

    def reconstruct(self,image:Image, u):
        g = self.gfn(image, u)
       # tildeIm = Im - 0.5*beta/momentum_Im * g

    def cheap_reconstruction(self, y):
        """
        Cheap initial reconstruction of the image. 
        """
        if self.algorithm == 'TV':
            denoised_im = denoise_tv_chambolle(y, weight = self.TV_weight)

        if self.algorithm == 'TGV':
            pass

        if self.algorithm == 'BM3D':
            pass

        if self.algorithm == 'none':
            pass

        return denoised_im
        
    def gfn(self, image:Image, u):
        M, N = image.image_size
        J = image.J
        n, q = J.shape
        u_vec = u.reshape(-1,1)
        Z0, W0 = self.Z0_calculator(self.gfn_MC, q)

        W0 = np.array([1.52247580e-09, 1.05911555e-06, 1.00004441e-04, 2.77806884e-03,
        3.07800339e-02, 1.58488916e-01, 4.12028687e-01, 5.64100309e-01,
        4.12028687e-01, 1.58488916e-01, 3.07800339e-02, 2.77806884e-03,
        1.00004441e-04, 1.05911555e-06, 1.52247580e-09])

        Z0 = np.array([[-4.5000,
                    -3.6700,
                    -2.9672,
                    -2.3257,
                    -1.7200,
                    -1.1361,
                    -0.5651,
                            0,
                        0.5651,
                        1.1361,
                        1.7200,
                        2.3257,
                        2.9672,
                        3.6700,
                        4.5000]]).T

        fZ0 = self.B_grad_J_integrand(J, u_vec, Z0)
        gradJ = np.sum(fZ0 * W0.reshape([1, 1, len(W0)]), axis = 2)
        g = gradJ.reshape([256, 256])

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
        fout_u0 = 0.5 * self.sigma**(-2) * (1/sum0) * np.sqrt(P2_Z[u_index_0,:] / (P1_Z[u_index_0,:]+1e-8)) * np.squeeze(np.tile(np.reshape(self.sigma*Z0,[1, MC, q]), [int(sum0), 1,1]))
        fout_u1 = 0.5 * self.sigma**(-2) * (1/sum1) * np.sqrt(P1_Z[u_index_1,:] / (P2_Z[u_index_1,:]+1e-8)) * np.squeeze(np.tile(np.reshape(self.sigma*Z0,[1, MC, q]), [int(sum1), 1,1]))

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

    def Imupdate_linear(self):
        pass

    def T(self,x):
        # If blur == 0
        return x
    
    def Tadj(self,x):
        # if blur == 0
        return x

    def Ainv(self, x, p):
        return x / (1/p + 2*self.alpha + 2*self.momentum_Im) 

if __name__ == '__main__':
    im = Image('heart')
    recon_params = Reconstruction_Params(momentum_im = 1,
                                         sigma = 1e-2,
                                         batch_size = 700,
                                         alpha = 1,
                                         beta = 2*1e2,
                                         gfn_MC = 3,
                                         threshold_gfn = 3.5905,
                                         max_sparsity_gfn = 1000000,
                                         method = 'quadrature',
                                         verbose = True
                                         )

    recon = Reconstructor(recon_params, 'TV', TV_weight = 1)
    u = loadmat(os.path.join('images','u.mat'))['u']
    rec_image = loadmat(os.path.join('images','Im.mat'))['Im']
    im.update_image(rec_image) # Update the image
    recon.gfn(im, u)

    # plt.imshow(im.image)
    # plt.show()