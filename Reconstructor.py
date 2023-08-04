import numpy as np
import matplotlib.pyplot as plt 
from skimage.restoration import denoise_tv_chambolle
from Bhatt_Calculator import Bhatt_Calculator
from Image import Image
from Parameters import Reconstruction_Params


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
        
    def gfn(self, image: Image, u):
        M, N = image.image_size
        J = image.J
        n, q = J.shape
        u_vec = u.reshape(-1,1)
        Z0, W0 = self.Z0_calculator(self.gfn_MC, q)


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

        # Evaluate distributions P1,P2, output is n x MC
        indices = np.arange(0, n)
        P1_Z, P2_Z = self.Pcalculator_sparse2(J, u_vec, Z0, indices)

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
    y = im.y
    recon_params = Reconstruction_Params(momentum_im = 1,
                                         sigma = 1e-2,
                                         batch_size = 700,
                                         alpha = 1,
                                         beta = 2*1e2,
                                         gfn_MC = 100,
                                         threshold_gfn = 3.5905,
                                         max_sparsity_gfn = 10000000,
                                         method = 'random',
                                         verbose = True
                                         )

    recon = Reconstructor(recon_params, 'TV', TV_weight = 1)
    denoised_im = abs(recon.cheap_reconstruction(y))
    print(denoised_im)
    plt.imshow(denoised_im)
    plt.show()