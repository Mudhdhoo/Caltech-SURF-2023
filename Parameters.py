from dataclasses import dataclass
from tabnanny import verbose

@dataclass
class Segmentation_Params:
    """
    Dataclass for storing the segmentation parameters.
    
    ------------ Parameters ------------
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
    delta: int
    GL_epsilon: float
    steps: int
    margin_proportion: float
    maxiterations: int
    grad_Bhatt_MC: int
    Bhatt_MC: int
    sigma: float
    beta: float
    gamma: float
    momentum_u: float
    threshold_seg: float
    max_sparsity_seg: int
    batch_size: int
    method: str
    dirac: bool
    verbose: bool

@dataclass
class Reconstruction_Params:
    momentum_im: float
    sigma: float
    batch_size: int
    alpha: float
    beta: float
    gfn_MC: int
    threshold_gfn: float
    max_sparsity_gfn: float
    method: str
    verbose: bool
