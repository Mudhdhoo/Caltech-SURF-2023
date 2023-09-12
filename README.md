# Joint Reconstruction-Segmentation Using the Bhattacharyya Coefficient
![Segmentation and reconstruction of a noisy heart image](/Results/github_image.png)
## Introduction
The practical application of image segmentation is often accompanied by the problem of noisy, distorted
images. Traditional sequential methods suffer from the loss of segmentation-relevant information after recon-
struction is performed, while contemporary learning-based methods are often hard to explain and do not give
an explicit reconstruction of the image. Joint reconstruction-segmentation is a recent approach to overcome these difficulties, performing both tasks at the same time, using one to guide the other. This project implements joint reconstruction-segmentation using the gradient flow stemming from the Bhattacharyya coefficient. Specifically, we model the segmentation as the variational problem of maximizing the distance between two probability distributions, one associated with the object to be segmented and the other with the image background. The reconstruction is modeled as the inverse problem of recovering the image from an indirect observation, under the constraint of respecting the previous segmentation.

## Usage

Install and activate the environment using conda: <br>
`conda env create -f environment.yml` <br>
`conda activate Joint_recon_seg`

To run the algorithm on the heart image: <br>
`python run.py` <br>
1. Load the image using the GUI.
1. Load the ground truth segmentation.
1. Draw an initial segmentation around the heart by dragging the mouse on the image.
1. Begin segmentation.

Additional sets of hyper-parameters can be added in [params.py](https://github.com/Mudhdhoo/Caltech-SURF-2023/blob/master/Code/params.py). Number of iterations, noise levels, printing and plotting options are available in [run.py](https://github.com/Mudhdhoo/Caltech-SURF-2023/blob/master/Code/run.py).

## Further Work
Some directions for future work on this project include:
- Implementing Gaussian quadrature sampling for RGB images.
- Code for GPU computing.
- Automatic parameter tuning.
- Nicer looking interface.