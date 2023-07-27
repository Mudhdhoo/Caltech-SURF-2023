delta = 8
GL_epsilon = 1e0
steps = 10
margin_proportion = 0.0225
maxiterations = 50
grad_Bhatt_MC = 10
Bhatt_MC = 50
sigma = 1e-2
#sigma = 1e-10
beta = 2*1e2
gamma = 2*2*1e-2
momentum_u = 1e-4
threshold_seg = 0.25    # TODO threshold_seg = min(threshold_seg,C);
max_sparsity_seg = 2000000 # TODO max_sparsity_seg = min(max_sparsity_seg,(M1*N1)^2);
batch_size = 700
method = 'random'
dirac = 0
verbose = True