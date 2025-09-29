import numpy as np
#from joblib import Parallel, delayed

from functions.functions_paulibasis import *
from functions.functions_estimation import *

########################################################
#PARAMETERS

#SYSTEM
n_q = np.array([2]) # number of Qubits - fixed in this implementation
dim = 2**n_q # dimension of Hilbert space
p = [create_pauli_basis(n_qi) for n_qi in n_q] # create Pauli basis

#ENSEMBLE
L_b = ['BDS_dirichlet'] # type of ensemble
L = 10000 # number of sampling points
rho_in_E = False # Flag whether the state to estimate is part of ensemble

#AVERAGES FOR BAYES RISK ESTIMATION
n_sample = 4000

#MEASUREMENTS
M_b = ['bell', 'pauli_BDS', 'MUB4', 'pauli', 'rand', 'rand_bipartite'] # type of measurement
M = [np.arange(3, 91, 3, dtype= int), np.arange(3, 91, 3, dtype= int), np.arange(5, 91, 5, dtype= int), np.arange(9, 91, 9, dtype= int), np.arange(3, 91, 3, dtype= int), np.arange(3, 91, 3, dtype= int)]

#METRIC
out_m = ['fidelity', 'HS', 'fid_MLE', 'HS_MLE', 'fid_recon', 'HS_recon'] # fixed!

#OUTCOME
out = np.zeros((len(out_m), len(L_b), len(dim), len(M_b), len(M[0]), n_sample))

#PARAMETERS COMPUTATION
cores = -2 # number of cores to Parallelize (-k := evey core expect k-cores)
threshold = 1 / (L**2) # threshold below which weights are cut off
n_active0 = np.arange(L)

#RANDOM SEED
seed = 20240726
rng = np.random.default_rng(seed)

########################################################
#ESTIMATION

def func(dim, p, m_basis, n_m, r, w0, rho_0, rng= None):    
    #Estimation
    O, b = create_POVM(n_m, p, dim, rng, type= m_basis, ret_basis= True)
    x = experiment(O, rho_0, rng)
    w = bayes_update(r, w0, x, O, n_active0, threshold)

    #MLE
    rho_mle = MLE_BDS(x, O)

    if m_basis == 'pauli_BDS': 
        rho_recon = recon_from_paulibell(x, b)
        fid_recon = np.round(fidelity(rho_0, rho_recon, p), decimals= 7)
        HS_recon = np.round(HS_dist(rho_0, rho_recon, p), decimals= 7)
    elif m_basis == 'bell': 
        rho_recon = recon_from_bell(x)
        fid_recon = np.round(fidelity(rho_0, rho_recon, p), decimals= 7)
        HS_recon = np.round(HS_dist(rho_0, rho_recon, p), decimals= 7)
    elif m_basis == 'pauli': 
        rho_recon = recon_from_pauli(x, b)
        fid_recon = np.round(fidelity(rho_0, rho_recon, p), decimals= 7)
        HS_recon = np.round(HS_dist(rho_0, rho_recon, p), decimals= 7)
    elif m_basis == 'MUB4': 
        rho_recon = recon_from_MUB4(x, b)
        fid_recon = np.round(fidelity(rho_0, rho_recon, p), decimals= 7)
        HS_recon = np.round(HS_dist(rho_0, rho_recon, p), decimals= 7)
    else:
        fid_recon = 10
        HS_recon = 10

    #Output
    rho_est = pointestimate(r, w)
    fid = np.round(fidelity(rho_0, rho_est, p), decimals= 7)
    HS = np.round(HS_dist(rho_0, rho_est, p), decimals= 7)
    fid_mle = np.round(fidelity(rho_0, rho_mle, p), decimals= 7)
    HS_mle = np.round(HS_dist(rho_0, rho_mle, p), decimals= 7)

    return fid, HS, fid_mle, HS_mle, fid_recon, HS_recon

########################################################
#MAIN

#Ensemble Types
for in_lb, lb_i in enumerate(L_b):

    #Dimensions
    for in_d, d_i in enumerate(dim):
        r, w0 = create_ensemble(L, p[in_d], d_i, rng, type= lb_i)
        if rho_in_E: rho_0 = [r[rng.integers(L)] for _ in range(n_sample)]
        else: rho_0 = [create_ensemble(1, p[in_d], d_i, rng, type= lb_i)[0][0] for _ in range(n_sample)]

        #Measurement Basis
        for in_mb, mb_i in enumerate(M_b):
            np.save(mb_i, np.ones(1))

            #Number of Measurements
            for in_m, m_i in enumerate(M[in_mb]):
                #Spawn Pseudo Random Number Generators for Paralelization
                #child_rngs = rng.spawn(n_sample)
                #out[:, in_lb, in_d, in_mb, in_m, :] = np.array(Parallel(n_jobs=cores)(delayed(func)(d_i, p[in_d], mb_i, m_i, r, w0, rho_0[k], child_rngs[k]) for k in range(n_sample))).T
                for k in range(n_sample):
                    out[:, in_lb, in_d, in_mb, in_m, k] = np.array(func(d_i, p[in_d], mb_i, m_i, r, w0, rho_0[k], rng))

np.save("data_figure_4-5", out)