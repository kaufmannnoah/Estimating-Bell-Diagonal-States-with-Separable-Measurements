import numpy as np
import qutip as qt
import scipy as sp
from functions.functions_paulibasis import *

########################################################
#ENSEMBLES

def create_ensemble(n, p, dim_n, rng, dim_k= None, type='ginibre'):
    match type:
        case 'ginibre': return sample_ginibre_ensemble(n, p, dim_n, rng, dim_k)
        case 'pure': return sample_pure_ensemble(n, p, dim_n, rng)
        case 'BDS': return sample_belldiag_ensemble(n, rng)
        case 'BDS_dirichlet': return sample_belldiag_ensemble(n, rng)

def sample_ginibre_ensemble(n, p, dim_n, rng= None, dim_k=None):
    # draw n states from the ginibre distribution (unbiased)
    # OUT: x_0: array of states sampled from Ginibre ensemble as Pauli vectors, w_0: uniform weights
    x_0 = np.zeros((n, dim_n**2))
    w_0 = np.ones(n)/n
    for i in range(n):
        dm = qt.rand_dm(dim_n, distribution= 'ginibre', rank= dim_k, seed= rng) # if dim_k == None (return full rank)
        x_0[i] = dm_to_bvector(dm.full(), p, dim_n) # calculate pauli representation
    return x_0, w_0
    
def sample_pure_ensemble(n, p, dim_n, rng= None):
    # draw n pure states (unbiased)
    # OUT: x_0: array of states sampled from Pure states as Pauli vectors, w_0: uniform weights
    x_0 = np.zeros((n, dim_n**2))
    w_0 = np.ones(n)/n
    for i in range(n):
        dm = qt.rand_dm(dim_n, distribution= 'pure', seed= rng)
        x_0[i] = dm_to_bvector(dm.full(), p, dim_n) # calculate pauli representation
    return x_0, w_0

def sample_belldiag_ensemble(n, rng= None):
    # draw n states that are diagonal in the bell basis (sample uniformly over a tetrahedron)
    # OUT: x_0: array of states sampled from diagonal states in the Bell basis as Pauli vectors, w_0: uniform weights
    rng = np.random.default_rng(rng)
    x_0 = np.zeros((n, 4**2))
    x_0[:, 0] = 1 / 4
    w_0 = np.ones(n)/n
    for i in range(n):
        x = rng.random(3) * 2 - 1
        while(point_in_tetrahedron(x) == 0):
            x = rng.random(3) * 2 - 1
        x_0[i, [5, 10, 15]] = x / 4 # 5, 10, 15 correspond to XX, YY and ZZ
    return x_0, w_0

def sample_belldiag_ensemble_dirichlet(n, rng= None):
    # draw n states that are diagonal in the bell basis (sample uniformly over a tetrahedron)
    # OUT: x_0: array of states sampled from diagonal states in the Bell basis as Pauli vectors, w_0: uniform weights
    rng = np.random.default_rng(rng)
    x_0 = np.zeros((n, 4**2))
    w_0 = np.ones(n)/n
    for i in range(n):
        r = rng.dirichlet(np.ones(4)) 
        x_0[i, :] = BDS_to_bvector(r) # 5, 10, 15 correspond to XX, YY and ZZ
    return x_0, w_0

########################################################
#MEASUREMENT BASIS

def create_POVM(M, p, dim, rng= None, type='rand', ret_basis= False):
    match type:
        case 'rand': return POVM_randbasis(M, p, dim, rng, ret_basis)
        case 'rand_bipartite': return POVM_randbasis_bipartite(M, p, dim, rng, ret_basis)
        case 'rand_separable': return POVM_randbasis_separable(M, p, dim, rng, ret_basis)
        case 'pauli': return POVM_paulibasis(M, p, dim, rng, ret_basis)
        case 'MUB4': return POVM_MUB4(M, p, rng, ret_basis)
        case 'pauli_BDS': return POVM_paulibasis_bds(M, p, dim, rng, ret_basis)
        case 'pauli_BDS_un': return POVM_paulibasis_bds_un(M, p, dim, rng, ret_basis)
        case 'bell': return POVM_bell(M, ret_basis)

def POVM_randbasis(M, p, dim, rng= None, ret_basis= False):
    # returns a complete set of orthogonal states, sampled according to the haar measure
    o = np.zeros((M, dim, dim**2))
    for m in range(M):
        u = qt.rand_unitary(dim, distribution= 'haar', seed= rng).full()
        o[m] = np.array([ket_to_bvector(u.T[i], p, dim) for i in range(dim)])
    if ret_basis: return o, 0
    else: return o

def POVM_randbasis_bipartite(M, p, dim, rng= None, ret_basis= False):
    # returns a complete set of orthogonal bipartite states sampled from the haar measure, the systems are split as equal as possible
    o = np.zeros((M, dim, dim**2))
    nq = int(np.log2(dim))
    if nq == 1: return POVM_randbasis(M, p, dim, rng, ret_basis)
    else:
        partition = [np.floor(nq / 2), np.ceil(nq / 2)]
        for m in range(M):
            u_i = [qt.rand_unitary(2**int(i), distribution= 'haar', seed= rng) for idi, i in enumerate(partition)]
            u = qt.tensor(u_i).full()
            o[m] = np.array([ket_to_bvector(u.T[i], p, dim) for i in range(dim)])
        if ret_basis: return o, 0
        else: return o

def POVM_randbasis_separable(M, p, dim, rng= None, ret_basis= False):
    # returns a complete set of orthogonal separable states sampled from the haar measure for each qubit
    o = np.zeros((M, dim, dim**2))
    nq = int(np.log2(dim))
    for m in range(M):
        u_i = [qt.rand_unitary(2, distribution= 'haar', seed= rng) for i in range(nq)]
        u = qt.tensor(u_i).full()
        o[m] = np.array([ket_to_bvector(u.T[i], p, dim) for i in range(dim)])
    if ret_basis: return o, 0
    else: return o

def POVM_paulibasis(M, p, dim, rng= None, ret_basis= False):
    # returns a complete set of orthogonal separable eigenstates of an n-qubit Pauli operator
    rng = np.random.default_rng(rng)
    o = np.zeros((M, dim, dim**2))
    nq = int(np.log2(dim))
    u_p = [qt.Qobj([[1, 0], [0, 1]]), 1/np.sqrt(2) * qt.Qobj(np.array([[1, 1], [1, -1]])), 1/np.sqrt(2) * qt.Qobj([[1, 1], [1.j, -1.j]]), ] #I, H, SH
    temp = rng.integers(3, size= (M, nq))
    for m in range(M):
        u_i = [u_p[temp[m, i]] for i in range(nq)]
        u = qt.tensor(u_i).full()
        o[m] = np.array([ket_to_bvector(u.T[i], p, dim) for i in range(dim)])
    if ret_basis: return o, temp
    else: return o

def POVM_paulibasis_bds(M, p, dim, rng= None, ret_basis= False):
    # returns a complete set of orthogonal separable eigenstates of an n-qubit Pauli operator
    rng = np.random.default_rng(rng)
    o = np.zeros((M, dim, dim**2))
    nq = int(np.log2(dim))
    u_p = [qt.Qobj([[1, 0], [0, 1]]), 1/np.sqrt(2) * qt.Qobj(np.array([[1, 1], [1, -1]])), 1/np.sqrt(2) * qt.Qobj([[1, 1], [1.j, -1.j]]), ] #I, H, SH
    #temp = rng.integers(3, size= M)
    temp = np.array([0, 1, 2] * int(M/3))
    for m in range(M):
        u_i = [u_p[temp[m]] for _ in range(nq)]
        u = qt.tensor(u_i).full()
        o[m] = np.array([ket_to_bvector(u.T[i], p, dim) for i in range(dim)])
    if ret_basis: return o, temp
    else: return o

def POVM_paulibasis_bds_un(M, p, dim, rng= None, ret_basis= False):
    # returns a complete set of orthogonal separable eigenstates of an n-qubit Pauli operator
    rng = np.random.default_rng(rng)
    o = np.zeros((M, dim, dim**2))
    nq = int(np.log2(dim))
    u_p = [qt.Qobj([[1, 0], [0, 1]]), 1/np.sqrt(2) * qt.Qobj(np.array([[1, 1], [1, -1]])), 1/np.sqrt(2) * qt.Qobj([[1, 1], [1.j, -1.j]]), ] #I, H, SH
    temp = rng.integers(3, size= M)
    #temp = np.array([0, 1, 2] * int(M/3))
    for m in range(M):
        u_i = [u_p[temp[m]] for _ in range(nq)]
        u = qt.tensor(u_i).full()
        o[m] = np.array([ket_to_bvector(u.T[i], p, dim) for i in range(dim)])
    if ret_basis: return o, temp
    else: return o

def POVM_bell(M, ret_basis= False):
    # returns bell state measurement
    o = np.zeros((M, 4, 16))
    for idi, i in enumerate(np.identity(4)):
        o[:, idi, :] = BDS_to_bvector(i)
    if ret_basis: return o, np.zeros(M)
    else: return o

def POVM_MUB4(M, p, rng= None, ret_basis= False):
    # teduces a random measurement with dim outcomes to 2 outcomes by averaging over the first dim/2 measurements and the last dim/2 meas
    rng = np.random.default_rng(rng) 
    o = np.zeros((M, 4, 16))
    #define 5 mutually unbiased bases
    M0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    M1 = 1/2 * np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])
    M2 = 1/2 * np.array([[1, -1.j, 1.j, 1], [1, -1.j, -1.j, -1], [1, 1.j, 1.j, -1], [1, 1.j, -1.j, 1]])
    M3 = 1/2 * np.array([[1, -1, -1.j, -1.j], [1, -1, 1.j, 1.j], [1, 1, 1.j, -1.j], [1, 1, -1.j, 1.j]])
    M4 = 1/2 * np.array([[1, -1.j, -1, -1.j], [1, -1.j, 1, 1.j], [1, 1.j, -1, 1.j], [1, 1.j, 1, -1.j]])
    MUB_4 = np.array([M0, M1, M2, M3, M4])
    temp = rng.integers(5, size= M)
    for i in range(M):
        o[i] = np.array([ket_to_bvector(j, p, 4) for j in MUB_4[temp[i]]])       
    if ret_basis: return o, temp
    else: return o

########################################################
# ESTIMATION

def prob_projectivemeas(oi, rho):
    # outcome probabilities of projective measurements specified in o, when measuring rho
    dim = np.sqrt(len(rho))
    prob = dim * np.array([np.sum(oo * rho) for oo in oi])
    prob[np.where(abs(prob) < 10**(-12))] = 0 # get rid of numerical instabilities causing smmall negative probabilities
    prob = prob / np.sum(prob) # renormalizing probabilities
    if abs(np.sum(prob)-1) > 0.1: print(np.sum(prob))
    return prob
                                          
def experiment(o, rho, rng= None):
    # measure rho in basis specified in POVM elements o
    rng = np.random.default_rng(rng)
    x = np.zeros(len(o))
    for ido, oi in enumerate(o):
        prob = prob_projectivemeas(oi, rho)
        x[ido] = rng.choice(np.arange(len(oi)), p= prob)
    return x
                                          
def likelihood(r, xi, oi):
    # calculate likelihood of measurement outcomes x for states in an array r 
    lh = np.array([np.sum(oi[int(xi)] * ri) for ri in r]) # proportional to probability (* dim is missing)
    return lh

def fidelity(a, b, p):
    # compute fidelity from density matrices in Pauli representation
    return qt.metrics.fidelity(bvector_to_dm(a, p), bvector_to_dm(b, p))**2

def HS_dist(a, b, p):
    # compute fidelity from density matrices in Pauli representation
    return qt.metrics.hilbert_dist(bvector_to_dm(a, p), bvector_to_dm(b, p))

########################################################
# BAYESIAN ESTIMATION

def bayes_update(r, w, x, o, n_active, threshold):
    # update weights according to likelihood and normalize    
    w_temp = w
    for i in range(len(x)):
        w_new = np.zeros(len(w_temp)) # needed such that weights below the threshold are 0
        w_new[n_active] = w_temp[n_active] * likelihood(r[n_active], x[i], o[i])
        w_new[n_active] = np.divide(w_new[n_active], np.sum(w_new[n_active]))
        w_temp = w_new
        n_active = n_active[np.where(w_new[n_active] > threshold)]
    return w_new
                                  
def pointestimate(x, w):
    # return point estimate of rho
    return np.average(x, axis=0, weights= w)

########################################################
# MLE

def loglikelihood_MLE(r, x):
    rho = BDS_to_bvector(r)
    LL = 0
    for x_i in x:
        a = np.sum(x_i * rho)
        if np.abs(a) < 10e-4: a = max(10e-6, a)
        LL += np.log(a)
    LL = LL - 0.001 * np.sum(np.abs(rho))
    return(-LL)

def MLE_BDS(x, o):
    xx = np.array([o[i, int(x[i])] for i in range(len(x))])
    trace_con = sp.optimize.LinearConstraint([[1, 1, 1, 1]], [1], [1])
    bnds = ((0, 1), (0, 1), (0, 1), (0, 1))
    r = sp.optimize.minimize(loglikelihood_MLE, x0= [1/3, 1/6, 1/12, 5/12], method= "SLSQP", args= (xx), bounds= bnds, constraints= [trace_con])
    return(BDS_to_bvector(r.x))

########################################################
# DIRECT RECONSTRUCTION

def direct_reconstruction_BDS(x, b, type= 'bell'):
    match type:
        case 'bell': return recon_from_bell(x)
        case 'pauli': return recon_from_paulibell(x, b)
        case 'pauli_BDS': return recon_from_pauli(x, b)
        case 'MUB4': return recon_from_MUB4(x, b)

def recon_from_bell(x):
    M = len(x)
    x_count = np.array([np.count_nonzero(x == i) for i in range(4)]) / M
    return BDS_to_bvector(x_count)

def recon_from_paulibell(k, b):
    x = np.copy(k)
    x[np.where(x == 3)] = 0
    x[np.where(x == 2)] = 1
    x[np.where(x == 1)] = 1
    zz = x[np.where(b==0)]
    xx = x[np.where(b==1)]
    yy = x[np.where(b==2)]
    if len(xx) != 0: x_count = np.array([np.count_nonzero(xx == 0), np.count_nonzero(xx == 1)]) / len(xx)
    else: x_count = np.ones(2) / 2
    if len(yy) != 0: y_count = np.array([np.count_nonzero(yy == 0), np.count_nonzero(yy == 1)]) / len(yy)
    else: y_count = np.ones(2) / 2
    if len(zz) != 0: z_count = np.array([np.count_nonzero(zz == 0), np.count_nonzero(zz == 1)]) / len(zz)
    else: z_count = np.ones(2) / 2
    p1 = (z_count[0] + x_count[0] + y_count[1] - 1) / 2
    p2 = (z_count[0] + x_count[1] + y_count[0] - 1) / 2
    p3 = (z_count[1] + x_count[0] + y_count[0] - 1) / 2
    p4 = (z_count[1] + x_count[1] + y_count[1] - 1) / 2
    return(BDS_to_bvector(np.array([p1, p2, p3, p4])))

def recon_from_pauli(k, b):
    ind_b = np.where(b[:, 0] == b[:, 1])
    x = k[ind_b]
    bb = b[ind_b, 0][0]
    x[np.where(x == 3)] = 0
    x[np.where(x == 2)] = 1
    x[np.where(x == 1)] = 1
    zz = x[np.where(bb == 0)]
    xx = x[np.where(bb == 1)]
    yy = x[np.where(bb == 2)]
    if len(xx) != 0: x_count = np.array([np.count_nonzero(xx == 0), np.count_nonzero(xx == 1)]) / len(xx)
    else: x_count = np.ones(2) / 2
    if len(yy) != 0: y_count = np.array([np.count_nonzero(yy == 0), np.count_nonzero(yy == 1)]) / len(yy)
    else: y_count = np.ones(2) / 2
    if len(zz) != 0: z_count = np.array([np.count_nonzero(zz == 0), np.count_nonzero(zz == 1)]) / len(zz)
    else: z_count = np.ones(2) / 2
    p1 = (z_count[0] + x_count[0] + y_count[1] - 1) / 2
    p2 = (z_count[0] + x_count[1] + y_count[0] - 1) / 2
    p3 = (z_count[1] + x_count[0] + y_count[0] - 1) / 2
    p4 = (z_count[1] + x_count[1] + y_count[1] - 1) / 2
    return(BDS_to_bvector(np.array([p1, p2, p3, p4])))

def recon_from_MUB4(k, b):
    x = np.copy(k)
    x[np.where(x == 3)] = 0
    x[np.where(x == 2)] = 1
    x[np.where(x == 1)] = 1
    x0 = x[np.where(b == 0)]
    x1 = x[np.where(b == 1)]
    x2 = x[np.where(b == 2)]
    if len(x0) != 0: x0_c = np.array([np.count_nonzero(x0 == 0), np.count_nonzero(x0 == 1)]) / len(x0)
    else: x0_c = np.ones(2) / 2
    if len(x1) != 0: x1_c = np.array([np.count_nonzero(x1 == 0), np.count_nonzero(x1 == 1)]) / len(x1)
    else: x1_c = np.ones(2) / 2
    if len(x2) != 0: x2_c = np.array([np.count_nonzero(x2 == 0), np.count_nonzero(x2 == 1)]) / len(x2)
    else: x2_c = np.ones(2) / 2
    p1 = (x0_c[0] + x1_c[0] + x2_c[0] - 1) / 2
    p2 = (x0_c[0] + x1_c[1] + x2_c[1] - 1) / 2
    p3 = (x0_c[1] + x1_c[0] + x2_c[1] - 1) / 2
    p4 = (x0_c[1] + x1_c[1] + x2_c[0] - 1) / 2
    return(BDS_to_bvector(np.array([p1, p2, p3, p4])))