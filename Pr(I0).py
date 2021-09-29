#!/usr/bin/env python
# coding: utf-8

# In[154]:


from scipy.stats import e
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
from sklearn.preprocessing import normalize
import math


# In[155]:


#for f in os.listdir('nns'):
w_dict = {}
nns = np.load('nns.pkl')
for f in os.listdir('embds_pkls'):
    embds = np.load(os.path.join('embds_pkls', f))
    for k, v in embds.items():
        key = k.split('_')[0] + '.png'
        nns[key] = nns[key] / np.linalg.norm(nns[key])
        v = v / np.linalg.norm(v)
        d = np.dot(nns[key], v)
        d = np.arccos(d) / math.pi
        w_dict[k[:-4]] = 1-d
print(w_dict)


# In[156]:


class _PSD(object):
    def __init__(self, M, cond=None, rcond=None, lower=True,
                 check_finite=True, allow_singular=True):
        # Compute the symmetric eigendecomposition.
        # Note that eigh takes care of array conversion, chkfinite,
        # and assertion that the matrix is square.
        s, u = scipy.linalg.eigh(M, lower=lower, check_finite=check_finite)

        eps = _eigvalsh_to_eps(s, cond, rcond)
        if np.min(s) < -eps:
            raise ValueError('the input matrix must be positive semidefinite')
        d = s[s > eps]
        if len(d) < len(s) and not allow_singular:
            raise np.linalg.LinAlgError('singular matrix')
        s_pinv = _pinv_1d(s, eps)
        U = np.multiply(u, np.sqrt(s_pinv))

        # Initialize the eagerly precomputed attributes.
        self.rank = len(d)
        self.U = U
        self.log_pdet = np.sum(np.log(d))

        # Initialize an attribute to be lazily computed.
        self._pinv = None

    @property
    def pinv(self):
        if self._pinv is None:
            self._pinv = np.dot(self.U, self.U.T)
        return self._pinv

def _pinv_1d(v, eps=1e-5):
    return np.array([0 if abs(x) <= eps else 1/x for x in v], dtype=float)
    

def _eigvalsh_to_eps(spectrum, cond=None, rcond=None):
    """
    Determine which eigenvalues are "small" given the spectrum.

    This is for compatibility across various linear algebra functions
    that should agree about whether or not a Hermitian matrix is numerically
    singular and what is its numerical matrix rank.
    This is designed to be compatible with scipy.linalg.pinvh.

    Parameters
    ----------
    spectrum : 1d ndarray
        Array of eigenvalues of a Hermitian matrix.
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.

    Returns
    -------
    eps : float
        Magnitude cutoff for numerical negligibility.

    """
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = spectrum.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(spectrum))
    return eps

def _process_parameters(dim, mean, cov):
    """
    Infer dimensionality from mean or covariance matrix, ensure that
    mean and covariance are full vector resp. matrix.

    """

    # Try to infer dimensionality
    if dim is None:
        if mean is None:
            if cov is None:
                dim = 1
            else:
                cov = np.asarray(cov, dtype=float)
                if cov.ndim < 2:
                    dim = 1
                else:
                    dim = cov.shape[0]
        else:
            mean = np.asarray(mean, dtype=float)
            dim = mean.size
    else:
        if not np.isscalar(dim):
            raise ValueError("Dimension of random variable must be a scalar.")

    # Check input sizes and return full arrays for mean and cov if necessary
    if mean is None:
        mean = np.zeros(dim)
    mean = np.asarray(mean, dtype=float)

    if cov is None:
        cov = 1.0
    cov = np.asarray(cov, dtype=float)

    if dim == 1:
        mean.shape = (1,)
        cov.shape = (1, 1)

    if mean.ndim != 1 or mean.shape[0] != dim:
        raise ValueError("Array 'mean' must be a vector of length %d." % dim)
    if cov.ndim == 0:
        cov = cov * np.eye(dim)
    elif cov.ndim == 1:
        cov = np.diag(cov)
    elif cov.ndim == 2 and cov.shape != (dim, dim):
        rows, cols = cov.shape
        if rows != cols:
            msg = ("Array 'cov' must be square if it is two dimensional,"
                   " but cov.shape = %s." % str(cov.shape))
        else:
            msg = ("Dimension mismatch: array 'cov' is of shape %s,"
                   " but 'mean' is a vector of length %d.")
            msg = msg % (str(cov.shape), len(mean))
        raise ValueError(msg)
    elif cov.ndim > 2:
        raise ValueError("Array 'cov' must be at most two-dimensional,"
                         " but cov.ndim = %d" % cov.ndim)

    return dim, mean, cov

def _process_quantiles(x, dim):
    """
    Adjust quantiles array so that last axis labels the components of
    each data point.

    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 0:
        x = x[np.newaxis]
    elif x.ndim == 1:
        if dim == 1:
            x = x[:, np.newaxis]
        else:
            x = x[np.newaxis, :]

    return x


# In[203]:


def _logpdf(x, mean, prec_U, log_det_cov):
    dev = x - mean
    maha = np.sum(np.square(np.dot(dev, prec_U)), axis=-1)
    return -0.5 * (log_det_cov + maha)

def importance_weight(x, mean_p, cov_p, Pis, means_q, covs_q):
    def logpdf(x, mean=None, cov=1, allow_singular=False):
        dim, mean, cov = _process_parameters(None, mean, cov)
        x = _process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)
        return _logpdf(x, mean, psd.U, psd.log_pdet)
    w_inv = 0 
    for i in range(Pis.shape[0]):
        w_inv += Pis[i] * np.exp(logpdf(x, mean=means_q[i], cov=covs_q[i]) - logpdf(x, mean=mean_p, cov=cov_p))
        print(np.exp(logpdf(x, mean=means_q[i], cov=covs_q[i]) - logpdf(x, mean=mean_p, cov=cov_p)))
    return 1 / w_inv


# In[204]:


def buildPi(K):
    Pi = np.random.dirichlet(([1]*K), 1)[0]
    return Pi

def buildSubModels(mean, cov, K):
    means = np.random.multivariate_normal(mean, cov, K)
    covs = np.repeat(cov[np.newaxis, :, :], K, axis=0)    
    return means, covs

def get_qz(z, Pis, means, covs):
    qz = 0
    for i in range(Pis.shape[0]):
        qz += Pis[i] * multivariate_normal.pdf(z, mean=means[i], cov=covs[i])
        print(Pis[i],multivariate_normal.pdf(z, mean=means[i], cov=covs[i]))
    return qz

def drawFromGMM(Pis, means, covs, N):
    zs = np.empty([N, 512])
    ps = np.empty([N])
    # using numpy multinomial dist
    print(Pis)
    trials = np.random.multinomial(1, Pis, N).tolist()
    #print(trials)
    for i in range(N):
        k = np.nonzero(trials[i])[0].tolist()[0]
        print(Pis[k])
        z = np.random.multivariate_normal(means[k], covs[k], 1)
        zs[i,:] = z
        qz = get_qz(z, Pis, means, covs)
        pz = multivariate_normal.pdf(z, mean=np.zeros(512), cov=np.eye(512))
        #print(qz, pz)
        ps[i] = importance_weight(z, mean_p=np.zeros(512), cov_p=np.eye(512), Pis=Pis, means_q=means, covs_q=covs)
        #ps[i] = pz / qz
        print(pz, qz)
        print(pz / qz)
    return zs, ps


# In[205]:


K = 1
N = 1
Pis = buildPi(K)
print(Pis)


# In[206]:


mean, cov = np.zeros(512), np.eye(512)
means, covs = buildSubModels(mean, np.eye(512)*1, K)
#print(means.shape)
#print(covs.shape)
zs, ps = drawFromGMM(Pis, means, covs, N)
print(ps)


# In[151]:


K = 100
Pis = buildPi(K)
print(Pis)


# In[152]:


print(zs.shape)

