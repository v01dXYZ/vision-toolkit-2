from scipy.stats import multivariate_normal
import numpy as np


cdef inline double _clamp_eps(double x, double eps) nogil:
    if x < eps:
        return eps
    return x


def evaluate_posterior_moments(double[:,:] a_s,
                               _means, _vars, pi, T):

    cdef int n = 0
    cdef int k = 0
    cdef int j = 0

    cdef int n_ = a_s.shape[0]
    cdef int d_ = a_s.shape[1]

    cdef int n_s = T.shape[0]
    cdef double[:,:] c_T = T

    cdef double eps = 1e-300

    # --- Emission probabilities (Python, SciPy) ---
    emm_prob = np.zeros((n_, n_s), dtype=np.float64)
    for k in range(n_s):
        # protection minimale
        if (not np.isfinite(_means[k]).all()) or (not np.isfinite(_vars[k]).all()):
            _means[k] = np.nan_to_num(_means[k], nan=0.0, posinf=0.0, neginf=0.0)
            _vars[k] = np.eye(d_, dtype=np.float64) * 2500.0

        # régularisation covariance
        _vars[k] = 0.5 * (_vars[k] + _vars[k].T)
        _vars[k] = _vars[k] + np.eye(d_, dtype=np.float64) * 1e-6

        emm_prob[:, k] = multivariate_normal.pdf(
            a_s, _means[k], _vars[k], allow_singular=True
        )

    cdef double[:,:] c_emm_prob = emm_prob

    # --- Forward / backward (C) ---
    cdef double[:,:] alpha = np.zeros((n_, n_s), dtype=np.float64)
    cdef double[:,:] beta  = np.zeros((n_, n_s), dtype=np.float64)
    cdef double[:]  norm_  = np.zeros(n_, dtype=np.float64)

    # init alpha
    for k in range(n_s):
        alpha[0, k] = pi[k] * c_emm_prob[0, k]

    norm_[0] = 0.0
    for k in range(n_s):
        norm_[0] += alpha[0, k]
    norm_[0] = _clamp_eps(norm_[0], eps)

    for k in range(n_s):
        alpha[0, k] /= norm_[0]

    # forward
    with nogil:
        for n in range(1, n_):
            for k in range(n_s):
                alpha[n, k] = 0.0
                for j in range(n_s):
                    alpha[n, k] += alpha[n-1, j] * c_T[j, k]
                alpha[n, k] *= c_emm_prob[n, k]

            norm_[n] = 0.0
            for k in range(n_s):
                norm_[n] += alpha[n, k]
            norm_[n] = _clamp_eps(norm_[n], eps)

            for k in range(n_s):
                alpha[n, k] /= norm_[n]

    # init beta (scaled version): beta[T-1,:] = 1
    for k in range(n_s):
        beta[n_-1, k] = 1.0

    # backward (scaled): divide by norm_[n+1]
    with nogil:
        for n in range(n_ - 2, -1, -1):
            for k in range(n_s):
                beta[n, k] = 0.0
                for j in range(n_s):
                    beta[n, k] += c_T[k, j] * c_emm_prob[n+1, j] * beta[n+1, j]
                beta[n, k] /= _clamp_eps(norm_[n+1], eps)

    # gamma
    gamma = np.asarray(alpha) * np.asarray(beta)
    gamma_sum = np.sum(gamma, axis=1).reshape((n_, 1))
    gamma_sum[gamma_sum < 1e-300] = 1e-300
    gamma /= gamma_sum

    # xi (stable)
    xi = np.zeros((n_, n_s, n_s), dtype=np.float64)
    for n in range(n_ - 1):
        denom = 0.0
        for k in range(n_s):
            for j in range(n_s):
                xi[n, k, j] = (alpha[n, k] *
                               T[k, j] *
                               emm_prob[n+1, j] *
                               beta[n+1, j])
                denom += xi[n, k, j]
        if denom < 1e-300:
            denom = 1e-300
        xi[n] /= denom

    return (gamma, xi)


def baum_welch(double[:,:] a_s, int n_s, int n_iters, _means):

    cdef int n = 0
    cdef int k = 0
    cdef int j = 0

    cdef int n_ = a_s.shape[0]
    cdef int d_ = a_s.shape[1]

    cdef double eps = 1e-300
    cdef double reg = 1e-6

    # --- init params ---
    pi = np.ones(n_s, dtype=np.float64) / float(n_s)
    T = np.ones((n_s, n_s), dtype=np.float64) / float(n_s)
    cdef double[:,:] c_T = T

    _vars = np.zeros((n_s, d_, d_), dtype=np.float64)
    for k in range(n_s):
        _vars[k] = np.eye(d_, dtype=np.float64) * 2500.0

    # emission probabilities
    emm_prob = np.zeros((n_, n_s), dtype=np.float64)
    for k in range(n_s):
        emm_prob[:, k] = multivariate_normal.pdf(
            a_s, _means[k], _vars[k], allow_singular=True
        )
    cdef double[:,:] c_emm_prob = emm_prob

    cdef double[:,:] alpha = np.zeros((n_, n_s), dtype=np.float64)
    cdef double[:,:] beta  = np.zeros((n_, n_s), dtype=np.float64)
    cdef double[:]  norm_  = np.zeros(n_, dtype=np.float64)

    iter_ = 0

    while iter_ < n_iters:

        # --- forward init ---
        for k in range(n_s):
            alpha[0, k] = pi[k] * c_emm_prob[0, k]

        norm_[0] = 0.0
        for k in range(n_s):
            norm_[0] += alpha[0, k]
        norm_[0] = _clamp_eps(norm_[0], eps)

        for k in range(n_s):
            alpha[0, k] /= norm_[0]

        # --- forward ---
        with nogil:
            for n in range(1, n_):
                for k in range(n_s):
                    alpha[n, k] = 0.0
                    for j in range(n_s):
                        alpha[n, k] += alpha[n-1, j] * c_T[j, k]
                    alpha[n, k] *= c_emm_prob[n, k]

                norm_[n] = 0.0
                for k in range(n_s):
                    norm_[n] += alpha[n, k]
                norm_[n] = _clamp_eps(norm_[n], eps)

                for k in range(n_s):
                    alpha[n, k] /= norm_[n]

        # --- backward init ---
        for k in range(n_s):
            beta[n_-1, k] = 1.0

        # --- backward (scaled) ---
        with nogil:
            for n in range(n_ - 2, -1, -1):
                for k in range(n_s):
                    beta[n, k] = 0.0
                    for j in range(n_s):
                        beta[n, k] += c_T[k, j] * c_emm_prob[n+1, j] * beta[n+1, j]
                    beta[n, k] /= _clamp_eps(norm_[n+1], eps)

        # --- gamma ---
        gamma = np.asarray(alpha) * np.asarray(beta)
        gamma_sum = np.sum(gamma, axis=1).reshape((n_, 1))
        gamma_sum[gamma_sum < 1e-300] = 1e-300
        gamma /= gamma_sum

        # --- xi (stable) ---
        xi = np.zeros((n_, n_s, n_s), dtype=np.float64)
        for n in range(n_ - 1):
            denom = 0.0
            for k in range(n_s):
                for j in range(n_s):
                    xi[n, k, j] = (alpha[n, k] *
                                   T[k, j] *
                                   emm_prob[n+1, j] *
                                   beta[n+1, j])
                    denom += xi[n, k, j]
            if denom < 1e-300:
                denom = 1e-300
            xi[n] /= denom

        # --- update pi ---
        pi = gamma[0].copy()
        pi_sum = pi.sum()
        if pi_sum < 1e-300:
            pi_sum = 1e-300
        pi /= pi_sum

        # --- update T ---
        for k in range(n_s):
            denom = np.sum(gamma[:-1, k])
            if denom < 1e-300:
                denom = 1e-300
            for j in range(n_s):
                T[k, j] = np.sum(xi[:-1, k, j]) / denom

        # renorm rows
        T_sum = np.sum(T, axis=1).reshape((n_s, 1))
        T_sum[T_sum < 1e-300] = 1e-300
        T /= T_sum
        c_T = T

        # --- update means ---
        a_np = np.asarray(a_s)
        for k in range(n_s):
            w = gamma[:, k]
            wsum = w.sum()
            if wsum < 1e-300:
                wsum = 1e-300
            _means[k] = (a_np.T @ w) / wsum

        # --- update covariances (regularized) ---
        for k in range(n_s):
            w = gamma[:, k]
            wsum = w.sum()
            if wsum < 1e-300:
                wsum = 1e-300

            cov = np.zeros((d_, d_), dtype=np.float64)
            for n in range(n_):
                dev = a_np[n] - _means[k]
                cov += w[n] * np.outer(dev, dev)

            cov /= wsum
            cov = 0.5 * (cov + cov.T)             # symmetrize
            cov += np.eye(d_, dtype=np.float64) * reg  # regularize

            # protection NaN/inf
            if (not np.isfinite(cov).all()) or (not np.isfinite(_means[k]).all()):
                _means[k] = np.nan_to_num(_means[k], nan=0.0, posinf=0.0, neginf=0.0)
                cov = np.eye(d_, dtype=np.float64) * 2500.0

            _vars[k] = cov

        # --- recompute emission probs ---
        for k in range(n_s):
            emm_prob[:, k] = multivariate_normal.pdf(
                a_s, _means[k], _vars[k], allow_singular=True
            )
        c_emm_prob = emm_prob

        # hard check (utile en debug)
        if not np.isfinite(emm_prob).all():
            raise ValueError("Emission probabilities contain NaN/inf (check means/covs/data).")

        iter_ += 1

    Z = np.argmax(gamma, axis=1)
    theta = (pi, T, _means, _vars)
    moments = (gamma, xi)
    return Z, theta, moments
