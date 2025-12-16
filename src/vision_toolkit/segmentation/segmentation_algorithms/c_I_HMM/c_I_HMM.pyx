from scipy.stats import norm
from scipy.special import logsumexp
import numpy as np

from vision_toolkit.utils.segmentation_utils import interval_merging
from vision_toolkit.utils.segmentation_utils import centroids_from_ints


def process_IHMM(dict data_set, dict config):
    """
    I-HMM algorithm.
    """
    cdef double[:] a_s = data_set['absolute_speed']

    cdef double i_low_vel  = config['HMM_init_low_velocity']
    cdef double i_high_vel = config['HMM_init_high_velocity']
    cdef double i_var      = config['HMM_init_variance']

    cdef int n_iter = config['HMM_nb_iters']
    cdef int s_f    = config['sampling_frequency']

    theta = baum_welch(a_s, 2, n_iter, i_low_vel, i_high_vel, i_var)

    s_s = Viterbi(theta[1], theta[3], theta[0])

    fix_s = int(np.argmin(theta[2]))

    wi_fix = np.where(s_s[:-1] == fix_s)[0]
    wi_fix = np.array(sorted(set(list(wi_fix) + list(wi_fix + 1))))

    i_fix = np.array([False]*config['nb_samples'])
    i_fix[wi_fix] = True

    x_a = data_set['x_array']
    y_a = data_set['y_array']

    i_sac = (i_fix == False)
    wi_sac = np.where(i_sac == True)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size=np.ceil(config['min_sac_duration']*s_f),
    )

    if config['verbose']:
        print('   Saccadic intervals identified with minimum duration: {s_du} sec'
              .format(s_du=config['min_sac_duration']))

    # i_sac events not retained as intervals are relabeled as fix events
    i_fix = np.array([True]*config['nb_samples'])
    for s_int in s_ints:
        i_fix[s_int[0]: s_int[1]+1] = False

    # second pass to merge saccade separated by short fixations
    fix_dur_t = int(np.ceil(config['min_fix_duration']*s_f))
    for i in range(1, len(s_ints)):
        s_int = s_ints[i]
        o_s_int = s_ints[i-1]
        if s_int[0] - o_s_int[-1] < fix_dur_t:
            i_fix[o_s_int[-1]: s_int[0]+1] = False

    if config['verbose']:
        print('   Close saccadic intervals merged with duration threshold: {f_du} sec'
              .format(f_du=config['min_fix_duration']))

    # Recompute fixation intervals
    wi_fix = np.where(i_fix == True)[0]

    f_ints = interval_merging(
        wi_fix,
        min_int_size=np.ceil(config['min_fix_duration']*s_f),
        max_int_size=np.ceil(config['max_fix_duration']*s_f),
        status=data_set['status'],
        proportion=config['status_threshold']
    )

    # Compute fixation centroids
    ctrds = centroids_from_ints(f_ints, x_a, y_a)

    # Recompute saccadic intervals
    i_sac = (i_fix == False)
    wi_sac = np.where(i_sac == True)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size=np.ceil(config['min_sac_duration']*s_f),
        status=data_set['status'],
        proportion=config['status_threshold']
    )

    if config['verbose']:
        print('   Fixations ans saccades identified using availability status threshold: {s_th}'
              .format(s_th=config['status_threshold']))

    assert len(f_ints) == len(ctrds), "Interval set and centroid set have different lengths"

    i_lab = np.array([False]*config['nb_samples'])
    for f_int in f_ints:
        i_lab[f_int[0]: f_int[1]+1] = True
    for s_int in s_ints:
        i_lab[s_int[0]: s_int[1]+1] = True

    return dict({
        'is_labeled': i_lab,
        'fixation_intervals': f_ints,
        'saccade_intervals': s_ints,
        'centroids': ctrds,
    })


def baum_welch(double[:] a_s, double d_n_s, int n_iters,
               double i_low_vel, double i_high_vel,
               double i_var):

    cdef int n = 0
    cdef int k = 0
    cdef int l = 0
    cdef int j = 0

    cdef int n_ = len(a_s)
    cdef int n_s = int(d_n_s)

    # --- stability constants ---
    cdef double eps = 1e-300
    cdef double var_floor = 1e-6

    # --- sanitize observations (Python-side) ---
    a_np = np.asarray(a_s, dtype=np.float64)
    if not np.isfinite(a_np).all():
        a_np = np.nan_to_num(a_np, nan=0.0, posinf=0.0, neginf=0.0)

    # --- init params ---
    pi = np.ones(n_s, dtype=np.float64) / float(n_s)

    T = np.ones((n_s, n_s), dtype=np.float64) / float(n_s)
    cdef double[:,:] c_T = T

    _means = np.array([i_low_vel, i_high_vel], dtype=np.float64)
    _vars  = np.ones(n_s, dtype=np.float64) * float(i_var)
    for k in range(n_s):
        if _vars[k] < var_floor:
            _vars[k] = var_floor

    # --- emission probabilities (n_ x n_s) ---
    emm_prob = np.zeros((n_, n_s), dtype=np.float64)
    for k in range(n_s):
        emm_prob[:, k] = norm.pdf(a_np, _means[k], np.sqrt(_vars[k]))
    cdef double[:,:] c_emm_prob = emm_prob

    # --- buffers ---
    cdef double[:,:] alpha = np.zeros((n_, n_s), dtype=np.float64)
    cdef double[:,:] beta  = np.zeros((n_, n_s), dtype=np.float64)
    cdef double[:]  norm_  = np.zeros(n_, dtype=np.float64)

    # used in xi computation (avoid realloc inside loops)
    xi = np.zeros((n_-1, n_s, n_s), dtype=np.float64)

    iter_ = 0
    while iter_ < n_iters:

        # ---------- Forward init ----------
        for k in range(n_s):
            alpha[0, k] = pi[k] * c_emm_prob[0, k]

        norm_[0] = 0.0
        for k in range(n_s):
            norm_[0] += alpha[0, k]
        if norm_[0] < eps:
            norm_[0] = eps

        for k in range(n_s):
            alpha[0, k] /= norm_[0]

        # ---------- Forward ----------
        with nogil:
            for n in range(1, n_):
                for k in range(n_s):
                    alpha[n, k] = 0.0
                    for l in range(n_s):
                        alpha[n, k] += alpha[n-1, l] * c_T[l, k]
                    alpha[n, k] *= c_emm_prob[n, k]

                norm_[n] = 0.0
                for k in range(n_s):
                    norm_[n] += alpha[n, k]
                if norm_[n] < eps:
                    norm_[n] = eps

                for k in range(n_s):
                    alpha[n, k] /= norm_[n]

        # ---------- Backward init (FIX) ----------
        # scaled version: beta[T-1,:] = 1
        for k in range(n_s):
            beta[n_-1, k] = 1.0

        # ---------- Backward (scaled, FIX) ----------
        # divide by norm_[n+1] to be consistent with forward scaling
        with nogil:
            for n in range(n_ - 2, -1, -1):
                for k in range(n_s):
                    beta[n, k] = 0.0
                    for j in range(n_s):
                        beta[n, k] += c_T[k, j] * c_emm_prob[n+1, j] * beta[n+1, j]
                    beta[n, k] /= norm_[n+1]

        # ---------- gamma ----------
        gamma = np.asarray(alpha) * np.asarray(beta)
        gsum = np.sum(gamma, axis=1, keepdims=True)
        gsum[gsum < eps] = eps
        gamma /= gsum

        # ---------- xi (stable) ----------
        # xi[n,k,j] ∝ alpha[n,k] * T[k,j] * emm[n+1,j] * beta[n+1,j]
        for n in range(n_ - 1):
            denom = 0.0
            for k in range(n_s):
                for j in range(n_s):
                    xi[n, k, j] = alpha[n, k] * T[k, j] * emm_prob[n+1, j] * beta[n+1, j]
                    denom += xi[n, k, j]
            if denom < eps:
                denom = eps
            xi[n] /= denom

        # ---------- M-step ----------
        # pi
        pi = gamma[0].copy()
        pi_sum = pi.sum()
        if pi_sum < eps:
            pi_sum = eps
        pi /= pi_sum

        # T
        for k in range(n_s):
            denom = np.sum(gamma[:-1, k])
            if denom < eps:
                denom = eps
            for j in range(n_s):
                T[k, j] = np.sum(xi[:, k, j]) / denom

        T_sum = np.sum(T, axis=1, keepdims=True)
        T_sum[T_sum < eps] = eps
        T /= T_sum
        c_T = T

        # means / vars
        for k in range(n_s):
            w = gamma[:, k]
            wsum = w.sum()
            if wsum < eps:
                wsum = eps

            _means[k] = np.sum(w * a_np) / wsum

            v = np.sum(w * (a_np - _means[k])**2) / wsum
            if v < var_floor:
                v = var_floor
            _vars[k] = v

        # recompute emissions
        for k in range(n_s):
            emm_prob[:, k] = norm.pdf(a_np, _means[k], np.sqrt(_vars[k]))
        c_emm_prob = emm_prob

        # hard check (debug)
        if not np.isfinite(emm_prob).all():
            raise ValueError("Emission probabilities contain NaN/inf (check data/means/vars).")

        iter_ += 1

    # output in log for Viterbi
    log_emm_prob = np.zeros((n_, n_s), dtype=np.float64)
    for k in range(n_s):
        log_emm_prob[:, k] = norm.logpdf(a_np, _means[k], np.sqrt(_vars[k]))

    theta = (np.log(pi + 1e-9), np.log(T + 1e-9), _means, log_emm_prob)
    return theta


def Viterbi(log_T, log_emm_prob, log_pi):
    """
    Log implementation of the Viterbi algorithm.
    """
    n_s = log_T.shape[0]
    n_ = len(log_emm_prob)

    D = np.zeros((n_s, n_), dtype=np.float64)
    backtracking = np.zeros((n_s, n_ - 1), dtype=np.int32)
    S = np.zeros(n_, dtype=np.int32)

    D[:, 0] = log_pi + log_emm_prob[0, :]

    for n in range(1, n_):
        for i in range(n_s):
            score = D[:, n - 1] + log_T[:, i]
            backtracking[i, n - 1] = int(np.argmax(score))
            D[i, n] = np.max(score) + log_emm_prob[n, i]

    S[-1] = int(np.argmax(D[:, -1]))
    for n in range(n_ - 2, -1, -1):
        S[n] = backtracking[int(S[n + 1]), n]

    return S
