import numpy as np
import xarray as xr
import pytensor.tensor as pt
import arviz as az
import seaborn as sns

def prepare_vectorized_bandit_data(df):
    """
    Converts the 1D trial dataframe into 2D/3D subject-batched arrays of shape (max_t, n_sub)
    and one-hot stimulus masks of shape (max_t, n_sub, 4) so pytensor.scan only loops over
    max_t (~98 trials) across all n_sub subjects in parallel without set_subtensor indexing.
    """
    df = df.reset_index(drop=True).copy()
    unique_subs = df['Sub'].unique()
    sub_idx_map = {sub: i for i, sub in enumerate(unique_subs)}
    df['SubIdx'] = df['Sub'].map(sub_idx_map)
    n_sub = len(unique_subs)
    df['TrialIdxSub'] = df.groupby('SubIdx').cumcount()
    max_t = int(df['TrialIdxSub'].max() + 1)

    t_idx = df['TrialIdxSub'].values
    s_idx = df['SubIdx'].values

    left_mat = np.zeros((max_t, n_sub), dtype='int64')
    right_mat = np.zeros((max_t, n_sub), dtype='int64')
    chosen_mat = np.zeros((max_t, n_sub), dtype='int64')
    rew_mat = np.zeros((max_t, n_sub), dtype='float64')
    win_mat = np.zeros((max_t, n_sub), dtype='int64')
    valid_mat = np.zeros((max_t, n_sub), dtype='float64')

    left_mat[t_idx, s_idx] = df['LeftID_int'].values
    right_mat[t_idx, s_idx] = df['RightID_int'].values
    chosen_mat[t_idx, s_idx] = df['ChosenID_int'].values
    rew_mat[t_idx, s_idx] = df['RewardScaled'].values
    if 'TrialType' in df.columns:
        win_mat[t_idx, s_idx] = (df['TrialType'] == 'W').astype('int64').values
    valid_mat[t_idx, s_idx] = 1.0

    eye4 = np.eye(4, dtype='float64')
    left_oh = eye4[left_mat]                                  # (max_t, n_sub, 4)
    right_oh = eye4[right_mat]                                # (max_t, n_sub, 4)
    chosen_oh = eye4[chosen_mat] * valid_mat[:, :, None]      # (max_t, n_sub, 4), 0 on padded trials

    return {
        'df': df,
        'unique_subs': unique_subs,
        'n_sub': n_sub,
        'max_t': max_t,
        't_idx': t_idx,
        's_idx': s_idx,
        'left_oh': left_oh,
        'right_oh': right_oh,
        'chosen_oh': chosen_oh,
        'rew_mat': rew_mat,
        'win_mat': win_mat,
        'valid_mat': valid_mat,
    }

def step_m1_vec(left_m, right_m, chosen_m, rew_s, Q_t, alpha):
    """Vectorized Single-Alpha step across all subjects simultaneously."""
    q_left = (Q_t * left_m).sum(axis=-1)
    q_right = (Q_t * right_m).sum(axis=-1)
    q_chosen = (Q_t * chosen_m).sum(axis=-1)
    PE = rew_s - q_chosen
    Q_next = Q_t + chosen_m * (alpha * PE)[:, None]
    return Q_next, q_left, q_right, Q_t

def step_m2_vec(left_m, right_m, chosen_m, rew_s, Q_t, alpha_pos, alpha_neg):
    """Vectorized Valence (Pos/Neg PE) step across all subjects simultaneously."""
    q_left = (Q_t * left_m).sum(axis=-1)
    q_right = (Q_t * right_m).sum(axis=-1)
    q_chosen = (Q_t * chosen_m).sum(axis=-1)
    PE = rew_s - q_chosen
    alpha_val = pt.switch(PE > 0, alpha_pos, alpha_neg)
    Q_next = Q_t + chosen_m * (alpha_val * PE)[:, None]
    return Q_next, q_left, q_right, Q_t

def step_m3_vec(left_m, right_m, chosen_m, rew_s, is_win_s, Q_t, alpha_win, alpha_lose):
    """Vectorized Task-Domain (Win/Lose) step across all subjects simultaneously."""
    q_left = (Q_t * left_m).sum(axis=-1)
    q_right = (Q_t * right_m).sum(axis=-1)
    q_chosen = (Q_t * chosen_m).sum(axis=-1)
    PE = rew_s - q_chosen
    alpha_val = pt.switch(is_win_s, alpha_win, alpha_lose)
    Q_next = Q_t + chosen_m * (alpha_val * PE)[:, None]
    return Q_next, q_left, q_right, Q_t

def step_m4_vec(left_m, right_m, chosen_m, rew_s, is_win_s, Q_t,
                alpha_win_pos, alpha_win_neg, alpha_lose_pos, alpha_lose_neg):
    """Vectorized Task x Valence (4 learning rates) step across all subjects simultaneously."""
    q_left = (Q_t * left_m).sum(axis=-1)
    q_right = (Q_t * right_m).sum(axis=-1)
    q_chosen = (Q_t * chosen_m).sum(axis=-1)
    PE = rew_s - q_chosen
    alpha_win_val = pt.switch(PE > 0, alpha_win_pos, alpha_win_neg)
    alpha_lose_val = pt.switch(PE > 0, alpha_lose_pos, alpha_lose_neg)
    alpha_val = pt.switch(is_win_s, alpha_win_val, alpha_lose_val)
    Q_next = Q_t + chosen_m * (alpha_val * PE)[:, None]
    return Q_next, q_left, q_right, Q_t

def step_kalman_vec(left_m, right_m, chosen_m, rew_s, valid_s, mu_t, sigma2_t, Q_sub, R_sub):
    """Vectorized Kalman Filter step across all subjects simultaneously."""
    sigma2_pred = sigma2_t + (Q_sub * valid_s)[:, None]
    q_left = (mu_t * left_m).sum(axis=-1)
    q_right = (mu_t * right_m).sum(axis=-1)
    mu_chosen = (mu_t * chosen_m).sum(axis=-1)
    sigma2_chosen = (sigma2_pred * chosen_m).sum(axis=-1)
    pe = rew_s - mu_chosen
    K = sigma2_chosen / (sigma2_chosen + R_sub)
    mu_next = mu_t + chosen_m * (K * pe)[:, None]
    sigma2_next = sigma2_pred - chosen_m * (K * sigma2_chosen)[:, None]
    return mu_next, sigma2_next, q_left, q_right

def add_bernoulli_ll(dt, y_obs, obs_name='choice_obs', p_name='p_right'):
    """Adds pointwise Bernoulli log-likelihood to a PyMC 6 DataTree (or PyMC 5 InferenceData)."""
    p = dt.posterior[p_name]
    y = xr.DataArray(y_obs, dims=[p.dims[-1]])
    ll = (np.log(p) * y + np.log(1.0 - p) * (1.0 - y)).transpose("chain", "draw", ...)
    ll = ll + 1e-12 * np.random.default_rng(42).standard_normal(ll.shape)
    if hasattr(dt, 'add_groups'):
        dt.add_groups(log_likelihood={obs_name: ll})
    else:
        dt['log_likelihood'] = xr.Dataset({obs_name: ll})

def get_hdi_bounds(dt, var_name, prob=0.89):
    """Extracts (low, high) HDI bounds compatibility helper for ArviZ 1.1+ DataTree and ArviZ 0.x."""
    samples = dt.posterior[var_name].values.reshape(-1)
    q_low = (1.0 - prob) / 2.0
    try:
        res = az.hdi(dt, var_names=[var_name], prob=prob)
        vals = res.ds[var_name].values if hasattr(res, 'ds') else res[var_name].values
        return float(vals[0]), float(vals[1])
    except Exception:
        return float(np.quantile(samples, q_low)), float(np.quantile(samples, 1.0 - q_low))

def plot_posterior_hdi(dt, var_name, prob=0.89, ax=None, color="teal", title=None, ref_val=0.0):
    """Clean KDE posterior + HDI bar plot compatible with ArviZ 1.1+ DataTree."""
    samples = dt.posterior[var_name].values.flatten()
    low, high = get_hdi_bounds(dt, var_name, prob=prob)
    mean_val = float(samples.mean())
    sns.histplot(samples, kde=True, stat="density", color=color, alpha=0.35, ax=ax)
    y_min, y_max = ax.get_ylim()
    ax.hlines(y=y_max * 0.05, xmin=low, xmax=high, color='black', linewidth=4,
              label=f"{int(prob*100)}% HDI [{low:.3f}, {high:.3f}]")
    ax.axvline(mean_val, color=color, linestyle='-', linewidth=2, label=f"Mean = {mean_val:.3f}")
    if ref_val is not None:
        ax.axvline(ref_val, color='red', linestyle='--', alpha=0.6, label=f"Ref = {ref_val}")
    if title:
        ax.set_title(title)
    ax.set_xlabel(var_name)
    ax.legend(loc='upper right', fontsize=9)
    return low, high

# Legacy 1D step functions for backward compatibility
def update_q_hierarchical(left, right, chosen, rew, is_first, sub, Q_t, alpha):
    Q_t_reset_row = pt.switch(is_first, pt.zeros(4), Q_t[sub])
    Q_t_reset = pt.set_subtensor(Q_t[sub], Q_t_reset_row)
    q_left = Q_t_reset[sub, left]
    q_right = Q_t_reset[sub, right]
    PE = rew - Q_t_reset[sub, chosen]
    Q_t_next = pt.set_subtensor(Q_t_reset[sub, chosen], Q_t_reset[sub, chosen] + alpha[sub] * PE)
    return Q_t_next, q_left, q_right, Q_t_reset_row

def update_m2_hierarchical(left, right, chosen, rew, is_first, sub, Q_t, alpha_pos, alpha_neg):
    Q_t_reset_row = pt.switch(is_first, pt.zeros(4), Q_t[sub])
    Q_t_reset = pt.set_subtensor(Q_t[sub], Q_t_reset_row)
    q_left = Q_t_reset[sub, left]
    q_right = Q_t_reset[sub, right]
    PE = rew - Q_t_reset[sub, chosen]
    alpha_val = pt.switch(PE > 0, alpha_pos[sub], alpha_neg[sub])
    Q_t_next = pt.set_subtensor(Q_t_reset[sub, chosen], Q_t_reset[sub, chosen] + alpha_val * PE)
    return Q_t_next, q_left, q_right, Q_t_reset_row

def update_m3_hierarchical(left, right, chosen, rew, is_first, is_win_trial, sub, Q_t, alpha_win, alpha_lose):
    Q_t_reset_row = pt.switch(is_first, pt.zeros(4), Q_t[sub])
    Q_t_reset = pt.set_subtensor(Q_t[sub], Q_t_reset_row)
    q_left = Q_t_reset[sub, left]
    q_right = Q_t_reset[sub, right]
    PE = rew - Q_t_reset[sub, chosen]
    alpha_val = pt.switch(is_win_trial, alpha_win[sub], alpha_lose[sub])
    Q_t_next = pt.set_subtensor(Q_t_reset[sub, chosen], Q_t_reset[sub, chosen] + alpha_val * PE)
    return Q_t_next, q_left, q_right, Q_t_reset_row

def update_m4_hierarchical(left, right, chosen, rew, is_first, is_win_trial, sub, Q_t,
                           alpha_win_pos, alpha_win_neg, alpha_lose_pos, alpha_lose_neg):
    Q_t_reset_row = pt.switch(is_first, pt.zeros(4), Q_t[sub])
    Q_t_reset = pt.set_subtensor(Q_t[sub], Q_t_reset_row)
    q_left = Q_t_reset[sub, left]
    q_right = Q_t_reset[sub, right]
    PE = rew - Q_t_reset[sub, chosen]
    alpha_win_val = pt.switch(PE > 0, alpha_win_pos[sub], alpha_win_neg[sub])
    alpha_lose_val = pt.switch(PE > 0, alpha_lose_pos[sub], alpha_lose_neg[sub])
    alpha_val = pt.switch(is_win_trial, alpha_win_val, alpha_lose_val)
    Q_t_next = pt.set_subtensor(Q_t_reset[sub, chosen], Q_t_reset[sub, chosen] + alpha_val * PE)
    return Q_t_next, q_left, q_right, Q_t_reset_row
