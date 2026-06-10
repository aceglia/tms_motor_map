import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy import stats

def min_map(grid_data_frame, part, key='correlation_coefficient'):
    if key == 'kl_divergence':
        cor_coef = grid_data_frame.loc[grid_data_frame['participant'] == part].loc[grid_data_frame['kl_divergence'] <= 0.15]
    elif key == 'euclid_cog_error':
        cor_coef = grid_data_frame.loc[grid_data_frame['participant'] == part].loc[grid_data_frame['euclid_cog_error'] <= 3.6]
    else:
        cor_coef = grid_data_frame.loc[grid_data_frame['participant'] == part].loc[grid_data_frame['correlation_coefficient'] >= 0.9]
    
    map_list = [np.nan, np.nan, np.nan]
    if not cor_coef.empty:
        for m, muscle in enumerate(cor_coef['muscle'].unique()):
            pd_tmp_muscle = cor_coef.loc[cor_coef['muscle'] == muscle]
            min_map = pd_tmp_muscle['map_number'].min()
            map_list[m] = min_map
    else:
        return map_list
    return map_list

def cosine_similarity(ref, pseudo):
    cos_sim = np.dot(ref.flatten(), pseudo.flatten()) / (np.linalg.norm(ref.flatten()) * np.linalg.norm(pseudo.flatten()))
    return cos_sim


def recompute_correlation(participants, grid_data_frame, muscle_list):
    for p in participants:
        cor_coef = [[np.nan for _ in range(len(muscle_list))]] + [
            [
                pearsonr(grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i+1].loc[grid_data_frame['muscle']==m]["zgf_list"].values[0].flatten(),
                        grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["zgf_list"].values[0].flatten())[0]
                for m in muscle_list
            ]
            # [
            #     cosine_similarity(grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i+1].loc[grid_data_frame['muscle']==m]["zgf_list"].values[0],
            #             grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["zgf_list"].values[0])
            #     for m in muscle_list
            # ]
            # [
            #     ssim(grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i+1].loc[grid_data_frame['muscle']==m]["zgf_list"].values[0],
            #             grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["zgf_list"].values[0], 
            #             data_range=grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["zgf_list"].values[0].max() - grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["zgf_list"].values[0].min()
            #             )
            #     for m in muscle_list
            # ]

            for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "correlation_coefficient"] = sum(cor_coef, [])
    return grid_data_frame

def recompute_euclid_dist(participants, grid_data_frame, muscle_list):
    for p in participants:
        cog_err_eucl = [[np.nan for _ in range(len(muscle_list))]]
        cog_err_eucl = cog_err_eucl + [
            [
        np.linalg.norm(np.array([grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i+1].loc[grid_data_frame['muscle']==m]["x_cog"],
                                 grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i+1].loc[grid_data_frame['muscle']==m]["y_cog"]])
                                   - np.array([grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["x_cog"],
                                                grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["y_cog"]]), axis=0)[0]
            for m in muscle_list
            ]
        for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "euclid_cog_error"] = sum(cog_err_eucl, [])
    return grid_data_frame

def rmse(ref, pred):
    return np.sqrt(np.mean((ref - pred)**2))

def recompute_area_error(participants, grid_data_frame, muscle_list):
    for p in participants:
        cog_err_eucl = [[np.nan for _ in range(len(muscle_list))]]
        cog_err_eucl = cog_err_eucl + [
            [rmse(np.array(grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i+1].loc[grid_data_frame['muscle']==m]["area"]),
                            np.array(grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["area"]))
            for m in muscle_list
            ]
        for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "area_error"] = sum(cog_err_eucl, [])
    return grid_data_frame

def recompute_volume_error(participants, grid_data_frame, muscle_list):
    for p in participants:
        cog_err_eucl = [[np.nan for _ in range(len(muscle_list))]]
        cog_err_eucl = cog_err_eucl + [
            [rmse(np.array(grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i+1].loc[grid_data_frame['muscle']==m]["volume"]),
                            np.array(grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["volume"]))
            for m in muscle_list
            ]
        for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "volume_error"] = sum(cog_err_eucl, [])
    return grid_data_frame


def kl_divergence(img_new, img_old, eps=1e-12, log_fct="2", jsd=False, root=False):
    # return rel_entr(img_old.flatten(), img_new.flatten()).sum()
    img_old = img_old.flatten()
    img_new = img_new.flatten()
    if log_fct == "2":
        log_fct = np.log2
    elif log_fct == "10":
        log_fct = np.log10
    elif log_fct == "e":
        log_fct = np.log
    else:
        raise RuntimeError("Not recognized log fucntion")
    p = np.clip(img_new.astype(float), eps, 1) / np.sum(img_new.astype(float))
    q = np.clip(img_old.astype(float), eps, 1) / np.sum(img_old.astype(float))
    m = 0.5 * (p + q)
    if jsd:
        value = 0.5 * (np.sum(p * log_fct(p / m)) + np.sum(q * log_fct(q / m)))
        return np.sqrt(value) if root else value
    return np.sum(p * log_fct(p / q))



def recompute_kl_divergence(participants, grid_data_frame, muscle_list):
    for p in participants:
        kl_div = [[np.nan for _ in range(len(muscle_list))]]
        kl_div = kl_div + [
            [
                kl_divergence(
                    np.array(
                        grid_data_frame.loc[grid_data_frame["participant"] == p]
                        .loc[grid_data_frame["map_number"] == i]
                        .loc[grid_data_frame["muscle"] == m]["zgf_list"]
                    )[0],
                    np.array(
                        grid_data_frame.loc[grid_data_frame["participant"] == p]
                        .loc[grid_data_frame["map_number"] == i + 1]
                        .loc[grid_data_frame["muscle"] == m]["zgf_list"]
                    )[0],
                    log_fct="e",
                    jsd=False, 
                    root=False,
                )
                for m in muscle_list
            ]
            for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "kl_divergence"] = sum(kl_div, [])
    return grid_data_frame


def bland_altman(
    method1,
    method2,
    confidence=0.95,
    ax=None,
    title="Bland-Altman Plot",
):
    """
    Bland-Altman analysis with confidence intervals.

    Parameters
    ----------
    method1, method2 : array-like
        Measurements from the two methods
    confidence : float
        Confidence level (default = 0.95)
    ax : matplotlib axis
        Existing axis
    title : str
        Plot title

    Returns
    -------
    results : dict
        Bland-Altman statistics
    """

    method1 = np.asarray(method1)
    method2 = np.asarray(method2)

    if method1.shape != method2.shape:
        raise ValueError("Inputs must have the same shape")

    # Bland-Altman values
    mean = (method1 + method2) / 2
    diff = method1 - method2

    n = len(diff)

    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)

    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    # t critical value
    alpha = 1 - confidence
    tval = stats.t.ppf(1 - alpha / 2, n - 1)

    # CI for bias
    se_bias = sd / np.sqrt(n)

    bias_ci = (
        bias - tval * se_bias,
        bias + tval * se_bias,
    )

    # SE for limits of agreement
    se_loa = sd * np.sqrt(1 + (1.96**2) / (2 * n))

    loa_upper_ci = (
        loa_upper - tval * se_loa,
        loa_upper + tval * se_loa,
    )

    loa_lower_ci = (
        loa_lower - tval * se_loa,
        loa_lower + tval * se_loa,
    )

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(mean, diff)

    # Main lines
    ax.axhline(bias, linestyle='--', label='Bias')
    ax.axhline(loa_upper, linestyle=':', label='Upper LoA')
    ax.axhline(loa_lower, linestyle=':', label='Lower LoA')

    # CI shaded regions
    ax.fill_between(
        [np.min(mean), np.max(mean)],
        bias_ci[0],
        bias_ci[1],
        alpha=0.2,
    )

    ax.fill_between(
        [np.min(mean), np.max(mean)],
        loa_upper_ci[0],
        loa_upper_ci[1],
        alpha=0.15,
    )

    ax.fill_between(
        [np.min(mean), np.max(mean)],
        loa_lower_ci[0],
        loa_lower_ci[1],
        alpha=0.15,
    )

    ax.set_xlabel("Mean of methods")
    ax.set_ylabel("Difference between methods")
    ax.set_title(title)

    ax.legend()

    results = {
        "bias": bias,
        "bias_ci": bias_ci,
        "sd": sd,
        "loa_upper": loa_upper,
        "loa_upper_ci": loa_upper_ci,
        "loa_lower": loa_lower,
        "loa_lower_ci": loa_lower_ci,
    }

    return results