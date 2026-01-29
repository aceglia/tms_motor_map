import numpy as np
from scipy.stats import pearsonr

def min_map(grid_data_frame, part, euclid = True, c='grid'):
    if euclid:
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

def recompute_correlation(participants, grid_data_frame, muscle_list):
    for p in participants:
        cor_coef = [[np.nan for _ in range(len(muscle_list))]] + [
            [
                pearsonr(grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i+1].loc[grid_data_frame['muscle']==m]["zgf_list"].values[0].flatten(),
                        grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["zgf_list"].values[0].flatten())[0]
                for m in muscle_list
            ]
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
                    jsd=True, 
                    root=False,
                )
                for m in muscle_list
            ]
            for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "kl_divergence"] = sum(kl_div, [])
    return grid_data_frame