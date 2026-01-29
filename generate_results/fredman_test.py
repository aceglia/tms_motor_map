import os
import pandas as pd
from biosiglive import load
from plot_wthin_method import *

os.environ['R_HOME'] =  'C:\\Program Files\\R\\R-4.5.1'
from rpy2.robjects import r, pandas2ri  
import rpy2.robjects as ro

r('library(lme4)')
r('library(robustlmm)')
r('library(emmeans)')
r('library(emmeans)')

def converter(data, output='r'):
    if output == 'r':    
        with (ro.default_converter + pandas2ri.converter).context():
            data_out = ro.conversion.get_conversion().py2rpy(data)
    else:
        with (ro.default_converter + pandas2ri.converter).context():
            data_out = ro.conversion.get_conversion().rpy2py(data)
    return data_out

def compute_lmm_omnibus(pd_data, k, show_residuals=True, GLM=False):
    pd_data_r = converter(pd_data, output='r')
    r.assign("data_r", pd_data_r)
    r("data_r$muscle <- factor(data_r$muscle, levels=c('fdi', 'ext_com', 'sup'))")
    r("data_r$condition <- factor(data_r$condition, levels=c('grid', 'pseudo'))")
    r('contrasts(data_r$muscle) <- contr.sum')
    r('contrasts(data_r$condition) <- contr.sum')
    r('library(effectsize)')
    model_fct = k + " ~ 1 + muscle + condition + muscle:condition + ( 1 | participant )"
    r(f"model_r <- rlmer({model_fct}, data=data_r)")


def return_pairwise_comparison(pd_data, k, pairwise_comparison):

    pd_data = pd_data[['participant', 'condition', 'muscle', k]].dropna()
    compute_lmm_omnibus(pd_data, k, show_residuals=False)

    pd_r = pd.DataFrame()
    for comp in pairwise_comparison:
        comp_tmp = r(f'summary(contrast(emmeans(model_r, ~{comp}), "pairwise", adjust="Bonferroni"))')
        comp_tmp_pd = converter(comp_tmp, output='pd')
        pd_tmp = comp_tmp_pd.iloc[[0]]
        if comp_tmp_pd.shape[0]!=1:
            pd_tmp_bis = comp_tmp_pd.iloc[[-1]]
            pd_tmp = pd.concat([pd_tmp, pd_tmp_bis], ignore_index=True)
        pd_tmp['key'] = k
        pd_tmp['condition'] = pd_data.condition.unique().tolist()[0]
        pd_tmp.drop(columns=['estimate', 'df', 'SE'], inplace=True)
        # pd_tmp.drop(columns=['z.ratio'], inplace=True)
        pd_r = pd.concat([pd_r, pd_tmp], ignore_index=True)
    return pd_r


if __name__ == '__main__':
    data_dir = "smooth_10_5_50"
    file = os.path.join(data_dir, 'maps_characteristics.csv')
    muscle_to_keep = ['fdi', 'ext_comm', 'sup']
    # number = file.removesuffix('.csv').split('_')[-1]
    pd_tmp = pd.read_csv(file)
    pd_tmp.drop(columns='Unnamed: 0')
    data_frame = pd_tmp.loc[pd_tmp['muscle'].isin(muscle_to_keep)]
    data_maps = load(os.path.join(data_dir, f'maps_values.bio'), merge=False)
    data_frame_tot = pd.DataFrame()
    for i in range(len(data_maps)):
        n_mus = len(muscle_to_keep)
        data_frame_tmp = pd.DataFrame({'participant': data_maps[i]['participant'] * n_mus, 
                        'map_number': data_maps[i]['map_number'] * n_mus, 
                        'x_list': data_maps[i]['x_list'][:n_mus],
                        'y_list': data_maps[i]['y_list'][:n_mus],
                        'zgf_list': data_maps[i]['zgf_list'][:n_mus],
                        'condition': data_maps[i]['condition'] * n_mus,
                        'xgf_list': data_maps[i]['xgf_list'][:n_mus],
                        'ygf_list': data_maps[i]['ygf_list'][:n_mus],
                        'muscle': list(data_frame['muscle'][:n_mus]),
                        })
        if data_frame_tot.empty:
            data_frame_tot = data_frame_tmp
        else:
            data_frame_tot = pd.concat([data_frame_tot, data_frame_tmp])

    # merge dataframe
    data_frame = pd.merge(data_frame, data_frame_tot, on=['participant','map_number', 'condition', 'muscle'])
    data_frame = data_frame.loc[data_frame["participant"] != 'P006_TN']
    participants = data_frame["participant"].unique()
    cond = ['grid', 'pseudo']
    pd_min_map = pd.DataFrame()
    for c in cond:
        # multiply map number by 49 
        grid_data_frame = data_frame.loc[data_frame["condition"] == c].loc[data_frame["participant"].isin(participants)]

        # grid_data_frame = grid_data_frame.loc[grid_data_frame["muscle"].isin(list(grid_data_frame['muscle'][:3]))]
        muscle_list = list(data_frame['muscle'][:3])
        keys = ['euclid_cog_error', 'correlation_coefficient', 'area_error', 'volume_error']
        grid_data_frame = recompute_correlation(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_euclid_dist(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_area_error(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_volume_error(participants, grid_data_frame, muscle_list)

        pd_maps = pd.DataFrame()
        for part in participants:
            maps_euclid = min_map(grid_data_frame, part, euclid = True, c=c)
            maps_corr = min_map(grid_data_frame, part, euclid = False, c=c)
            pd_tmp = pd.DataFrame({'participant': [part] * 3, 'map_number_euclid': maps_euclid,
                                    'correlation_coefficient': maps_corr, 'muscle': ['fdi', 'ext_comm', 'sup']})
            if pd_maps.empty:
                pd_maps = pd_tmp
            else:
                pd_maps = pd.concat([pd_maps, pd_tmp])

        pd_maps['min_map_number'] = pd_maps[['map_number_euclid', 'correlation_coefficient']].max(axis=1) 
        if c == 'grid':
            pd_maps['min_map_number'] = pd_maps['min_map_number'] * 49
        else:
            list_number = [44, 64, 94, 124, 154, 184]
            pd_maps['min_map_number'] = pd_maps['min_map_number'].apply(lambda x: list_number[x - 1])

        for muscle in muscle_list:
            mean = pd_maps.loc[pd_maps['muscle'] == muscle]['min_map_number'].mean()
            std = pd_maps.loc[pd_maps['muscle'] == muscle]['min_map_number'].std()
            cv = ((std / mean) * 100).round(2)  
            print(f"cv for {c} condition and {muscle} muscle: {cv}")
        from scipy.stats import friedmanchisquare
        friedman_test = friedmanchisquare(
            pd_maps.loc[pd_maps['muscle'] == 'fdi']['min_map_number'],
              pd_maps.loc[pd_maps['muscle'] == 'ext_comm']['min_map_number'],
                pd_maps.loc[pd_maps['muscle'] =='sup']['min_map_number'])
        Q = friedman_test[0]
        p_value = friedman_test[1]
        W = Q / (len(participants) * (3- 1))
        # plt.figure(c)
        # sns.pointplot(x='muscle', y='min_map_number', data=pd_maps, hue='participant', 
        #               ci=95, join=True, legend=False, alpha=0.5, ax=plt.gca(), dodge=True)
        # sns.pointplot(x='muscle', y='min_map_number', data=pd_maps, ci=95, join=True, color='k', ax=plt.gca())
        pd_maps['condition'] = c
        pd_min_map = pd.concat([pd_min_map, pd_maps], ignore_index=True)

    parwise_comparison = ['condition', 'muscle', 'condition*muscle']
    pd_pairwise = return_pairwise_comparison(pd_min_map, 'min_map_number', parwise_comparison)
    print(pd_pairwise)

    pd_min_map.to_csv(os.path.join(data_dir, f"{c}_maps_number.csv"))
        # print(f"friedman test for {c} condition: {friedman_test}, W = {W}")
        # pd_maps.to_csv(os.path.join(data_dir, f"{c}_maps_number.csv"))





    #do friedman test with muscles as repeated measure for all participants
