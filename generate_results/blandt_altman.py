from unittest import result

from utils import bland_altman
import numpy as np
import pandas as pd
import os
if __name__ == '__main__':
    results_dir = r"D:\Documents\Programmation\tms_motor_map\results\smooth_6_6_0_ransac"
    data_frame = pd.read_csv(os.path.join(results_dir, "maps_characteristics.csv"))
    data_frame = data_frame.loc[data_frame["participant"] != '006_TN']
    muscle_list = list(data_frame['muscle'][:3])
    data_frame = data_frame.loc[data_frame['muscle'].isin(muscle_list)]
    list_points_tot = [[49, 98, 147, 196, 245],
                       [24, 44, 64, 84, 104, 124, 144, 164, 184],
                       [24, 44, 64, 84, 104, 124, 144, 164, 184],
                        # [34, 64, 94, 124, 154, 184],
                        # [34, 64, 94, 124, 154, 184]
                ] 
    min_maps = pd.read_csv(os.path.join(results_dir, "maps_min_map.csv"))
    pd_reduced = pd.DataFrame()
    for i, participant in enumerate(min_maps['participant'].unique()):
        for j, muscle in enumerate(min_maps['muscle'].unique()):
            for c, cond in enumerate(['grid', 'pseudo']):
                min_map = min_maps.loc[(min_maps['participant'] == participant) & (min_maps['muscle'] == muscle) & (min_maps['condition'] == cond)]['min_map_number'].values[0]
                if not np.isfinite(min_map):
                    continue
                pd_tmp = data_frame.loc[(data_frame['participant'] == participant) & (data_frame['muscle'] == muscle) & (data_frame['condition'] == cond) & (data_frame['map_number'] == list_points_tot[c].index(min_map))]
                pd_reduced = pd.concat([pd_reduced, pd_tmp], ignore_index=True)

    # fig, axes = plt.subplots(1, 4, figsize=(10, 4))
    import matplotlib.pyplot as plt
    comp = ['grid-grid', 'pseudo-pseudo', 'pseudo-grid']
    rating_title = ['X-COG', 'Y-COG', 'Area', 'Volume']
    for r, rating in enumerate(['x_cog', 'y_cog', 'area', 'volume']):
        plt.figure(figsize=(5, 5), num=f"{comp[c]} - {rating_title[r]}")
        pd_icc_min = pd.DataFrame()
        for c, cond in enumerate(['grid', 'pseudo', 'pseudo']):
            min_map_tmp = pd_reduced.loc[pd_reduced['condition'] == cond]
            min_map_tmp['ref'] = 0
            if c == 2:
                ref = pd_reduced.loc[pd_reduced['condition'] == 'grid']
                # ref = data_frame.loc[data_frame['condition'] == 'grid'].loc[data_frame['map_number'] == list_points_tot[0].index(list_points_tot[0][-1])]
            else:
                ref = data_frame.loc[data_frame['condition'] == cond].loc[data_frame['map_number'] == list_points_tot[c].index(list_points_tot[c][-1])] 
            ref['ref'] = 1
            concat_pd = pd.concat([ref, min_map_tmp], ignore_index=True)
            bland_altman(ref[rating], min_map_tmp[rating], title=f"{comp[c]} - {rating_title[r]}")
            for mus in concat_pd['muscle'].unique():
                icc_df_min = pg.intraclass_corr(data=concat_pd.loc[concat_pd['muscle'] == mus], targets='participant', raters='ref', ratings=rating, nan_policy='omit').round(2)
                icc_df_min['muscle'] = mus
                icc_df_min['rating'] = rating
                icc_df_min['comp'] = comp[c]
                pd_icc_min = pd.concat([pd_icc_min, icc_df_min], ignore_index=True)