import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import os

def main(results_dir):
    data_frame = pd.read_csv(os.path.join(results_dir, "maps_characteristics.csv"))
    data_frame = data_frame.loc[data_frame["participant"] != '006_TN']
    muscle_list = list(data_frame['muscle'][:3]) if not "sci" in results_dir else list(data_frame['muscle'][1:3])
    data_frame = data_frame.loc[data_frame['muscle'].isin(muscle_list)]
    list_points_tot = [[49, 98, 147, 196, 245],
                       [24, 44, 64, 84, 104, 124, 144, 164, 184],
                       [24, 44, 64, 84, 104, 124, 144, 164, 184],
                        # [34, 64, 94, 124, 154, 184],
                        # [34, 64, 94, 124, 154, 184]
                ] if 'sci' not in results_dir else [[49, 98, 147, 196],
                        # [34, 64, 94, 124, 154, 184],
                        # [34, 64, 94, 124, 154, 184]
                        [24, 44, 64, 84, 104, 124, 144, 164, 184],
                        [24, 44, 64, 84, 104, 124, 144, 164, 184]
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

    fig, axes = plt.subplots(1, 4, figsize=(10, 4))
    comp = ['grid-grid', 'pseudo-pseudo', 'pseudo-grid']
    rating_title = ['COG-ML', 'COG-AP', 'Area', 'Volume']
    for r, rating in enumerate(['x_cog', 'y_cog', 'area', 'volume']):
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
            for mus in concat_pd['muscle'].unique():
                icc_df_min = pd.DataFrame()
                # icc_df_min = pg.intraclass_corr(data=concat_pd.loc[concat_pd['muscle'] == mus], targets='participant', raters='ref', ratings=rating, nan_policy='omit').round(2)
                # compute rmse for ref 1 and ref 0
                rmse = np.sqrt(np.nanmean((concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 1][rating].values - concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 0][rating].values) ** 2))
            # icc_df_min = pg.intraclass_corr(data=concat_pd, targets='participant', raters='ref', ratings=rating, nan_policy='omit').round(2)
                icc_df_min['muscle'] = mus
                icc_df_min['rating'] = rating
                icc_df_min['comp'] = comp[c]
                icc_df_min['rmse'] = rmse
                pd_icc_min = pd.concat([pd_icc_min, icc_df_min], ignore_index=True)

        icc3_min = pd_icc_min.loc[pd_icc_min['Type'] == 'ICC2']
        # icc3_min.groupby(['comp', 'rating']).mean(numeric_only=True)
        ax = axes[r]
        ax.set_title(rating_title[r], fontsize=18)
        sns.barplot(x='comp', y='rmse', data=icc3_min, ax=ax, palette="rocket", alpha=0.5, order=comp)
        # ann = [ax.bar_label(cont, fontsize=14, color=cont.patches[0]._facecolor, padding=40, fmt='%.2f')  for cont in ax.containers[:3]]

        ax.hlines(0.75, ax.get_xlim()[0], ax.get_xlim()[1], colors='green', linestyles='dashed')
        ax.hlines(0.5, ax.get_xlim()[0], ax.get_xlim()[1], colors='red', linestyles='dashed')
        # ax.set_ylim(0, 1 + 0.1)
        if r in [0]:
            ax.set_ylabel('ICC', fontsize=16)
            ax.tick_params(axis='y', labelrotation=0, labelsize=14)
        elif r in [1, 2, 3]:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_xticklabels(['Grid-\nGrid', 'Pseudo-\nPseudo', 'Pseudo-\nGrid'], rotation=0, fontsize=14)
        sns.despine()
    # plt.savefig(os.path.join(results_dir, 'ICC.png'))
    plt.show()
    plt.close()
    print('figure saved to', os.path.join(results_dir, 'ICC.png'))

if __name__ == '__main__':
    seeds = [0]
    smooth_1 = [6]
    smooth_2 =  [6]
    all_folder = []
    for s in seeds:
        for s1 in smooth_1:
            for s2 in smooth_2:
                all_folder.append(rf"D:\Documents\Programmation\tms_motor_map\results\smooth_{s1}_{s2}_{s}_ransac_sci")

    main(all_folder[0])