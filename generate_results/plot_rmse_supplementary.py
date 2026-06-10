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
                        # [34, 64, 94, 124, 154, 184],
                        # [34, 64, 94, 124, 154, 184]
                        [24, 44, 64, 84, 104, 124, 144, 164, 184],
                        [24, 44, 64, 84, 104, 124, 144, 164, 184]
                ] if 'sci' not in results_dir else [[49, 98, 147, 196],
                        # [34, 64, 94, 124, 154, 184],
                        # [34, 64, 94, 124, 154, 184]
                        [24, 44, 64, 84, 104, 124, 144, 164, 184],
                        [24, 44, 64, 84, 104, 124, 144, 164, 184]
                ]
    min_map_file = os.path.join(results_dir, "maps_min_map.csv")
    if not os.path.exists(min_map_file):
        raise RuntimeError('Min map file not found. Run generate_results/plot_wthin_method.py first.')
    min_maps = pd.read_csv(os.path.join(results_dir, "maps_min_map.csv"))
    min_maps = min_maps.loc[min_maps['muscle'].isin(muscle_list)]
    pd_reduced = pd.DataFrame()
    for i, participant in enumerate(min_maps['participant'].unique()):
        for j, muscle in enumerate(min_maps['muscle'].unique()):
            for c, cond in enumerate(['grid', 'pseudo']):
                min_map = min_maps.loc[(min_maps['participant'] == participant) & (min_maps['muscle'] == muscle) & (min_maps['condition'] == cond)]['min_map_number'].values[0]
                if not np.isfinite(min_map):
                    continue
                pd_tmp = data_frame.loc[(data_frame['participant'] == participant) & (data_frame['muscle'] == muscle) & (data_frame['condition'] == cond) & (data_frame['map_number'] == list_points_tot[c].index(min_map))]
                pd_reduced = pd.concat([pd_reduced, pd_tmp], ignore_index=True)

    # fig, axes = plt.subplots(2, 2, figsize=(15, 5))
    # fig, axes_min = plt.subplots(2, 2, figsize=(15, 5))
    # fig_tot = plt.figure(figsize=(15, 5), num='subplots')

    # axes_tot = []
    # locator_list = [5,5,350, 0.05]
    col = [[(0, 0), (0, 3)], [(0, 0), (0, 3)],
            [(1, 0), (1, 3)], [(1, 0), (1, 3)]]
    col_span = [[3, 1], [3, 1], [3, 1], [3, 1]]
    # fig, axs = plt.subplot_mosaic([['x_cog_line', 'x_cog_bar', 'y_cog_line', 'y_cog_bar'],
    #                                ['area_line', 'area_bar', 'volume_line', 'volume_bar']],
    #                               figsize=(6, 6),
    #                               width_ratios=(4, 1), height_ratios=(1, 4),
    #                               layout='constrained')
    fig_tot = plt.figure(layout='constrained', figsize=(10, 4))
    subfigs = fig_tot.subfigures(2, 2, wspace=0.05)
    comp = ['grid-grid', 'pseudo-pseudo', 'pseudo-grid']
    rating_title = ['CoG-ML (mm)', "CoG-AP (mm)", 'Area (%)', 'Volume']

    for r, rating in enumerate(['x_cog', 'y_cog', 'area', 'normalize_volume']):
        pd_icc = pd.DataFrame()
        pd_icc_min = pd.DataFrame()
        for c, cond in enumerate(['grid', 'pseudo', 'pseudo']):
            list_points = list_points_tot[c][:-1] if c != 2 else list_points_tot[c]
            for m, map in enumerate(list_points):
                pseudo = data_frame.loc[data_frame['condition'] == cond].loc[data_frame['map_number'] == list_points.index(map)]
                pseudo['ref'] = 0
                if m == 0:
                    min_map_tmp = pd_reduced.loc[pd_reduced['condition'] == cond]
                    min_map_tmp['ref'] = 0
                if c == 2:
                    ref = data_frame.loc[data_frame['condition'] == 'grid'].loc[data_frame['map_number'] == list_points_tot[0].index(list_points_tot[0][-1])]
                else:
                    ref = data_frame.loc[data_frame['condition'] == cond].loc[data_frame['map_number'] == list_points_tot[c].index(list_points_tot[c][-1])] 
                ref['ref'] = 1
                merged = pseudo.reset_index(drop=True).merge(on=['participant', 'muscle'], right=ref.reset_index(drop=True))
                concat_pd = pd.concat([ref, pseudo], ignore_index=True)
                for mus in merged['muscle'].unique():
                    if rating == 'cog':
                        # icc_df = pg.intraclass_corr(data=concat_pd.loc[concat_pd['muscle'] == mus], targets='participant', raters='ref', ratings='x_cog', nan_policy='omit').round(2)
                        x_ref = concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 1]['x_cog'].values
                        y_ref = concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 1]['y_cog'].values
                        x_pseudo = concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 0]['x_cog'].values
                        y_pseudo = concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 0]['y_cog'].values
                        rmse = np.nanmean(np.sqrt((x_pseudo - x_ref) ** 2 + (y_pseudo - y_ref) ** 2))
                    else:
                        # icc_df = pg.intraclass_corr(data=concat_pd.loc[concat_pd['muscle'] == mus], targets='participant', raters='ref', ratings=rating, nan_policy='omit').round(2)
                        rmse = np.sqrt(np.nanmean((concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 1][rating].values - concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 0][rating].values) ** 2))
                    if rating == 'area':
                        rmse = rmse * 100 / 3600
                    dict = {
                             'RMSE': [rmse],
                             'rating': [rating],
                             'map': [map],
                             'comp': [comp[c]],
                             'muscle': [mus]
                        }
                    icc_df = pd.DataFrame(dict)
                    pd_icc = pd.concat([pd_icc, icc_df], ignore_index=True)
                # rename column 'ICC' into 'ICC_all'
                # icc_df.rename(columns={'ICC': 'ICC_all'}, inplace=True)
                if m == 0:
                    concat_pd = pd.concat([ref, min_map_tmp], ignore_index=True)
                    # concat_pd = concat_pd.loc[concat_pd['muscle'] =='ext_comm']
                    for mus in concat_pd['muscle'].unique():
                        icc_df_min = pd.DataFrame()
                        if rating == 'cog':
                            # icc_df_min = pg.intraclass_corr(data=concat_pd.loc[concat_pd['muscle'] == mus], targets='participant', raters='ref', ratings='x_cog', nan_policy='omit').round(2)
                            x_ref = concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 1]['x_cog'].values
                            y_ref = concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 1]['y_cog'].values
                            x_pseudo = concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 0]['x_cog'].values
                            y_pseudo = concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 0]['y_cog'].values
                            rmse = np.nanmean(np.sqrt((x_pseudo - x_ref) ** 2 + (y_pseudo - y_ref) ** 2))
                        else:
                            # icc_df_min = pg.intraclass_corr(data=concat_pd.loc[concat_pd['muscle'] == mus], targets='participant', raters='ref', ratings=rating, nan_policy='omit').round(2)
                            rmse = np.sqrt(np.nanmean((concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 1][rating].values - concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 0][rating].values) ** 2))
                        # icc_df_min = pg.intraclass_corr(data=concat_pd.loc[concat_pd['muscle'] == mus], targets='participant', raters='ref', ratings=rating, nan_policy='omit').round(2)
                        # rmse = np.sqrt(np.nanmean((concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 1][rating].values - concat_pd.loc[concat_pd['muscle'] == mus].loc[concat_pd['ref'] == 0][rating].values) ** 2))
                        if rating == 'area':
                            rmse = rmse * 100 / 3600
                        dict = {
                             'RMSE': [rmse],
                             'rating': [rating],
                             'map': ['min'],
                             'comp': comp[c],
                             'muscle': mus
                        }
                        icc_df_min = pd.DataFrame(dict)
                        pd_icc_min = pd.concat([pd_icc_min, icc_df_min], ignore_index=True)
                # pd_icc = pd.concat([pd_icc, icc_df], ignore_index=True)
        # ax = axes.flatten()[r]
        loc = col[r]
        span = col_span[r]
        # icc3 = pd_icc
        # icc3_min = pd_icc_min
        icc3 = pd_icc # .loc[pd_icc['Type'] == 'ICC2']
        icc3_min = pd_icc_min # .loc[pd_icc_min['Type'] == 'ICC2']
        plt.rcParams["svg.fonttype"] = 'none'

        subfig = subfigs.flatten()[r]
        subfig.suptitle(rating_title[r], fontsize=16)
        gs = subfig.add_gridspec(1, 4, wspace=0.02)
        # axs[rating + '_line']
        # gs = gridspec.GridSpec(1, 4)
        # gs.update(left=0.05, right=0.48, wspace=0.05) if r % 2 == 0 else gs.update(left=0.55, right=0.98, wspace=0.05)
        axes_tmp = []
        for i in range(2):
            loc_tmp = loc[i]
            span_tmp = span[i]
            # axes_tot.append(plt.subplot2grid((2, 8), loc=loc_tmp, colspan=span_tmp, fig=fig_tot, rowspan=1))
            if i == 0:
                axes_tmp.append(fig_tot.add_subplot(gs[:, loc_tmp[1]:loc_tmp[1]+span_tmp]))
                ax = axes_tmp[i]
                sns.lineplot(x='map', y='RMSE', hue='comp', data=icc3, ax=ax, marker='o', palette="rocket",legend=r==0, hue_order=comp)
                ax.set_title('Full-size comparison', fontsize=12)
                if r==0:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, ['Grid-Grid', 'Pseudo-Pseudo', 'Pseudo-Grid'], title='', frameon=False)
            else:
                axes_tmp.append(fig_tot.add_subplot(gs[:, loc_tmp[1]:loc_tmp[1]+span_tmp], sharey=axes_tmp[i-1]))
                ax = axes_tmp[i]
                sns.barplot(x='comp', y='RMSE', data=icc3_min, ax=ax, palette="rocket", order=comp)
                ax.set_title('Tailored maps', fontsize=12)
                # ax.set_yticklabels([])
                ax.set_ylabel('')
            # ax.hlines(0.9, ax.get_xlim()[0], ax.get_xlim()[1], colors='gray', linestyles='dashed')
            # ax.set_ylim(0, 1 + 0.1)
            if r in [0, 2] and i == 0:
                ax.set_ylabel('RMSE')
            # elif r in [0, 2] and i == 1:
            #     ax.set_ylabel('')
            #     ax.set_yticklabels([])
            # elif r in [1, 3]:
            #     ax.set_ylabel('')
            #     ax.set_yticklabels([])
            if r in [2, 3] and i == 0:
                ax.set_xlabel('Stimulation number')
            elif r in [2, 3] and i == 1:
                ax.set_xlabel('')
                ax.set_xticklabels(['Grid-\nGrid', 'Pseudo-\nPseudo', 'Pseudo-\nGrid'], rotation=0)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])
    plt.savefig(os.path.join(results_dir, 'RMSE_sup.png'))
    plt.show()
    plt.close()
    print('figure saved to', os.path.join(results_dir, 'RMSE_sup.png'))

if __name__ == '__main__':
    seeds = [0]
    smooth_1 = [6]
    smooth_2 =  [6]
    all_folder = []
    for s in seeds:
        for s1 in smooth_1:
            for s2 in smooth_2:
                all_folder.append(rf"D:\Documents\Programmation\tms_motor_map\results\smooth_{s1}_{s2}_{s}_ransac")

    main(all_folder[0])
    # ctx = mp.get_context("spawn")
    # with ctx.Pool(processes=4) as pool:
    #     pool.map(main, all_folder)