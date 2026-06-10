import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from biosiglive import load
from scipy.stats import pearsonr
import numpy as np

import os

def return_r2(ref, pseudo):
    r2 = pearsonr(ref, pseudo)[0]
    return r2

def main(results_dir):
    data_maps = load(os.path.join(results_dir, "maps_values.bio"), merge=False)
    data_frame_tot = pd.DataFrame()
    for i in range(len(data_maps)):
        data_frame_tmp = pd.DataFrame(
            {
                "participant": data_maps[i]["participant"] * 5,
                "map_number": data_maps[i]["map_number"] * 5,
                "x_list": data_maps[i]["x_list"],
                "y_list": data_maps[i]["y_list"],
                "zgf_list": data_maps[i]["zgf_list"],
                "condition": data_maps[i]["condition"] * 5,
                "xgf_list": data_maps[i]["xgf_list"],
                "ygf_list": data_maps[i]["ygf_list"],
                "muscle": list(data_maps[i]["muscle"][:5]),
            }
        )
        if data_frame_tot.empty:
            data_frame_tot = data_frame_tmp
        else:
            data_frame_tot = pd.concat([data_frame_tot, data_frame_tmp])
    data_frame = data_frame_tot.loc[data_frame_tot["participant"] != '006_TN']
    muscle_list = list(data_frame['muscle'][:3]) if not "sci" in results_dir else list(data_frame['muscle'][1:3])
    data_frame = data_frame.loc[data_frame['muscle'].isin(muscle_list)]
    list_points_tot = [[49, 98, 147, 196, 245],
                        [24, 44, 64, 84, 104, 124, 144, 164, 184],
                        [24, 44, 64, 84, 104, 124, 144, 164, 184]
                ]
    min_map_file = os.path.join(results_dir, "maps_min_map.csv")
    if not os.path.exists(min_map_file):
        raise RuntimeError('Min map file not found. Run generate_results/plot_wthin_method.py first.')
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

    if 'sci' in results_dir:
        list_points_tot[0] = list_points_tot[0][:-1]

    pd_res = pd.DataFrame()
    pd_res_min = pd.DataFrame()
    for c, cond in enumerate(['grid', 'pseudo']):
        list_points = list_points_tot[c][:-1] if c != 2 else list_points_tot[c]
        for m, map in enumerate(list_points):
            pseudo = data_frame.loc[(data_frame['condition'] == cond) & (data_frame['map_number'] == list_points.index(map))]
            ref = data_frame.loc[(data_frame['condition'] == cond) & (data_frame['map_number'] == list_points_tot[c].index(list_points_tot[c][-1]))] 
            if m == 0:
                min_map_tmp = pd_reduced.loc[pd_reduced['condition'] == cond]
                min_map_tmp['ref'] = 0
            for r in range(len(pseudo)):
                r2 = return_r2(ref.iloc[r].zgf_list.flatten(), pseudo.iloc[r].zgf_list.flatten())
                pd_tmp = pd.DataFrame({
                    'participant': [pseudo.iloc[r]['participant']],
                    'map_number': [m],
                    'number_stim': [map],
                    'r2': r2,
                    'condition': [cond],
                    'muscle': [pseudo.iloc[r]['muscle']],

                })
                pd_res = pd.concat([pd_res, pd_tmp])
                if m ==0:
                    r2 = return_r2(ref.iloc[r].zgf_list.flatten(), min_map_tmp.iloc[r].zgf_list.flatten())
                    pd_tmp = pd.DataFrame({
                        'participant': [pseudo.iloc[r]['participant']],
                        'map_number': ['min_map'],
                        'number_stim': [map],
                        'r2': r2,
                        'condition': [cond],
                        'muscle': [pseudo.iloc[r]['muscle']],

                    })
                    pd_res_min = pd.concat([pd_res_min, pd_tmp], ignore_index=True)
        
    # pr_sup = pd_res.loc[pd_res.r2 > 0.9]
    # test = pr_sup.groupby(['participant', 'map_number', 'condition', 'muscle']).max()

    loc = [(0, 0), (0, 3)]
    span = [3, 1]
    # fig, axs = plt.subplot_mosaic([['x_cog_line', 'x_cog_bar', 'y_cog_line', 'y_cog_bar'],
    #                                ['area_line', 'area_bar', 'volume_line', 'volume_bar']],
    #                               figsize=(6, 6),
    #                               width_ratios=(4, 1), height_ratios=(1, 4),
    #                               layout='constrained')
    fig_tot = plt.figure(layout='constrained', figsize=(10, 4))
    subfigs = fig_tot.subfigures(1, 1, wspace=0.05)
    # icc3 = pd_icc
    plt.rcParams["svg.fonttype"] = 'none'

    subfig = subfigs
    subfig.suptitle('Corelation with full-size maps', fontsize=16)
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
            sns.lineplot(x='number_stim', y='r2', hue='condition', data=pd_res, ax=ax,
                          marker='o', palette="rocket", hue_order=['grid', 'pseudo'], errorbar='sd')
            ax.set_title('Full-size correlation', fontsize=12)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, ['Grid', 'Pseudo'], title='', frameon=False)
        else:
            axes_tmp.append(fig_tot.add_subplot(gs[:, loc_tmp[1]:loc_tmp[1]+span_tmp], sharey=axes_tmp[i-1]))
            ax = axes_tmp[i]
            sns.barplot(x='condition', y='r2', data=pd_res_min, ax=ax, palette="rocket", order=['grid', 'pseudo'])
            ax.set_title('Tailored maps', fontsize=12)
            # ax.set_yticklabels([])
            ax.set_ylabel('')
        # ax.hlines(0.9, ax.get_xlim()[0], ax.get_xlim()[1], colors='gray', linestyles='dashed')
        # ax.set_ylim(0, 1 + 0.1)
        if i == 0:
            ax.set_ylabel('Pearson coefficient', fontsize=12)
        # elif r in [0, 2] and i == 1:
        #     ax.set_ylabel('')
        #     ax.set_yticklabels([])
        # elif r in [1, 3]:
        #     ax.set_ylabel('')
        #     ax.set_yticklabels([])
        if i == 0:
            ax.set_xlabel('Stimulation number')
        elif i == 1:
            ax.set_xlabel('')
            ax.set_xticklabels(['Grid', 'Pseudo'], rotation=0)
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])
    
    # sns.pointplot(pd_res[pd_res['condition'] == 'grid'], x="number_stim", y="r2")
    # sns.pointplot(pd_res[pd_res['condition'] == 'pseudo'], x="number_stim", y="r2")
    # sns.boxplot(pd_res, x="number_stim", y="r2", hue='condition')
    # plt.hlines(0.9, 0, plt.gca().get_xlim()[1], colors='gray', linestyles='dashed')
                
    plt.savefig(os.path.join(results_dir, 'r2_sup.png'))
    plt.show()
    plt.close()
    print('figure saved to', os.path.join(results_dir, 'r2_sup.png'))

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
    # ctx = mp.get_context("spawn")
    # with ctx.Pool(processes=4) as pool:
    #     pool.map(main, all_folder)