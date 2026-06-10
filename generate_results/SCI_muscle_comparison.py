import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data(data_dir, sci=False):
    data_frame = pd.read_csv(os.path.join(data_dir, "maps_characteristics.csv"))
    data_frame = data_frame.loc[data_frame["participant"] != '006_TN']
    muscle_list = list(data_frame['muscle'].unique()[1:3])
    data_frame = data_frame.loc[data_frame['muscle'].isin(muscle_list)]
    list_points_tot = [[49, 98, 147, 196, 245],
                        # [34, 64, 94, 124, 154, 184],
                        # [34, 64, 94, 124, 154, 184]
                        [24, 44, 64, 84, 104, 124, 144, 164, 184],
                        [24, 44, 64, 84, 104, 124, 144, 164, 184]
                ] if not "sci" in data_dir else [[49, 98, 147, 196],
                        # [34, 64, 94, 124, 154, 184],
                        # [34, 64, 94, 124, 154, 184]
                        [24, 44, 64, 84, 104, 124, 144, 164, 184],
                        [24, 44, 64, 84, 104, 124, 144, 164, 184]
                ]
    min_maps = pd.read_csv(os.path.join(data_dir, "maps_min_map.csv"))
    pd_reduced = pd.DataFrame()
    for i, participant in enumerate(min_maps['participant'].unique()):
        for j, muscle in enumerate(min_maps['muscle'].unique()):
            for c, cond in enumerate(['grid', 'pseudo']):
                min_map = min_maps.loc[(min_maps['participant'] == participant) & (min_maps['muscle'] == muscle) & (min_maps['condition'] == cond)]['min_map_number'].values[0]
                if not np.isfinite(min_map):
                    continue
                
                pd_tmp = data_frame.loc[(data_frame['participant'] == participant) & (data_frame['muscle'] == muscle) & (data_frame['condition'] == cond) & (data_frame['map_number'] == list_points_tot[c].index(min_map))]
                pd_reduced = pd.concat([pd_reduced, pd_tmp], ignore_index=True)
    return pd_reduced


def compute_z_score(pd_ref, pd_sci):
    pd_z = pd.DataFrame()
    for cond in ['grid', 'pseudo']:
        for rating in ['area', 'volume']:
            for muscle in ['ext_comm','sup']:
                ref_data = pd_ref.loc[(pd_ref['condition'] == cond) & (pd_ref['muscle'] == muscle)][rating]
                sci_data = pd_sci.loc[(pd_sci['condition'] == cond) & (pd_sci['muscle'] == muscle)][rating]
                z_score_sci = (sci_data.values - np.mean(ref_data.values)) / np.std(ref_data.values)
                z_score_ref = (ref_data.values - np.mean(ref_data.values)) / np.std(ref_data.values)
                nb_data = len(z_score_ref)
                pd_tmp = {
                    'condition': [cond]*nb_data,
                    'muscle': [muscle]*nb_data,
                    'rating': [rating]*nb_data,
                    'z_ref': z_score_ref,
                    'z_sci': np.repeat(z_score_sci, len(z_score_ref)),
                }
                pd_z = pd.concat([pd_z, pd.DataFrame(pd_tmp)])
    return pd_z


if __name__ == '__main__':
    data_dir_sci = r"D:\Documents\Programmation\tms_motor_map\results\smooth_6_6_0_ransac_sci"
    data_dir = r'D:\Documents\Programmation\tms_motor_map\results\smooth_6_6_0_ransac' 
    pd_sci = get_data(data_dir_sci, True)
    pd_parts = get_data(data_dir)
    res_pd = pd.DataFrame()
    for mus in pd_parts['muscle'].unique():
        for rate in ['x_cog', 'y_cog', 'area', 'volume']:
            for cond in ['grid', 'pseudo']:
                pd_parts_tmp = pd_parts.loc[(pd_parts['muscle'] == mus) & (pd_parts['condition'] == cond)][rate]
                pd_sci_tmp = pd_sci.loc[(pd_sci['muscle'] == mus) & (pd_parts['condition'] == cond)][rate]
                Q25 = pd_parts_tmp.quantile(0.10)
                Q95 = pd_parts_tmp.quantile(0.90)
                mean_parts = pd_parts_tmp.mean()
                sci_value = pd_sci_tmp.values
                pd_tmp = {
                    "q25": Q25, 
                    "q95": Q95, 
                    'mean_parts':mean_parts, 
                    "mean_sci": sci_value,
                    'muscle': mus,
                    'rating': rate, 
                    'condition': cond,
                }
                res_pd = pd.concat([res_pd, pd.DataFrame(pd_tmp, index=[0])], ignore_index=True)
    pd_res = compute_z_score(pd_parts, pd_sci)
    # set font svg to None
    plt.rcParams['svg.fonttype'] = 'none'
    rating = ['x_cog', 'y_cog',
               'area', 'volume']
    y_label = {'x_cog': 'COG AP (mm)','y_cog': 'COG ML (mm)', 
               'area': 'Area', 'volume': 'Normalized volume'}
    x_label = ['Extensor digitorum', 'Supinator']
    x_label = ['Grid', 'Pseudo']
    fontbase = 14
    big= fontbase + 2
    bigger = fontbase + 4
    fig, ax = plt.subplots(1, 2, sharey=False)
    axes = ax.flatten()
    for r, rate in enumerate(rating[2:]): 
        ax = axes[rating.index(rate) - 2]
        pd_tmp_rate_sci = pd_sci[[rate, 'muscle', 'condition']] #.loc[pd_sci['condition'] == key]
        pd_tmp_rate = pd_parts[[rate, 'muscle', 'condition']] #.loc[pd_parts['condition'] == key]
        # pd_dif_sci = compute_diff_pd(pd_tmp_rate_sci, rate)
        # pd_to_plot = pd_res.loc[pd_res['rating'] == rate]
        
        # palette = sns.color_palette('rocket')
        # # pd_dif = compute_diff_pd(pd_tmp_rate, rate) 
        # # sns.lineplot(x='muscle', y=rate, data=pd_tmp_rate.loc[pd_tmp_rate['condition'] == 'grid'], ax=ax)
        # # sns.lineplot(x='condition', y=rate, data=pd_tmp_rate, ax=ax, palette='rocket', hue='muscle', alpha=0.5)#,  errorbar='sd')
        # sns.violinplot(x='condition', y='z_ref', data=pd_to_plot, hue='muscle', ax=ax, palette=[palette[1], palette[3]], alpha=0.4, cut=0, inner_kws={'alpha': 0.5})#,  errorbar='sd')
        # sns.swarmplot(x='condition', y='z_ref', data=pd_to_plot, hue='muscle', ax=ax, palette=[palette[1], palette[3]], dodge=True)#,  errorbar='sd')
        # pd_sci_unique = pd_res.groupby(['muscle', "condition", 'rating']).mean(numeric_only=True).reset_index()
        # markers = ['*', 'o']
        # for m, mus in enumerate(pd_sci_unique['muscle'].unique()):
        #     y = pd_sci_unique.loc[(pd_sci_unique['rating'] == rate) & (pd_sci_unique['muscle'] == mus)]['z_sci'].values
        #     cond = pd_sci_unique.loc[(pd_sci_unique['rating'] == rate) & (pd_sci_unique['muscle'] == mus)]['condition'].values
        #     x = [0.1 if c == 'grid' else 0.9 for c in cond]
        #     ax.scatter(x, y, color='k', s=50, marker=markers[m], label=mus)
        # sns.swarmplot(x='condition', y=rate, data=pd_tmp_rate, ax=ax, color='gray', dodge=True)#,  errorbar='sd')

        sns.pointplot(x='condition', y=rate, data=pd_tmp_rate_sci, hue='muscle', ax=ax, palette='rocket')
        sns.pointplot(x='condition', y=rate, data=pd_tmp_rate_sci, ax=ax)
        sns.lineplot(x='condition', y=rate, hue="muscle", data=pd_tmp_rate, ax=ax)
        # ax.margins(x=0.1)
        # sns.lineplot(x='condition', y='muscle_diff',  data=pd_dif, ax=ax)
        # sns.pointplot(x='condition', y='muscle_diff',  data=pd_dif_sci, ax=ax)
        ax.set_title(y_label[rate], fontsize=bigger)
        # ax.legend(labels=['SCI patient'], fontsize=fontbase, loc='upper right')
        ax.legend(fontsize=fontbase, loc='upper right')


        if r == 0:
            ax.set_ylabel('Z-score', fontsize=big)
            ax.tick_params(axis='y', labelrotation=0, labelsize=fontbase)
        ax.set_xticklabels(x_label, fontsize=big)
        ax.set_xlabel('')

    plt.show()
