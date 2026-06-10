import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from biosiglive import load
from scipy.stats import pearsonr
import os

def return_r2(ref, pseudo):
    r2 = pearsonr(ref, pseudo)[0]
    # r2 = np.corrcoef(ref, pseudo)[0, 1]
    # r2 = r2_score(ref, pseudo)
    # res = linregress(ref, pseudo)
    # r2 = res.rvalue ** 2
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
    muscle_list = list(data_frame['muscle'][:3])
    data_frame = data_frame.loc[data_frame['muscle'].isin(muscle_list)]
    list_points_tot = [[49, 98, 147, 196, 245],
                       [24, 44, 64, 84, 104, 124, 144, 164, 184],
                       [24, 44, 64, 84, 104, 124, 144, 164, 184],
                        # [34, 64, 94, 124, 154, 184],
                        # [34, 64, 94, 124, 154, 184]
                ]
    if 'sci' in results_dir:
        list_points_tot[0] = list_points_tot[0][:-1]

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

    pd_res = pd.DataFrame()
    for c, cond in enumerate(['grid', 'pseudo']):
        # for m, map in enumerate(list_points):
        pseudo = pd_reduced.loc[(pd_reduced['condition'] == cond)]
        ref = data_frame.loc[(data_frame['condition'] == cond) & (data_frame['map_number'] == list_points_tot[c].index(list_points_tot[c][-1]))] 
        for r in range(len(pseudo)):
            r2 = return_r2(ref.iloc[r].zgf_list.flatten(), pseudo.iloc[r].zgf_list.flatten())
            
            # r2 = np.dot(ref.iloc[r].zgf_list.ravel(),pseudo.iloc[r].zgf_list.ravel()) / (
            #     np.linalg.norm(ref.iloc[r].zgf_list.ravel())*np.linalg.norm(pseudo.iloc[r].zgf_list.ravel())
            # )
            # r2 = ssim(ref.iloc[r].zgf_list, pseudo.iloc[r].zgf_list, data_range=max(ref.iloc[r].zgf_list.max(), pseudo.iloc[r].zgf_list.max()) - min(ref.iloc[r].zgf_list.min(), pseudo.iloc[r].zgf_list.min()))
            # if r2 < 0.8:
            #     print(1)
            # r2 = return_ssim(ref.iloc[r].zgf_list, pseudo.iloc[r].zgf_list)
            pd_tmp = pd.DataFrame({
                'participant': [pseudo.iloc[r]['participant']],
                # 'map_number': [m],
                # 'number_stim': [map],
                'r2': r2,
                'condition': [cond],
                'muscle': [pseudo.iloc[r]['muscle']],
            })
            pd_res = pd.concat([pd_res, pd_tmp])
        
    # pr_sup = pd_res.loc[pd_res.r2 > 0.9]
    rocket_palette = sns.color_palette("rocket")
    # test = pr_sup.groupby(['participant', 'map_number', 'condition', 'muscle']).max()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(pd_res, y='r2', x='condition', palette=[rocket_palette[1], rocket_palette[3]], alpha=0.5, ax=ax, cut=0)
    sns.swarmplot(pd_res, y='r2', x='condition', palette=[rocket_palette[1], rocket_palette[3]], ax=ax, dodge = True )
    ax.set_xlabel('Methods', fontsize=16)
    ax.set_ylabel('Pearson correlation', fontsize=16)
    ax.tick_params(axis='y', labelrotation=0, labelsize=14)
    
    ax.set_xlabel('')
    ax.set_xticklabels(['Grid', 'Pseudo'], rotation=0, fontsize=16)
                
    plt.savefig(os.path.join(results_dir, 'r2.png'))
    plt.close()
    print('figure saved to', os.path.join(results_dir, 'r2.png'))

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