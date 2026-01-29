import os
import pandas as pd
import matplotlib.pyplot as plt
from biosiglive import load
from utils import *


if __name__ == '__main__':
    result_dir = 'smooth_5_5_10'
    data_frame = pd.read_csv(os.path.join(result_dir, "maps_characteristics.csv"))

    data_maps = load(os.path.join(result_dir, 'maps_values.bio'), merge=False)[-144:]
    data_frame_tot = pd.DataFrame()
    for i in range(len(data_maps)):
        data_frame_tmp = pd.DataFrame({'participant': data_maps[i]['participant'] * 5, 
                        'map_number': data_maps[i]['map_number'] * 5, 
                        'x_list': data_maps[i]['x_list'],
                        'y_list': data_maps[i]['y_list'],
                        'zgf_list': data_maps[i]['zgf_list'],
                        'condition': data_maps[i]['condition'] * 5,
                        'xgf_list': data_maps[i]['xgf_list'],
                        'ygf_list': data_maps[i]['ygf_list'],
                        'muscle': list(data_frame['muscle'][:5]),
                        })
        if data_frame_tot.empty:
            data_frame_tot = data_frame_tmp
        else:
            data_frame_tot = pd.concat([data_frame_tot, data_frame_tmp])
    # merge dataframe
    data_frame = pd.merge(data_frame, data_frame_tot, on=['participant','map_number', 'condition', 'muscle'])
    
    data_frame = data_frame.loc[data_frame["participant"] != 'P006_TN']
    participants = data_frame["participant"].unique()

    condition = ['grid', 'pseudo']
    # sns.set(font_scale=1.5)
    keys = ['euclid_cog_error', 'correlation_coefficient'] 
    df_cond = pd.DataFrame()
    for c, cond in enumerate(condition):
        grid_data_frame = data_frame.loc[data_frame["condition"] == cond].loc[data_frame["participant"].isin(participants)]
        grid_data_frame = grid_data_frame.loc[grid_data_frame["muscle"].isin(list(grid_data_frame['muscle'][:3]))]
        muscle_list = list(data_frame['muscle'][:3])
        #, 'area_error', 'volume_error']
        grid_data_frame = recompute_correlation(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_euclid_dist(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_area_error(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_volume_error(participants, grid_data_frame, muscle_list)
             

        df_cond = pd.concat([df_cond, grid_data_frame], ignore_index=True)

    df_cond = df_cond[["map_number", 'correlation_coefficient','euclid_cog_error', 
                                "muscle", 'participant', 'condition']].dropna()
    df_cond.loc[df_cond['condition'] == 'grid', 'map_number'] = (df_cond.loc[df_cond['condition'] == 'grid', 'map_number'].values + 1) * 49
    # replace 1 by 44, 2 by 64, 3 by 94, 4 by 124, 5 by 154, 6 by 184   
    list_number = [44, 64, 94, 124, 154, 184]
    df_cond.loc[df_cond['condition'] == 'pseudo', 'map_number'] = df_cond.loc[df_cond['condition'] == 'pseudo', 'map_number'].apply(lambda x: list_number[x - 1])
    # set svg font to none
    plt.rcParams['svg.fonttype'] = 'none'

    pd_maps = pd.DataFrame()
    for part in participants:
        for c in condition:
            maps_euclid = min_map(df_cond.loc[df_cond['condition'] == c], part, euclid = True, c=cond)
            maps_corr = min_map(df_cond.loc[df_cond['condition'] == c], part, euclid = False, c=cond)
            pd_tmp = pd.DataFrame({'participant': [part] * 3,'condition': [c] * 3, 'map_number_euclid': maps_euclid, 'correlation_coefficient': maps_corr, 'muscle': ['fdi', 'ext_comm', 'sup']})
            pd_maps = pd.concat([pd_maps, pd_tmp], ignore_index=True)
    pd_maps['min_map_number'] = pd_maps[['map_number_euclid', 'correlation_coefficient']].max(axis=1) 
    pd_maps.to_csv(os.path.join(result_dir, 'maps_min_map.csv'))
    
    # data_frame.loc[data_frame['condition'] == 'grid', 'map_number'] = (data_frame.loc[data_frame['condition'] == 'grid', 'map_number'].values + 1) * 49
    # # replace 1 by 44, 2 by 64, 3 by 94, 4 by 124, 5 by 154, 6 by 184   
    # list_number = [24, 44, 64, 94, 124, 154, 184]
    # data_frame.loc[data_frame['condition'] == 'pseudo', 'map_number'] = data_frame.loc[data_frame['condition'] == 'pseudo', 'map_number'].apply(lambda x: list_number[x - 1])
    # # set svg font to none
    # data_frame = data_frame[["map_number", 'area','volume', 'x_cog', 'y_cog', 
    #                             "muscle", 'participant', 'condition']]
    # # to_compare_maps = data_frame[["map_number", 'area','volume', 'x_cog', 'y_cog', 
    # #                             "muscle", 'participant', 'condition']]
    # ratings = ['area','volume', 'x_cog', 'y_cog']
    # pd_tot = pd.DataFrame()
    # for r in ratings:
    #     for part in participants:
    #         for muscle in ['fdi', 'ext_comm','sup']:
    #             for cond in condition:
    #                 map_number = data_frame.loc[data_frame['condition'] == cond, 'map_number'].unique()[-1]
    #                 reference_map = data_frame.loc[(data_frame['condition'] == cond) & (data_frame['map_number'] == map_number)]
    #                 to_compare_maps_tmp = data_frame.loc[(data_frame['participant'] == part) & (data_frame['condition'] == cond) & (data_frame['muscle'] == muscle)]
    #                 min_map_tmp = pd_maps.loc[(pd_maps['participant'] == part) & (pd_maps['condition'] == cond) & (pd_maps['muscle'] == muscle)]['min_map_number'].values[0]
    #                 to_compare_maps_tmp = to_compare_maps_tmp.loc[to_compare_maps_tmp['map_number'] == min_map_tmp]
    #                 reference_map_tmp = reference_map.loc[(reference_map['participant'] == part) & (reference_map['condition'] == cond) & (reference_map['muscle'] == muscle)]
    #                 if not to_compare_maps_tmp.empty and not reference_map_tmp.empty:
    #                     error = np.sqrt(np.mean((to_compare_maps_tmp[r].values - reference_map_tmp[r].values)**2))
    #                     pd_tmp = pd.DataFrame({'participant': [part],'condition': [cond], r + '_error': [error],'muscle': [muscle]})
    #                     pd_tot = pd.concat([pd_tot, pd_tmp], ignore_index=True)
