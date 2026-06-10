import pandas as pd
import os


if __name__ == '__main__':
    seeds = [0]
    smooth_1 = [6]
    smooth_2 = [6]
    all_folder = []
    for s in seeds:
        for s1 in smooth_1:
            for s2 in smooth_2:
                all_folder.append(rf"D:\Documents\Programmation\tms_motor_map\results\smooth_{s1}_{s2}_{s}_ransac")

    maps_data = os.path.join(all_folder[0], "maps_characteristics.csv")
    data_frame = pd.read_csv(maps_data)
    data_frame = data_frame[['participant', 'condition', 'map_number', 'x_cog', 'y_cog', 'area', 'volume', 'muscle']]
    #new column being the mean of x_cog and y_cog
    data_frame['cog'] = (data_frame['x_cog']**2 + data_frame['y_cog']**2) ** 0.5
    data_frame = data_frame.loc[data_frame['muscle'].isin(['fdi', 'ext_comm', 'sup'])]
    pseudo_df = data_frame.loc[data_frame['condition'] == 'pseudo']
    grid_df = data_frame.loc[data_frame['condition'] == 'grid']
    pseudo_df.to_csv(os.path.join(all_folder[0], "pseudo_maps_LMM.csv"), index=False)
    grid_df.to_csv(os.path.join(all_folder[0], "grid_maps_LMM.csv"), index=False)

    # create a new columns for each map number and fill it with the mean value of the corresponding map 
    pd_pseudo_maps = pseudo_df.loc[pseudo_df['map_number'] == 0]
    pd_pseudo_maps.drop(columns=f'map_number', inplace=True)
    for i in range(1, max(pseudo_df.map_number.unique())):
        pd_pseudo_maps = pd.merge(pd_pseudo_maps, pseudo_df.loc[pseudo_df['map_number'] == i], on=('participant', 'condition', 'muscle'), suffixes=('', f'_{i}'))

    pd_grid_maps = grid_df.loc[grid_df['map_number'] == 0]
    pd_grid_maps.drop(columns=f'map_number', inplace=True)
    for i in range(1, max(grid_df.map_number.unique())):
        pd_grid_maps = pd.merge(pd_grid_maps, grid_df.loc[grid_df['map_number'] == i], on=('participant', 'condition', 'muscle'), suffixes=('', f'_{i}'))

    pd_pseudo_maps.to_csv(os.path.join(all_folder[0], "pseudo_maps_stats.csv"), index=False)
    pd_grid_maps.to_csv(os.path.join(all_folder[0], "grid_maps_stats.csv"), index=False)