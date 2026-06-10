import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

if __name__ == '__main__':
    data_dir = r'D:\Documents\Programmation\tms_motor_map\results\smooth_6_6_0_ransac'
    pd_data = pd.read_csv(os.path.join(data_dir, 'maps_characteristics.csv'))
    min_maps = pd.read_csv(os.path.join(data_dir, "maps_min_map.csv"))
    nb_layers = ((min_maps.loc[min_maps['condition'] == 'grid']['min_map_number'] // 49) -1 ) * 120
    min_maps['min_map_number'] *= 4
    time_grid = min_maps.loc[min_maps['condition'] == 'grid']['min_map_number']
    time_pseudo = min_maps.loc[min_maps['condition'] == 'pseudo']['min_map_number']
    print('grid_time_raw', time_grid.mean(), time_grid.std())
    print('pseudo_time', time_pseudo.mean(), time_pseudo.std())
    print('grid_time_with_pause', (time_grid + (nb_layers)).mean(), (time_grid + (nb_layers)).std())

    pd_data = pd_data.drop(columns=['Unnamed: 0'])
    time = pd_data['time_to_compute']
    mean_time = np.mean(time)
    std_time = np.std(time)
    print(f"Mean time to compute: {mean_time:.2f}s, Std: {std_time:.2f}s")
