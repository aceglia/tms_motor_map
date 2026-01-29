from tkinter.font import families
from biosiglive import load
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_single_map(x, y, zgf, ax=None, n_point_grid=50, x_cog=None, y_cog=None, x_real=None, y_real=None):
    if ax is None:
        _, ax = plt.subplots()
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    xi = np.linspace(x_min, x_max, n_point_grid)
    yi = np.linspace(y_min, y_max, n_point_grid)
    xi, yi = np.meshgrid(xi, yi)
    x_flat = xi.flatten()
    y_flat = yi.flatten()
    z_flat = zgf.flatten()
    x = x_flat[z_flat > np.max(z_flat) * 0.1]
    y = y_flat[z_flat > np.max(z_flat) * 0.1]
    if x.size < 4:
        return 0, 0
    ax.contourf(xi, yi, zgf, n_point_grid, cmap="jet")
    if x_real is not None and y_real is not None:
        ax.scatter(x_real, y_real, s=8, alpha=0.5, marker="o", facecolors="none", edgecolors='white', linewidths=0.8)
    # if x_cog is not None and y_cog is not None:
    #     ax.scatter(x_cog, y_cog, c="k", s=150, marker="x")


if __name__ == '__main__':
    sci = False
    result_dir = 'smooth_10_5_10' if not sci else 'sci_smooth_10_5_2543'
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

    # for participant in data_frame_tot['participant'].unique():
    participant_to_plot = 'P010_TN' if not sci else 'P004_TN_SCI'
    muscle_to_plot = 'fdi' if not sci else 'ext_comm'
    map_number_list = [[49, 98, 147, 196, 245], [44, 64, 94, 124, 184]]
    map_num = [0, 5] if not sci else [0, 1, 2]
    data_frame_tmp = data_frame_tot #.loc[(data_frame_tot['participant'] == participant_to_plot) & (data_frame_tot['muscle'] == muscle_to_plot)]
    data_frame_tmp = data_frame_tot.loc[data_frame_tot['participant'] != 'P006_TN']

    data_frame_tmp = data_frame_tmp.loc[~((data_frame_tmp['condition'] == 'pseudo') & (data_frame_tmp['map_number'].isin(map_num)))]
    # data_frame_tmp = data_frame_tmp.loc[~((data_frame_tmp['condition'] == 'grid'))] # & (data_frame_tmp['map_number'].isin([0])))]

    # fig, axes = plt.subplots(2, 5)
    fig = plt.figure(constrained_layout=True, num=participant_to_plot)
    cond_y_label = ['Grid', 'Pseudo-random']
    subfigs = fig.subfigures(nrows=2, ncols=1)
    for c, cond in enumerate(data_frame_tmp['condition'].unique()):
        subfigs[c].suptitle(f'{cond_y_label[c]}')
        fig.supylabel(f"Medio-lateral (mm)")
        fig.supxlabel(f"Antero-posterior (mm)")
        map_number_list_tmp = data_frame_tmp.loc[data_frame_tmp['condition'] == cond]['map_number'].unique()
        axes = subfigs[c].subplots(nrows=1, ncols=len(map_number_list_tmp))
        for m, map_number in enumerate(map_number_list_tmp):
            data_frame_tmp_map = data_frame_tmp.loc[(data_frame_tmp['condition'] == cond) & (data_frame_tmp['map_number'] == map_number)]
            x_list = data_frame_tmp_map['xgf_list'].loc[data_frame_tmp_map['xgf_list'].index == 0].mean()
            y_list = data_frame_tmp_map['ygf_list'].loc[data_frame_tmp_map['ygf_list'].index == 0].mean()
            z_list = data_frame_tmp_map['zgf_list'].loc[data_frame_tmp_map['zgf_list'].index == 0].mean()
            # x_real = data_frame_tmp_map['x_list'].loc[data_frame_tmp_map['x_list'].index == 0].mean()
            # y_real = data_frame_tmp_map['y_list'].loc[data_frame_tmp_map['y_list'].index == 0].mean()
            ax = axes[m]
            plot_single_map(x_list, y_list, z_list, ax, 50, x_real=None, y_real=None)
            ax.set_title(f"Nb stim: {map_number_list[c][m]}")
            ax.set_aspect("equal")
    plt.show()
