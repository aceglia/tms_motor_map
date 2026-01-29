import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygridfit import GridFit
import seaborn as sns
from scipy.stats import pearsonr
from biosiglive import load

def min_map(grid_data_frame, part, euclid = True):
    if euclid:
        cor_coef = grid_data_frame.loc[grid_data_frame['participant'] == part].loc[grid_data_frame['euclid_cog_error'] <= 3.6]
    else:
        cor_coef = grid_data_frame.loc[grid_data_frame['participant'] == part].loc[grid_data_frame['correlation_coefficient'] >= 0.9]
    if not cor_coef.empty:
        map_muscle = (cor_coef['map_number'].values, cor_coef['muscle'].values)
        map_list = []
        
        for muscle in ['fdi', 'ext_comm', 'sup']:
            idx_muscle = np.where(cor_coef['muscle'].values == muscle)[0]
            if idx_muscle.size == 0:
                map_list.append(-1)
            else:
                if c == 'pseudo':
                    maps = [24, 44, 64, 94, 124, 154, 184][min(map_muscle[0][idx_muscle])]
                else:
                    maps = min(map_muscle[0][idx_muscle] + 1)
                map_list.append(maps)
    else:
        return [-1] * 3
    return map_list

def recompute_correlation(participants, grid_data_frame, muscle_list):
    for p in participants:
        cor_coef = [[np.nan for _ in range(len(muscle_list))]] + [
            [
                pearsonr(grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i+1].loc[grid_data_frame['muscle']==m]["zgf_list"].values[0].flatten(),
                        grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["zgf_list"].values[0].flatten())[0]
                for m in muscle_list
            ]
            for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "correlation_coefficient"] = sum(cor_coef, [])
    return grid_data_frame

def recompute_euclid_dist(participants, grid_data_frame, muscle_list):
    for p in participants:
        cog_err_eucl = [[np.nan for _ in range(len(muscle_list))]]
        cog_err_eucl = cog_err_eucl + [
            [
        np.linalg.norm(np.array([grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i+1].loc[grid_data_frame['muscle']==m]["x_cog"],
                                 grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i+1].loc[grid_data_frame['muscle']==m]["y_cog"]])
                                   - np.array([grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["x_cog"],
                                                grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["y_cog"]]), axis=0)[0]
            for m in muscle_list
            ]
        for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "euclid_cog_error"] = sum(cog_err_eucl, [])
    return grid_data_frame

def rmse(ref, pred):
    return np.sqrt(np.mean((ref - pred)**2))

def recompute_area_error(participants, grid_data_frame, muscle_list):
    for p in participants:
        cog_err_eucl = [[np.nan for _ in range(len(muscle_list))]]
        cog_err_eucl = cog_err_eucl + [
            [rmse(np.array(grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i+1].loc[grid_data_frame['muscle']==m]["area"]),
                            np.array(grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["area"]))
            for m in muscle_list
            ]
        for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "area_error"] = sum(cog_err_eucl, [])
    return grid_data_frame

def recompute_volume_error(participants, grid_data_frame, muscle_list):
    for p in participants:
        cog_err_eucl = [[np.nan for _ in range(len(muscle_list))]]
        cog_err_eucl = cog_err_eucl + [
            [rmse(np.array(grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i+1].loc[grid_data_frame['muscle']==m]["volume"]),
                            np.array(grid_data_frame.loc[grid_data_frame["participant"] == p].loc[grid_data_frame["map_number"] == i].loc[grid_data_frame['muscle']==m]["volume"]))
            for m in muscle_list
            ]
        for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "volume_error"] = sum(cog_err_eucl, [])
    return grid_data_frame

def plot(df, x, y, hue, name):
    # fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=3, num=name)
    muscle_name = ['fdi', 'ext_comm', 'sup']
    muscle_name_title = ['FDI', 'Extenseur commun', 'Supinateur']
    # replace muscles name in dataframe by ['FDI', 'Extenseur commun', 'Supinateur']
    # df['muscle'] = df['muscle'].replace({'fdi': 'FDI', 'ext_comm': 'Extenseur commun','sup': 'Supinateur'})

    # for j in range(3):
    plt.figure(name)
    ax = plt.gca()
    sns.lineplot(
        df.loc[df[y] != 0], #.loc[df["muscle"] == muscle_name[j]],
        x=x,
        y=y,
        # hue=hue,
        marker="o",
        ax=ax, 
    )
    # if j != 2:
    #     ax[j].get_legend().remove()
    if 'correlation' in y:
        ax.axhline(y=0.9, color='r', linestyle='--')
        ax.set_ylabel('Correlation coefficient')
    if 'euclid' in y:
        ax.axhline(y=3.6, color='g', linestyle='--')
        ax.set_ylabel('Euclidian distance')
    if 'pseudo' in name:
        ax.set_xticks(list(range(1, 7)))
        ax.set_xticklabels([str(i) for i in [44, 64, 94, 124, 154, 184]])
        ax.set_xlabel('Number of points')
    else:
        ax.set_xticks(list(range(1, 5)))
        ax.set_xticklabels([str(i) for i in range(2, 6)])
        ax.set_xlabel('Map number')

    plt.savefig(f"{name}.png")

if __name__ == '__main__':
    result_dir = 'sci_smooth_7_3_10'
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
    paticipants = list(range(2, 14))
    paticipants.pop(paticipants.index(6))
    participants = [f"P{p:03d}_TN" for p in paticipants]
    cond = ['grid', 'pseudo']
    sns.set(font_scale=1.5)
    for c in cond:
        grid_data_frame = data_frame.loc[data_frame["condition"] == c].loc[data_frame["participant"].isin(participants)]
        grid_data_frame = grid_data_frame.loc[grid_data_frame["muscle"].isin(list(grid_data_frame['muscle'][:3]))]
        muscle_list = list(data_frame['muscle'][:3])
        keys = ['euclid_cog_error', 'correlation_coefficient', 'area_error', 'volume_error']
        grid_data_frame = recompute_correlation(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_euclid_dist(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_area_error(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_volume_error(participants, grid_data_frame, muscle_list)

        pd_maps = pd.DataFrame()
        for part in participants:
            maps_euclid = min_map(grid_data_frame, part, euclid = True)
            maps_corr = min_map(grid_data_frame, part, euclid = False)
            pd_tmp = pd.DataFrame({'participant': [part] * 3, 'map_number_euclid': maps_euclid, 'correlation_coefficient': maps_corr, 'muscle': ['fdi', 'ext_comm', 'sup']})
            if pd_maps.empty:
                pd_maps = pd_tmp
            else:
                pd_maps = pd.concat([pd_maps, pd_tmp])
        # muscle_list = [1, 2, 3]
        # paticipants = list(range(2, 14))
        # paticipants.pop(paticipants.index(6))
        # # replace muscles name in dataframe by [1, 2, 3]

        # pd_maps = pd_maps.replace({'fdi': 1, 'ext_comm': 2,'sup': 3})
        # # replace participants name in dataframe by [2, 3, 4, 5, 7, 8, 9, 10, 11,12, 13]
        # pd_maps['participant'] = pd_maps['participant'].replace({f"P{p:03d}_TN": p for p in paticipants})

        pd_maps.to_csv(os.path.join(result_dir, f"{c}_maps.csv"))

        paticipants = list(range(2, 14))
        paticipants.pop(paticipants.index(6))
        fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1, num=f"{c}_maps_euclid")
        sns.barplot(x='participant', y='map_number_euclid', data=pd_maps, ax=ax
                    , hue='muscle'
                    )
        y_lim = (1, 5) if c == 'grid' else (24, 184)
        ax.set_ylim(y_lim)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([str(i) for i in paticipants])
        y_label = 'Map number' if c == 'grid' else 'Number of points'
        ax.set_ylabel(y_label)
        ax.set_xlabel('Participant')
        
        if c == 'pseudo':
            ax.set_title(f"Number of points needed for a COG euclidian distance of 3.6 mm.")
        else:
            ax.set_title(f"Number of maps needed for a COG euclidian distance of 3.6 mm.")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=['FDI', 'Extenseur commun', 'Supinateur'])
        # plt.savefig(f"{c}_maps_euclid.png")

        fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1, num=f"{c}_maps_corr")
        sns.barplot(x='participant', y='correlation_coefficient', data=pd_maps, hue='muscle', ax=ax)
        y_lim = (1, 5) if c == 'grid' else (24, 184)
        ax.set_ylim(y_lim)
        y_label = 'Map number' if c == 'grid' else 'Number of points'
        ax.set_ylabel(y_label)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([str(i) for i in paticipants])
        ax.set_xlabel('Participant number')
        if c == 'pseudo':
            ax.set_title(f"Number of points needed for a correlation coefficient of 0.9.")
        else:
            ax.set_title(f"Number of maps needed for a correlation coefficient of 0.9.")
        # change legend of muscle by ['FDI', 'Extenseur commun', 'Supinateur']
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=['FDI', 'Extenseur commun', 'Supinateur'])
        # plt.savefig(f"{c}_maps_corr.png")
    plt.show()
