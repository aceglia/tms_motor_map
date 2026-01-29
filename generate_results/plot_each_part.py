import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from biosiglive import load

def min_map(grid_data_frame, part, euclid = True, c='grid'):
    if euclid:
        cor_coef = grid_data_frame.loc[grid_data_frame['participant'] == part].loc[grid_data_frame['euclid_cog_error'] <= 3.6]
    else:
        cor_coef = grid_data_frame.loc[grid_data_frame['participant'] == part].loc[grid_data_frame['correlation_coefficient'] >= 0.9]
    map_list = []
    if not cor_coef.empty:
        for muscle in cor_coef['muscle'].unique():
            pd_tmp_muscle = cor_coef.loc[cor_coef['muscle'] == muscle]
            min_map = pd_tmp_muscle['map_number'].min()
            map_list.append(min_map)
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
    data_dir = "smooth_8_5_84682"
    file = os.path.join(data_dir, 'maps_characteristics.csv')
    muscle_to_keep = ['fdi', 'ext_comm', 'sup']
    number = file.removesuffix('.csv').split('_')[-1]
    pd_tmp = pd.read_csv(file)
    pd_tmp.drop(columns='Unnamed: 0')
    data_frame = pd_tmp.loc[pd_tmp['muscle'].isin(muscle_to_keep)]
    data_maps = load(os.path.join(data_dir, f'maps_values.bio'), merge=False)
    data_frame_tot = pd.DataFrame()
    for i in range(len(data_maps)):
        n_mus = len(muscle_to_keep)
        data_frame_tmp = pd.DataFrame({'participant': data_maps[i]['participant'] * n_mus, 
                        'map_number': data_maps[i]['map_number'] * n_mus, 
                        'x_list': data_maps[i]['x_list'][:n_mus],
                        'y_list': data_maps[i]['y_list'][:n_mus],
                        'zgf_list': data_maps[i]['zgf_list'][:n_mus],
                        'condition': data_maps[i]['condition'] * n_mus,
                        'xgf_list': data_maps[i]['xgf_list'][:n_mus],
                        'ygf_list': data_maps[i]['ygf_list'][:n_mus],
                        'muscle': list(data_frame['muscle'][:n_mus]),
                        })
        if data_frame_tot.empty:
            data_frame_tot = data_frame_tmp
        else:
            data_frame_tot = pd.concat([data_frame_tot, data_frame_tmp])

    # merge dataframe
    data_frame = pd.merge(data_frame, data_frame_tot, on=['participant','map_number', 'condition', 'muscle'])
    data_frame = data_frame.loc[data_frame["participant"] != 'P006_TN']
    participants = data_frame["participant"].unique()
    cond = ['grid', 'pseudo']
    # sns.set(font_scale=1.5)
    df_cond = pd.DataFrame()
    for c in cond:
        grid_data_frame = data_frame.loc[data_frame["condition"] == c].loc[data_frame["participant"].isin(participants)]
        # grid_data_frame = grid_data_frame.loc[grid_data_frame["muscle"].isin(list(grid_data_frame['muscle'][:3]))]
        muscle_list = list(data_frame['muscle'][:3])
        keys = ['euclid_cog_error', 'correlation_coefficient', 'area_error', 'volume_error']
        grid_data_frame = recompute_correlation(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_euclid_dist(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_area_error(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_volume_error(participants, grid_data_frame, muscle_list)
        df_cond = pd.concat([df_cond, grid_data_frame], ignore_index=True)



    df_cond = df_cond[["map_number", 'correlation_coefficient','euclid_cog_error', 
                                "muscle", 'participant', 'condition']].dropna()
    df_cond.loc[df_cond['condition'] == 'grid', 'map_number'] = (df_cond.loc[df_cond['condition'] == 'grid', 'map_number'].values + 1) * 49 
    list_number = [44, 64, 94, 124, 154, 184]
    df_cond.loc[df_cond['condition'] == 'pseudo', 'map_number'] = df_cond.loc[df_cond['condition'] == 'pseudo', 'map_number'].apply(lambda x: list_number[x - 1]) 
    pd_maps = pd.DataFrame()
    for part in participants:
        for c in cond:
            maps_euclid = min_map(df_cond.loc[df_cond['condition'] == c], part, euclid = True, c=cond)
            maps_corr = min_map(df_cond.loc[df_cond['condition'] == c], part, euclid = False, c=cond)
            pd_tmp = pd.DataFrame({'participant': [part] * 3,'condition': [c] * 3, 'map_number_euclid': maps_euclid, 'correlation_coefficient': maps_corr, 'muscle': ['fdi', 'ext_comm', 'sup']})
            if pd_maps.empty:
                pd_maps = pd_tmp
            else:
                pd_maps = pd.concat([pd_maps, pd_tmp], ignore_index=True)
    pd_maps['min_map_number'] = pd_maps[['map_number_euclid', 'correlation_coefficient']].max(axis=1) 

    fig, axes = plt.subplots(nrows=1, ncols=2, num=f"min_maps", sharey=True)
    for ax, c in zip(axes, cond):
        sns.barplot(x='participant', y='min_map_number', data=pd_maps.loc[pd_maps['condition'] == c], hue='muscle', ax=ax, legend=c=='pseudo')
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([str(i)[1:-3] for i in participants])
        ax.set_ylabel('Number of points')
        ax.set_xlabel('Participant')
        plt.show()
        ax.set_title(f"{c}")
        if c =='pseudo':
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=['FDI', 'Extenseur commun', 'Supinateur'])
    plt.show()
#     # plt.savefig(f"{c}_maps_euclid.png")
