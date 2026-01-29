import os 
from requests import get
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data(data_dir):
    file = os.path.join(data_dir, 'maps_characteristics.csv')
    muscle_to_keep = ['ext_comm', 'sup']
    number = file.removesuffix('.csv').split('_')[-1]
    pd_tmp = pd.read_csv(file)
    pd_tmp.drop(columns='Unnamed: 0')
    pd_tmp.loc[pd_tmp['condition'] == 'grid', 'map_number'] = (pd_tmp.loc[pd_tmp['condition'] == 'grid', 'map_number'].values + 1) * 49
    list_number = [24, 44, 64, 94, 124, 154, 184]
    pd_tmp.loc[pd_tmp['condition'] == 'pseudo', 'map_number'] = pd_tmp.loc[pd_tmp['condition'] == 'pseudo', 'map_number'].apply(lambda x: list_number[x])
    data_frame = pd_tmp.loc[pd_tmp['muscle'].isin(muscle_to_keep)]
    min_map_number= {'grid':[196, 196], 'pseudo':[184, 184]}
    min_map_number = {'grid':[196, 196], 'pseudo':[44, 64]}
    pd_tot = pd.DataFrame()
    for k, key in enumerate(min_map_number.keys()):
        for m, mus in enumerate(muscle_to_keep):
            pd_tmp_mus = data_frame.loc[(data_frame['condition'] == key) & (data_frame['muscle'] == mus) & (data_frame['map_number'] == min_map_number[key][m])]
            pd_tot = pd.concat([pd_tot, pd_tmp_mus], ignore_index=True)
    # add a column that is x_cog + y_cog
    pd_tot['cog'] = pd_tot['x_cog'] + pd_tot['y_cog']
    return pd_tot


def compute_diff_pd(pd_tot, rating):
    pd_diff = pd.DataFrame()
    for cond in ['grid', 'pseudo']:
        pd_tmp = pd_tot.loc[pd_tot['condition'] == cond]
        diff = pd_tmp.loc[pd_tmp['muscle'] == 'ext_comm', rating].values - pd_tmp.loc[pd_tmp['muscle'] =='sup', rating].values
        pd_to_concat = pd.DataFrame({'muscle': ['ext_comm-sup'] * len(diff),'muscle_diff': diff, 'condition': [cond]* len(diff)})
        pd_diff = pd.concat([pd_diff, pd_to_concat], ignore_index=True)
    return pd_diff

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

def compute_min_map(data_dir, participants, condition):
    file = os.path.join(data_dir, 'maps_characteristics.csv')
    pd_tmp = pd.read_csv(file)
    muscle_to_keep = ['ext_comm', 'sup']
    pd_tmp = pd_tmp.loc[pd_tmp['muscle'].isin(muscle_to_keep)]
    pd_tmp.drop(columns='Unnamed: 0')
    pd_maps = pd.DataFrame()
    for part in participants:
        for c in condition:
            maps_euclid = min_map(pd_tmp.loc[pd_tmp['condition'] == c], part, euclid = True)
            maps_corr = min_map(pd_tmp.loc[pd_tmp['condition'] == c], part, euclid = False)
            pd_tmp = pd.DataFrame({'participant': [part] * 2,'condition': [c] * 2, 'map_number_euclid': maps_euclid, 'correlation_coefficient': maps_corr, 'muscle': pd_tmp.muscle.unique()})
            if pd_maps.empty:
                pd_maps = pd_tmp
            else:
                pd_maps = pd.concat([pd_maps, pd_tmp], ignore_index=True)
    pd_maps['min_map_number'] = pd_maps[['map_number_euclid', 'correlation_coefficient']].max(axis=1) 


if __name__ == '__main__':
    data_dir_sci = "sci_smooth_10_5_2543"
    data_dir = 'smooth_10_5_2543' 

    pd_sci = get_data(data_dir_sci)
    pd_parts = get_data(data_dir)
    pd_sci = compute_min_map(data_dir_sci, pd_sci.participant.unique(), ['grid', 'pseudo'])
    pd_parts = compute_min_map(data_dir, pd_parts.participant.unique(), ['grid', 'pseudo'])
    # set font svg to None
    plt.rcParams['svg.fonttype'] = 'none'
    rating = ['x_cog', 'y_cog', 'area', 'volume']
    y_label = {'x_cog': 'COG AP (mm)','y_cog': 'COG ML (mm)', 'area': 'Area  (mm²)', 'volume': 'Normalized volume'}
    x_label = ['Extensor digitorum', 'Supinator']
    x_label = ['Grid', 'Pseudo']
    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    axes = ax.flatten()
    for rate in rating: 
        ax = axes[rating.index(rate)]
        pd_tmp_rate_sci = pd_sci[[rate, 'muscle', 'condition']] #.loc[pd_sci['condition'] == key]
        pd_tmp_rate = pd_parts[[rate, 'muscle', 'condition']] #.loc[pd_parts['condition'] == key]
        pd_dif_sci = compute_diff_pd(pd_tmp_rate_sci, rate)
        pd_dif = compute_diff_pd(pd_tmp_rate, rate) 
        # sns.lineplot(x='muscle', y=rate, data=pd_tmp_rate.loc[pd_tmp_rate['condition'] == 'grid'], ax=ax)
        # sns.pointplot(x='muscle', y=rate, hue='condition', data=pd_tmp_rate_sci, ax=ax)
        # sns.lineplot(x='condition', y=rate, data=pd_tmp_rate.loc[pd_tmp_rate['condition'] == 'grid'], ax=ax)
        sns.pointplot(x='condition', y=rate, data=pd_tmp_rate_sci, ax=ax)
        ax.margins(x=0.1)
        # sns.lineplot(x='condition', y='muscle_diff',  data=pd_dif, ax=ax)
        # sns.pointplot(x='condition', y='muscle_diff',  data=pd_dif_sci, ax=ax)
        ax.set_ylabel(y_label[rate])
        ax.set_xticklabels(x_label)
    plt.show()
