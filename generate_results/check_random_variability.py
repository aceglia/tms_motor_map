import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

data_dir = "smooth_8_5_batch_2"
all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and f.endswith(".csv") and 'pseudo' in f]
# file = all_files[-1]
# global_pd = pd.read_csv(os.path.join(data_dir, file))
ref_grid = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and f.endswith(".csv") and 'grid' in f]
muscle_to_keep = ['fdi', 'ext_comm', 'sup']
global_pd = pd.DataFrame()
for f, file in enumerate(all_files):
    # global_pd = pd.DataFrame()
    pd_tmp = pd.read_csv(os.path.join(data_dir, file))
    # remove columns with Unnamed inside the name
    pd_tmp.drop(columns=[col for col in pd_tmp.columns if 'Unnamed' in col], inplace=True)
    pd_tmp['batch_number'] = f
    pd_tmp = pd_tmp.loc[pd_tmp['muscle'].isin(muscle_to_keep)]
    global_pd = pd.concat([global_pd, pd_tmp], ignore_index=True)
for part in global_pd.participant.unique():
    pd_tmp = global_pd.loc[global_pd['participant'] == part].copy()
    sns.lineplot(x='map_number', y='correlation_coefficient', data=pd_tmp, hue='muscle')
    plt.show()


    # pd_grid = pd.read_csv(os.path.join(data_dir, ref_grid[0]))
    # pd_grid.drop(columns=['Unnamed: 0'], inplace=True)
    # global_pd = pd.concat([pd_tmp, pd_grid], ignore_index=True)
    # global_pd.to_csv(os.path.join(data_dir, file).replace('_pseudo', ''))

# global_pd.drop(columns=['Unnamed: 0'], inplace=True)
# pd_results = pd.DataFrame()
# for participant in global_pd['participant'].unique():
#     pd_tmp = global_pd.loc[global_pd['participant'] == participant].copy()
#     for number_map in pd_tmp['map_number'].unique():
#         pd_tmp_map = pd_tmp.loc[pd_tmp['map_number'] == number_map].copy()
#         mean = pd_tmp_map.groupby(['muscle', 'participant', 'condition', 'map_number']).mean()
#         std = pd_tmp_map.groupby(['muscle', 'participant', 'condition', 'map_number']).std()
#         mean.columns = [f"{col}_mean" for col in mean.columns]
#         std.columns = [f"{col}_std" for col in std.columns]
#         result_tmp = mean.reset_index().merge(std.reset_index(), on=['participant', 'condition','map_number','muscle'])

#         pd_results = pd.concat([pd_results, result_tmp], ignore_index=True)

# muscle_to_keep = ['fdi', 'sup', 'ext_comm']
# pd_results = pd_results.loc[pd_results['muscle'].isin(muscle_to_keep)]
# pd_results.to_csv(os.path.join(data_dir, 'all_results_pseudo_random_variability.csv'))
# # global_pd = global_pd.loc[global_pd['muscle'].isin(muscle_to_keep)]
# # # plot boxplot for each number of maps with hue being muscles
# sns.boxplot(x='map_number', y='y_cog_std', hue='muscle', data=pd_results)


