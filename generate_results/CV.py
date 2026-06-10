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

    maps_data = os.path.join(all_folder[0], "maps_min_map.csv")
    maps_data = pd.read_csv(maps_data)
    pd_out = pd.DataFrame(columns=["condition", "muscle", "cv"])
    for cond in maps_data.condition.unique():
        for mus in maps_data.muscle.unique():
            df_tmp = maps_data[(maps_data.condition == cond) & (maps_data.muscle == mus)]
            cv_tmp = (df_tmp.min_map_number.values.std() / df_tmp.min_map_number.values.mean()) * 100
            pd_out_tmp = pd.DataFrame({"condition": [cond], "muscle": [mus], "cv": [cv_tmp]})
            pd_out = pd.concat([pd_out, pd_out_tmp])