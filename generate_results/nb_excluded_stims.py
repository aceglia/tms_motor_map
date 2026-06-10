from biosiglive import load
import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sci = False
    result_dir = (
        r"D:\Documents\Programmation\tms_motor_map\results\smooth_8_4_5" if not sci else "sci_smooth_10_5_2543"
    )
    data_frame = pd.read_csv(os.path.join(result_dir, "maps_characteristics.csv"))

    data_maps = load(os.path.join(result_dir, "maps_values.bio"), merge=False)[-144:]
    data_frame_tot = pd.DataFrame()
    for i in range(len(data_maps)):
        data_frame_tmp = pd.DataFrame(
            {
                "nb_excluded_stims": [np.count_nonzero(np.isnan(da)) / da.shape[0] * 100 for da in data_maps[i]["z_list"]],
                "participant": data_maps[i]["participant"] * 5,
                "map_number": data_maps[i]["map_number"] * 5,
                "condition": data_maps[i]["condition"] * 5,
                "muscle": list(data_frame["muscle"][:5]),
            }
        )
        data_frame_tot = pd.concat([data_frame_tot, data_frame_tmp])
    data_frame_tot = data_frame_tot.loc[data_frame_tot.muscle.isin(data_frame_tot.muscle.unique()[:3])]
    data_frame_tot.describe()
    sns.boxplot(x="condition", y="nb_excluded_stims", data=data_frame_tot)
    plt.show()
    