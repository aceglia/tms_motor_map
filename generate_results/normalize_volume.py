from biosiglive import load
import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sci = True
    result_dir = (
        r"D:\Documents\Programmation\tms_motor_map\results\smooth_6_6_0_ransac" if not sci else r"D:\Documents\Programmation\tms_motor_map\results\smooth_6_6_0_ransac_sci"
    )
    data_frame = pd.read_csv(os.path.join(result_dir, "maps_characteristics.csv"))

    participants = data_frame.participant.unique()
    for part in participants:
        volume = data_frame.loc[data_frame['participant'] == part]['volume']
        normalize_volume = volume.values / volume.max()
        data_frame.loc[data_frame['participant'] == part, 'normalize_volume'] = normalize_volume

    data_frame.to_csv(os.path.join(result_dir, "maps_characteristics.csv"), index=False)