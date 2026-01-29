import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

if __name__ == '__main__':
    data_dir = 'smooth_10_5_time'
    pd_data = pd.read_csv(os.path.join(data_dir, 'maps_characteristics.csv'))
    pd_data = pd_data.drop(columns=['Unnamed: 0'])
    time = pd_data['time_to_compute']
    mean_time = np.mean(time)
    std_time = np.std(time)
    print(f"Mean time to compute: {mean_time:.2f}s")
