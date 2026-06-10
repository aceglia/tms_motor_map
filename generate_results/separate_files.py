import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    main_dir = r"D:\Documents\Programmation\tms_motor_map\results"
    list_dir = os.listdir(main_dir)
    dirs = [
        dir
        for dir in list_dir
        if "sci" not in dir
        and os.path.isdir(os.path.join(main_dir, dir))
        and "smooth" in dir
        and "test" not in dir
        and "time" not in dir
        and "with_tmslyzer" not in dir
        and "batch" not in dir
    ]
    seeds = np.array([int(dir.split("_")[-1]) for dir in dirs])
    p_s = np.array([int(dir.split("_")[1]) for dir in dirs])
    g_s = np.array([int(dir.split("_")[2]) for dir in dirs])
    for s in np.unique(seeds):
        idxs = np.where(seeds == s)[0]
        np.unique(g_s[idxs])
        for g_smo in g_s[idxs]:
            folder_tmp = os.path.join(main_dir, "smooth_{}")


