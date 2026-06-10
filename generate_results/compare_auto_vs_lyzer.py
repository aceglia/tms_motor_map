import numpy as np
import pandas as pd
import os
from biosiglive import load
from utils import cosine_similarity
from scipy.stats import pearsonr

if __name__ == '__main__':
    lyzer_dir = r"D:\Documents\Programmation\tms_motor_map\results\smooth_6_6_0_ransac_tmslyzer"
    ransac_dir = r"D:\Documents\Programmation\tms_motor_map\results\smooth_6_6_0_ransac"
    data_lyzer = load(os.path.join(lyzer_dir, "maps_values.bio"), merge=False)
    data_ransac = load(os.path.join(ransac_dir, "maps_values.bio"), merge=False)
    lyzer_data_frame = pd.DataFrame()
    for i in range(len(data_lyzer)):
        data_frame_tmp = pd.DataFrame(
            {
                "participant": data_lyzer[i]["participant"] * 5,
                "map_number": data_lyzer[i]["map_number"] * 5,
                "x_list": data_lyzer[i]["x_list"],
                "y_list": data_lyzer[i]["y_list"],
                "zgf_list": data_lyzer[i]["zgf_list"],
                "condition": data_lyzer[i]["condition"] * 5,
                "xgf_list": data_lyzer[i]["xgf_list"],
                "ygf_list": data_lyzer[i]["ygf_list"],
                "muscle": list(data_lyzer[i]["muscle"][:5]),
            }
        )
        if lyzer_data_frame.empty:
            lyzer_data_frame = data_frame_tmp
        else:
            lyzer_data_frame = pd.concat([lyzer_data_frame, data_frame_tmp])
    ransac_data_frame = pd.DataFrame()
    for i in range(len(data_ransac)):
        data_frame_tmp = pd.DataFrame(
            {
                "participant": data_ransac[i]["participant"] * 5,
                "map_number": data_ransac[i]["map_number"] * 5,
                "x_list": data_ransac[i]["x_list"],
                "y_list": data_ransac[i]["y_list"],
                "zgf_list": data_ransac[i]["zgf_list"],
                "condition": data_ransac[i]["condition"] * 5,
                "xgf_list": data_ransac[i]["xgf_list"],
                "ygf_list": data_ransac[i]["ygf_list"],
                "muscle": list(data_ransac[i]["muscle"][:5]),
            }
        )
        if ransac_data_frame.empty:
            ransac_data_frame = data_frame_tmp
        else:
            ransac_data_frame = pd.concat([ransac_data_frame, data_frame_tmp])
        
    lyzer_data_frame = lyzer_data_frame.loc[lyzer_data_frame["participant"] != '006_TN']
    muscle_list = list(lyzer_data_frame['muscle'][:3])
    lyzer_data_frame = lyzer_data_frame.loc[lyzer_data_frame['muscle'].isin(muscle_list)]
    ransac_data_frame = ransac_data_frame.loc[ransac_data_frame["participant"] != '006_TN']
    ransac_data_frame = ransac_data_frame.loc[ransac_data_frame['muscle'].isin(muscle_list)]
    cos_sim = []
    pearson = []
    for i in range(len(ransac_data_frame.zgf_list)):
        try:
            cos_sim.append(cosine_similarity(ransac_data_frame.zgf_list.iloc[i], lyzer_data_frame.zgf_list.iloc[i]))
            pearson.append(pearsonr(ransac_data_frame.zgf_list.iloc[i].flatten(), lyzer_data_frame.zgf_list.iloc[i].flatten())[0])
        except:
            continue
    print(np.nanmean(cos_sim), np.nanstd(cos_sim))
    print(np.nanmean(pearson), np.nanstd(pearson))

