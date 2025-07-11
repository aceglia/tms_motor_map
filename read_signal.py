import csv
import time
import numpy as np
import os


def read_signal_file(file_name):
        while True:
            try:
                rows = []
                with open(file_name, "r") as file:
                    reader = csv.reader(file, delimiter="\t")
                    headers_ok = False
                    for row in reader:
                        if 's' in row:
                             headers = row
                             headers_ok = True
                        elif headers_ok:
                            rows.append(row)
                
                file_info = file_name.split(os.sep)[-1]
                name = file_info.split(".")[0]
                frame = f'Frame {int(file_info.split("_")[-1].split(".")[0])}'
                dict = {
                    "file_name": name,
                    "frame_number": frame,
                    "state": None,
                    "chanel_names": headers,
                    "time": np.array(rows, dtype=np.float64)[:, 0:1],
                    "data": np.array(rows, dtype=np.float64)[:, 1:],
                }
                break
            except:
                time.sleep(0.001)
        return dict

if __name__ == '__main__':
    read_signal_file(r'D:\Documents\Udem\Postdoctorat\Projet transfert nerveux\data\test_NH_001\test_mapping_NH001.data_frame_1.txt')