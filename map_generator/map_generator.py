from matplotlib import pyplot as plt
from scipy.io import loadmat
from biosiglive.file_io.save_and_load import _read_all_lines, dic_merger
import numpy as np
from pygridfit import GridFit, TiledGridFit
from scipy.signal import correlate
from map_generator.utils import (
    exclude_outliers,
    get_area_and_volume,
    get_cog,
    to_plane_coordinates,
    get_plane_from_points,
    rotate_points,
)


class MapGenerator:
    def __init__(self, data_dir_path, data_name_base, trial_list=(), output_path=None, data_rate=2000):
        self.data_dir_path = data_dir_path
        self.trial_list = trial_list
        self.output_path = output_path
        self.data_name_base = data_name_base
        self.all_data = []
        self.angle = None
        self.data_rate = data_rate

    def _get_baseline_mep(self, signal_data, stimulation_time, windows):
        center = stimulation_time * self.data_rate
        baseline_wind = windows[0]
        mep_wind = windows[1]
        baseline_frames = [
            int(center - ((baseline_wind[i] / 1000) * self.data_rate)) for i in range(len(baseline_wind))
        ]
        mep_frames = [int(center + ((mep_wind[i] / 1000) * self.data_rate)) for i in range(len(mep_wind))]
        mep_data = signal_data[mep_frames[0] : mep_frames[1], :, :]
        baseline = signal_data[baseline_frames[0] : baseline_frames[1], :, :]
        return baseline, mep_data

    def process_peaks(self, peak_to_peak, mep_threeshold=None, std_threeshold=3.5, baseline=None, lag=None):
        rms_baseline = np.sqrt(np.mean(baseline ** 2, axis=0))
        mean_base = np.nanmean(rms_baseline, axis=0)
        std_baseline = np.nanstd(rms_baseline, axis=0)
        if mep_threeshold is not None:
            all_mep = peak_to_peak[peak_to_peak * 1e6 > mep_threeshold]
        else:
            all_mep = peak_to_peak
        replace_by = np.nan
        mean = np.nanmean(all_mep)
        std = np.nanstd(all_mep)        
        lower = mean_base - 2 * std_baseline  
        upper = mean_base + 2 * std_baseline
        lower_p2p = mean - std_threeshold * std
        upper_p2p = mean + std_threeshold * std

        p2p_mask = (rms_baseline >= lower) & (rms_baseline <= upper) & (peak_to_peak >= lower_p2p) & (peak_to_peak <= upper_p2p)
        peak_to_peak[~p2p_mask] = replace_by

        if peak_to_peak.shape[-1] - np.sum(np.isnan(peak_to_peak)) < 4:
            peak_to_peak = np.zeros_like(peak_to_peak)
        # if lag is not None:
        #     idx_to_remove = np.argwhere(np.abs(lag) > 5)[:, 0]
        #     idx_mep = np.argwhere(peak_to_peak * 1e6 > 40)
        #     idx_tot = np.intersect1d(idx_to_remove, idx_mep)
        #     peak_to_peak[idx_tot] = replace_by
        return peak_to_peak

    def generate_single_map(self, mep_data, baseline, points, n_point_grid, tiled=True, p2p=None, pseudo=False, smoothness=None):
        mep_threeshold = 25
        mean = np.mean(mep_data[:, :, :], axis=-1)
        # all_lag = np.ndarray((mep_data.shape[1], mep_data.shape[2]))
        # for i in range(mep_data.shape[-1]):
        #     for j in range(mep_data.shape[1]):
        #         corr = correlate(mep_data[:, j, i], mean[:, j], mode="full")
        #         # corr = correlate(mep_data[:, 0, i], mean, mode="full")
        #         lag = np.argmax(corr, axis=0) - (len(mep_data) - 1)
        #         all_lag[j, i] = (lag / self.data_rate) * 1000
        all_lag = None

        peak_to_peak = np.ptp(mep_data, axis=0) if p2p is None else p2p * 1e-6
        
        lag_thres = 5
        # compute correlation between each mep_data with the mean

        # peak_to_peak = self.process_mep(np.ptp(mep_data, axis=0), baseline, mep_threeshold) if p2p is None else p2p
        # peak_to_peak = np.ptp(mep_data, axis=0)
        # import matplotlib.pyplot as plt
        # plt.plot(peak_to_peak[0, :]*1e6)
        # mean = np.nanmean(peak_to_peak[0, peak_to_peak[0, :]*1e6 > 50] * 1e6)
        # std = np.nanstd(peak_to_peak[0, peak_to_peak[0, :]*1e6 > 50] * 1e6)
        # plt.axhline(mean + 3.5 * std, color="r")
        # plt.axhline(mean - 3.5 * std, color="r")
        # plt.show()
        # plt.plot(points[:, 0], points[:, 1])
        x_list, y_list, z_list = [], [], []
        xgf_list, ygf_list, zgf_list = [], [], []
        x_cog_list, y_cog_list = [], []
        area_list = []
        volume_list = []
        if "SCI" in self.data_name_base and not pseudo:
            points[points[:, 1] > 34, 1] = np.nan
        for i in range(peak_to_peak.shape[0]):
            # std_threeshold = 1.5 if ('P009_TN' in self.data_name_base) else 3.5
            std_threeshold = 3.5
            z = self.process_peaks(peak_to_peak[i, :], mep_threeshold=None, std_threeshold=std_threeshold, baseline=baseline[:, i, :])
            x, y = points[:, 0].copy(), points[:, 1].copy()
            z[np.isnan(x) | np.isnan(y)] = np.nan
            x[np.isnan(z)] = np.nan
            y[np.isnan(z)] = np.nan
            x_min, x_max = np.nanmin(x), np.nanmax(x)
            y_min, y_max = np.nanmin(y), np.nanmax(y)
            # x_min, y_min = -30, -30
            # x_max, y_max = 30, 30
            # x = np.clip(x, x_min, x_max)
            # y = np.clip(y, y_min, y_max)

            xi_fit = np.linspace(x_min, x_max, n_point_grid)
            yi_fit = np.linspace(y_min, y_max, n_point_grid)
            # z[np.isnan(z)] = 0
            # if (np.nanmax(z) - np.nanmin(z)) != 0:
            #     normalized_z = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z))
            #     normalized_z *= 30
            if np.nanmax(z) != 0:
                normalized_z = z / np.nanmax(z)
                normalized_z *= 30
            # elif np.nanmax(z) != 0:
            #     normalized_z =  ( z / np.nanmax(z) ) * 30
            # else:
            #     normalized_z = z 
            # normalized_z = z * 1e6
            # to_divide = 1 if np.nanmax(z) == 0 else np.nanmax(z)
            # normalized_z = z / to_divide
            smoothness = smoothness if smoothness is not None else 5
            if tiled:
                gf = TiledGridFit(
                    x,
                    y,
                    normalized_z,
                    xnodes=xi_fit,
                    ynodes=yi_fit,
                    smoothness=smoothness,
                    interp="triangle",
                    regularizer="gradient",
                    solver="normal",
                    tilesize=150,
                    overlap=0.35,
                ).fit()
            else:
                gf = GridFit(
                    x,
                    y,
                    normalized_z,
                    xi_fit,
                    yi_fit,
                    extend="never",
                    smoothness=smoothness,
                    interp="nearest",
                    regularizer="gradient",
                    solver="normal",
                    autoscale="on",
                ).fit()
            
            zgf = np.clip(gf.zgrid, a_min=0, a_max=gf.zgrid.max()) / 30
            # factor = 1e6 
            # zgf = (zgf - np.nanmin(z * factor)) / (np.nanmax(z * factor) - np.nanmin(z * factor)) if np.nanmax(gf.zgrid) - np.nanmin(gf.zgrid) != 0 else zgf

            xgf = gf.xgrid
            ygf = gf.ygrid

            # fig = plt.figure(figsize=(10, 6))
            # ax = fig.add_subplot(111, projection='3d') # Use projection='3d'
            # surf = ax.plot_surface(xgf, ygf, zgf, cmap='viridis', edgecolor='none')
            # plt.show()
            if np.all(zgf == 0):
                area, volume = 0, 0
                x_cog, y_cog = 0, 0
            else:
                area, volume = get_area_and_volume(
                    xgf.flatten(),
                    ygf.flatten(),
                    zgf.flatten(),
                )
                x_cog, y_cog = get_cog(xgf.flatten(), ygf.flatten(), zgf.flatten())

            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
            xgf_list.append(xgf)
            ygf_list.append(ygf)
            zgf_list.append(zgf)
            x_cog_list.append(x_cog)
            y_cog_list.append(y_cog)
            area_list.append(area)
            volume_list.append(volume)

        map_caracteristics = {
            "x_list": x_list,
            "y_list": y_list,
            "z_list": z_list,
            "xgf_list": xgf_list,
            "ygf_list": ygf_list,
            "zgf_list": zgf_list,
            "x_cog_list": x_cog_list,
            "y_cog_list": y_cog_list,
            "area_list": area_list,
            "volume_list": volume_list,
        }
        return map_caracteristics

    def get_local_projected_points(self, points, idx_axis_1=None):
        (x, y, z), com = get_plane_from_points(points)
        # create plane coordinates system
        local = np.array([to_plane_coordinates(p, com, x, y, z) for p in points])

        # z_threshold = 2
        # mean_z = np.mean(local[:, 2])
        # std_z = np.std(local[:, 2])
        # idx_excluded_z = np.where(np.abs(local[:, 2]) > mean_z + z_threshold * std_z)
        # mask = np.ones(points.shape[0], dtype=bool)
        # mask[idx_excluded_z[0]] = False
        point_cleaned = points

        plane = get_plane_from_points(point_cleaned)
        local = np.array([to_plane_coordinates(p, (0, 0, 0), x, y, z) for p in point_cleaned - com])
        rotated_local = rotate_points(local[:, :2], idx_axis_1=idx_axis_1)
        rotated_local[:, 0] = -rotated_local[:, 0]

        return rotated_local, idx_excluded_z

    def _get_chan_names(self, chaninfo):
        return [str(chaninfo[i][1][0]) for i in range(chaninfo.shape[0])]
    
    def _load_from_wave_data(self, wave_data):
        items = list(wave_data[0][0].dtype.fields.keys())
        chanel_names = self._get_chan_names(wave_data[0][0][items.index("chaninfo")].reshape(-1))
        frames = list(range(wave_data[0][0][items.index("frames")][0][0]))
        array = wave_data[0][0][items.index("values")]
        array = np.swapaxes(array, 0, -1)
        return array, chanel_names, frames
    
    def load_mat_file(self, path):
        try:
            mat_file = loadmat(path)
        except:
            raise ValueError("Not able to load the .mat file. Try exporting in version 6 or lower of matlab.")           
        wave_data = [key for key in mat_file.keys() if "wave_data" in key][0]

        if len(wave_data) > 0:
            return self._load_from_wave_data(mat_file[wave_data])
        
        else:
            raise ValueError("No recognized data found in the .mat file.")

    def _load_data(self, max_lines=None, idx_list=None):
        data_from_mat = []
        for file_name in self.data_path_list:
            data = _read_all_lines(file_name, data=[], merge=False)
            if max_lines:
                data = data[:max_lines]
            min_signal_shape = min(
                [d["signal_data"]["data"].shape[0] for d in data if d["signal_data"]["data"].shape[0] > 2000]
            )
            for d in data:
                d["signal_data"]["time"] = d["signal_data"]["time"][:min_signal_shape]
                d["signal_data"]["data"] = d["signal_data"]["data"][:min_signal_shape, :, None]
                d["brainsight_data"]["position"] = np.array(d["brainsight_data"]["position"])[:, None]
                d["brainsight_data"]["target_position"] = np.array(d["brainsight_data"]["target_position"])[:, None]
                d["brainsight_data"]["target_name"] = np.array([d["brainsight_data"]["target_name"]])[:, None]

            # idx_wrong = [i for i, d in enumerate(data) if d["signal_data"]["data"].shape[0] != min_signal_shape]
            # if idx_wrong:
            #     shape_value = data[idx_wrong[0]]["signal_data"]["data"].shape[0]
            #     to_fill = min_signal_shape - shape_value
            #     data[idx_wrong[0]]["signal_data"]["data"] = np.concatenate([data[idx_wrong[0]]["signal_data"]["data"], np.zeros((to_fill, 6, 1))], axis=0)

            new_dict = None
            for d in data:
                new_dict = dic_merger(d, new_dict)
            self.all_data.append(new_dict)

            ## read from mat file instead of pkl file saved from txt file exported via signal in live
            # mat_file = file_name.replace('.pkl', '.mat').replace('data_trial_', '')
            # mat_data = self.load_mat_file(mat_file)
            # frames_mat = [v + 1 for v in mat_data[-1]]
            # values = np.swapaxes(mat_data[0], 0, -1)
            # synch_data_path = file_name.replace('data_trial_', 'synch_trial_').replace('.pkl', '.txt')
            # with open(synch_data_path, 'r') as f:
            #     frames = [int(row[row.find('Frame ') + len('Frame '):row.find(';')]) for row in f]
            # reordered_values = np.zeros((values.shape[0], values.shape[1], len(frames)))
            # for i, frame in enumerate(frames):
            #     idx = frames_mat.index(frame)
            #     reordered_values[..., i] = values[:, :, idx]
            # reordered_values = reordered_values[..., ::-1]
            # data_from_mat.append(reordered_values)

        self.brainsight_data = [new_dict["brainsight_data"] for new_dict in self.all_data]
        self.position = [brainsight_data["position"].reshape(4, 4, -1).T for brainsight_data in self.brainsight_data]
        self.signal_data = [new_dict["signal_data"]["data"] for new_dict in self.all_data]
        # self.signal_data = data_from_mat
        # center signal data
        self.signal_data = [
            signal_data - np.mean(signal_data, axis=0, keepdims=True) for signal_data in self.signal_data
        ]
        self.target_position = [
            brainsight_data["target_position"].reshape(4, 4, -1).T for brainsight_data in self.brainsight_data
        ]
        # position, signal_data, target_position = self._exclude_data(position, signal_data, target_position)
