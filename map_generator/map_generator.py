from matplotlib import pyplot as plt
from scipy.io import loadmat
from biosiglive.file_io.save_and_load import _read_all_lines, dic_merger
import numpy as np
from pygridfit import GridFit, TiledGridFit
from map_generator.plot_utils import plot_single_map, plot_2d_points

from map_generator.utils import (
    get_idx_to_rotate,
    project_points_to_plane,
    get_area_and_volume,
    get_cog,
    to_plane_coordinates,
    get_plane_from_points,
    rotate_points,
)


class MapGenerator:
    def __init__(self, data_dir_path=None, data_name_base=None, trial_list=(), output_path=None, data_rate=2000):
        self.data_dir_path = data_dir_path
        self.trial_list = trial_list
        self.output_path = output_path
        self.data_name_base = data_name_base
        self.all_data = []
        self.angle = None
        self.data_rate = data_rate
        self.n_point_grid = 50
        self.mep_file_data = None
        self.map_characteristics = None

    def _get_baseline_mep(self, signal_array, stimulation_time, windows):
        center = stimulation_time * self.data_rate
        baseline_wind = windows[0]
        mep_wind = windows[1]
        baseline_frames = [
            int(center - ((baseline_wind[i] / 1000) * self.data_rate)) for i in range(len(baseline_wind))
        ]
        mep_frames = [int(center + ((mep_wind[i] / 1000) * self.data_rate)) for i in range(len(mep_wind))]
        mep_data = signal_array[mep_frames[0] : mep_frames[1], :, :]
        baseline = signal_array[baseline_frames[0] : baseline_frames[1], :, :]
        return baseline, mep_data

    def process_peaks(self, peak_to_peak, mep_threeshold=None, std_threeshold=3.5, baseline=None, lag=None):
        rms_baseline = np.sqrt(np.mean(baseline**2, axis=0))
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

        p2p_mask = (
            (rms_baseline >= lower)
            & (rms_baseline <= upper)
            & (peak_to_peak >= lower_p2p)
            & (peak_to_peak <= upper_p2p)
        )
        peak_to_peak[~p2p_mask] = replace_by

        if peak_to_peak.shape[-1] - np.sum(np.isnan(peak_to_peak)) < 4:
            peak_to_peak = np.zeros_like(peak_to_peak)
        # if lag is not None:
        #     idx_to_remove = np.argwhere(np.abs(lag) > 5)[:, 0]
        #     idx_mep = np.argwhere(peak_to_peak * 1e6 > 40)
        #     idx_tot = np.intersect1d(idx_to_remove, idx_mep)
        #     peak_to_peak[idx_tot] = replace_by
        return peak_to_peak

    def compute_map(
        self, mep_data, baseline, points, n_point_grid, tiled=True, p2p=None, pseudo=False, smoothness=None
    ):
        peak_to_peak = np.ptp(mep_data, axis=0) if p2p is None else p2p * 1e-6
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
        # if "SCI" in self.data_name_base and not pseudo:
        #     points[points[:, 1] > 34, 1] = np.nan
        for i in range(peak_to_peak.shape[0]):
            std_threeshold = 3.5
            z = self.process_peaks(
                peak_to_peak[i, :], mep_threeshold=None, std_threeshold=std_threeshold, baseline=baseline[:, i, :]
            )
            x, y = points[:, 0].copy(), points[:, 1].copy()
            mask = np.isnan(x) | np.isnan(y) | np.isnan(z)
            z[mask] = np.nan
            x[mask] = np.nan
            y[mask] = np.nan
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
            scale_value = np.nanmax(np.hstack([x, y]))
            if np.nanmax(z) != 0:
                normalized_z = z / np.nanmax(z)
                normalized_z *= scale_value
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

            zgf = np.clip(gf.zgrid, a_min=0, a_max=gf.zgrid.max()) / scale_value
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
                    xgf.flatten(), ygf.flatten(), zgf.flatten(), n_tot=self.n_point_grid**2
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

    def generate_map(
        self, stimulation_time=1, windows=([50, 5], [18, 40]), n_point_grid=50, smoothness=None, tiled=False, **kwargs
    ):
        if isinstance(self.signal_array, list):
            self._stack_data()
        self.n_point_grid = n_point_grid
        rotated_points, mep_from_file, signal_array = self.get_projected_points(**kwargs)
        baseline, mep_data = self._get_baseline_mep(signal_array, stimulation_time=stimulation_time, windows=windows)
        self.map_characteristics = self.compute_map(
            mep_data, baseline, rotated_points, n_point_grid, p2p=mep_from_file, tiled=tiled, smoothness=smoothness
        )

    def get_projected_points(self, exclude_outliers=True, **kwargs):
        points = self.position[:, 3, :3].copy()
        idx_zero = np.where(np.all(points == 0, axis=1))[0]
        target_names_tmp = self.target_names
        target_position_tmp = self.target_position
        if len(idx_zero) > 0:
            points[idx_zero] = np.nan
            signal_array = self.signal_array.copy()
            signal_array[..., idx_zero] = np.nan
            if self.mep_file_data is not None:
                mep_data_tmp = self.mep_file_data.copy()
                mep_data_tmp[..., idx_zero] = np.nan
            else:
                mep_data_tmp = None
        else:
            signal_array = self.signal_array
            mep_data_tmp = self.mep_file_data if self.mep_file_data is not None else None

        (x, y, z), com, _ = get_plane_from_points(points, to_center=None, **kwargs)
        projection = project_points_to_plane(points, z, com)
        local = np.array([to_plane_coordinates(p, com, x, y, z) for p in projection])

        if exclude_outliers:
            mean_points = np.nanmean(local, axis=0)
            std_points = np.nanstd(local, axis=0)
            mask = (abs(local[:, 0]) >= (mean_points[0] + 3 * std_points[0])) | (
                abs(local[:, 1]) >= (mean_points[1] + 3 * std_points[1])
            )
            local[mask] = np.nan

        idx_axis_1, corners = get_idx_to_rotate(target_names_tmp, local, **kwargs)
        # if None in idx_axis_1:
        #     raise ValueError(
        #         "Not enough valid points to compute the rotation. Please check the data and ensure that there are valid points for the specified corners."
        #     )

        self.projected_corners = corners
        local_changed = False
        if np.isnan(local[idx_axis_1, :].sum(axis=1)).any():
            # replace the points by the target if the points are not available.
            proj_target = project_points_to_plane(target_position_tmp[:, 3, :3], z, com)
            local_target = np.array([to_plane_coordinates(p, com, x, y, z) for p in proj_target])
            mask = local.copy()
            mask[~np.isnan(local)] = 1
            local[idx_axis_1, :] = local_target[idx_axis_1, :]
            local_changed = True
            rotated_targets = rotate_points(local_target[:, :2], idx_axis_1=idx_axis_1)

        rotated_points = rotate_points(local[:, :2], idx_axis_1=idx_axis_1)

        if local_changed:
            local_changed = False
            rotated_points *= mask[:, :2]

        if exclude_outliers:
            mean_points = np.nanmean(rotated_points, axis=0)
            std_points = np.nanstd(rotated_points, axis=0)
            mask = (abs(rotated_points[:, 0]) >= (mean_points[0] + 3 * std_points[0])) | (
                abs(rotated_points[:, 1]) >= (mean_points[1] + 3 * std_points[1])
            )
            rotated_points[mask] = np.nan

        # from map_generator.plot_utils import plot_2d_points
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, num="Projected points")

        # plot_2d_points(local, ax[0], colorized_points=(to_plot, colors))
        # # plot_2d_points(local_target, ax[1], colorized_points=(to_plot, colors))
        # # plot_2d_points(local, ax[0], colorized_points=(to_plot, colors))
        # plot_2d_points(rotated_points, ax[0], colorized_points=self.projected_corners)
        return rotated_points, mep_data_tmp, signal_array

    def _get_chan_names(self, chaninfo):
        return [str(chaninfo[i][1][0]) for i in range(chaninfo.shape[0])]

    def _load_from_wave_data(self, wave_data):
        items = list(wave_data[0][0].dtype.fields.keys())
        chanel_names = self._get_chan_names(wave_data[0][0][items.index("chaninfo")].reshape(-1))
        frames = list(range(1, wave_data[0][0][items.index("frames")][0][0] + 1))
        interval = wave_data[0][0][items.index("interval")][0][0]
        array = wave_data[0][0][items.index("values")]
        array = np.swapaxes(array, 0, -1)
        return array, chanel_names, frames, interval

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

    def _load_data(self, max_lines=None, stack_data=True):
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
            self.data_rate = 1 / (data[0]["signal_data"]["time"][1, 0] - data[0]["signal_data"]["time"][0, 0])
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
        self.signal_data = [new_dict["signal_data"] for new_dict in self.all_data]
        self.brainsight_data = [new_dict["brainsight_data"] for new_dict in self.all_data]
        self.position = [brainsight_data["position"].reshape(4, 4, -1).T for brainsight_data in self.brainsight_data]
        self.signal_array = [new_dict["signal_data"]["data"] for new_dict in self.all_data]
        # self.signal_data = data_from_mat
        # center signal data
        self.signal_array = [
            signal_array - np.mean(signal_array, axis=0, keepdims=True) for signal_array in self.signal_array
        ]
        self.target_position = [
            brainsight_data["target_position"].reshape(4, 4, -1).T for brainsight_data in self.brainsight_data
        ]
        self.target_names = [brainsight_data["target_name"].reshape(-1) for brainsight_data in self.brainsight_data]
        # position, signal_array, target_position = self._exclude_data(position, signal_array, target_position)
        if stack_data:
            self._stack_data()

    def _stack_data(self):
        idx = min(
            [si.shape[0] for si in self.signal_array]
        )  # make sure number of sample are the same. Sometime with signal it migth differ
        self.signal_array = [si[:idx] for si in self.signal_array]
        self.signal_array = self._concat(self.signal_array, axis=-1)
        self.position = self._concat(self.position, axis=0)
        self.target_position = self._concat(self.target_position, axis=0)
        self.target_names = self._concat(self.target_names, axis=0)

    @staticmethod
    def _concat(data, axis):
        if isinstance(data, list):
            data = np.concatenate(data, axis=axis)
        return data

    def from_loaded_data(self, loaded_data, stack_data=True):
        self.all_data = loaded_data
        self.brainsight_data = [new_dict["brainsight_data"] for new_dict in self.all_data]
        self.position = [brainsight_data["position"].reshape(4, 4, -1).T for brainsight_data in self.brainsight_data]
        self.signal_data = [new_dict["signal_data"] for new_dict in self.all_data]
        self.signal_array = [new_dict["signal_data"]["data"] for new_dict in self.all_data]
        # center signal data
        self.signal_array = [
            signal_array - np.mean(signal_array, axis=0, keepdims=True) for signal_array in self.signal_array
        ]
        self.target_position = [
            brainsight_data["target_position"].reshape(4, 4, -1).T for brainsight_data in self.brainsight_data
        ]
        self.target_names = [brainsight_data["target_name"].reshape(-1) for brainsight_data in self.brainsight_data]

        self.data_rate = 1 / (
            self.all_data[0]["signal_data"]["time"][1, 0] - self.all_data[0]["signal_data"]["time"][0, 0]
        )
        if stack_data:
            self._stack_data()

    def plot(self, ax=None, show=True):
        if self.map_characteristics is None:
            raise ValueError("Map characteristics not computed. Please run generate_map() first.")
        xgf_list = self.map_characteristics["xgf_list"]
        ygf_list = self.map_characteristics["ygf_list"]
        zgf_list = self.map_characteristics["zgf_list"]
        x_real = self.map_characteristics["x_list"]
        y_real = self.map_characteristics["y_list"]
        if ax is None:
            fig, ax = plt.subplots()
        for i in range(len(xgf_list)):
            xgf, ygf, zgf = xgf_list[i], ygf_list[i], zgf_list[i]
            x_real_i, y_real_i = x_real[i], y_real[i]
            plot_single_map(
                xgf,
                ygf,
                zgf,
                ax=ax,
                n_point_grid=self.n_point_grid,
                x_cog=None,
                y_cog=None,
                area=None,
                volume=None,
                x_real=x_real_i,
                y_real=y_real_i,
            )
        if show:
            plt.show()
        return ax

    def plot_projection(self, ax=None, show=True):
        rotated_points = np.array([self.map_characteristics["x_list"][0], self.map_characteristics["y_list"][0]]).T
        ax = plot_2d_points(rotated_points, ax, colorized_points=self.projected_corners)
        return ax
