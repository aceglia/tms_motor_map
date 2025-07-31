import stat
from biosiglive.file_io.save_and_load import _read_all_lines, dic_merger
import numpy as np
from pygridfit import GridFit, TiledGridFit
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

    @staticmethod
    def rolling_rms(x, N):
        xc = np.cumsum(abs(x)**2)
        return np.sqrt((xc[N:] - xc[:-N]) / N)

    def process_mep(self, peak_to_peak, baseline, mep_threeshold=25):
        for i in range(peak_to_peak.shape[0]):
            mep_values = peak_to_peak[i, peak_to_peak[i, :] * 1e6 > mep_threeshold]
            mep_values_to_exclude = mep_values.mean() + 3.5 * mep_values.std()
            peak_to_peak[i, peak_to_peak[i, :] > mep_values_to_exclude] = np.nan
            # rms_baseline = [self.rolling_rms(baseline[:, i, j], 10) for j in range(baseline.shape[2])]
            # rms_baseline_mean = [np.nanmean(rms_baseline[j]) for j in range(len(rms_baseline))]
            # rms_baseline_std = [np.nanstd(rms_baseline[j]) for j in range(len(rms_baseline))]
        return peak_to_peak

    def generate_single_map(
        self,
        mep_data,
        baseline,
        points,
        n_point_grid,
        tiled=True,
        p2p=None,
    ):
        peak_to_peak = self.process_mep(np.ptp(mep_data, axis=0), baseline, mep_threeshold) if p2p is None else p2p

        mep_threeshold = 25
        x_list, y_list, z_list = [], [], []
        x_cog_list, y_cog_list = [], []
        area_list = []
        volume_list = []
        for i in range(peak_to_peak.shape[0]):
            z = peak_to_peak[i, :]
            
            x, y = points[:, 0], points[:, 1]
            z[np.isnan(x) | np.isnan(y)] = np.nan
            x_min, x_max = np.nanmin(x), np.nanmax(x)
            y_min, y_max = np.nanmin(y), np.nanmax(y)

            xi_fit = np.linspace(x_min, x_max, n_point_grid)
            yi_fit = np.linspace(y_min, y_max, n_point_grid)

            if (np.nanmax(z) - np.nanmin(z)) != 0:
                normalized_z = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z))
            else:
                normalized_z = z
            if tiled:
                gf = TiledGridFit(
                    x,
                    y,
                    normalized_z,
                    xnodes=xi_fit,
                    ynodes=yi_fit,
                    smoothness=5,
                    interp="triangle",
                    regularizer="gradient",
                    solver="normal",
                    tilesize=120,
                    overlap=0.15,
                ).fit()
            else:
                gf = GridFit(x, y, normalized_z, xi_fit, yi_fit).fit()

            zgf = np.clip(gf.zgrid, a_min=0, a_max=gf.zgrid.max())
            xgf = gf.xgrid
            ygf = gf.ygrid
            area, volume = get_area_and_volume(xgf.flatten(), ygf.flatten(), zgf.flatten())
            x_list.append(x)
            y_list.append(y)
            z_list.append(zgf)
            x_cog, y_cog = get_cog(xgf.flatten(), ygf.flatten(), zgf.flatten())
            x_cog_list.append(x_cog)
            y_cog_list.append(y_cog)
            area_list.append(area)
            volume_list.append(volume)

        map_caracteristics = {
            "x_list": x_list,
            "y_list": y_list,
            "z_list": z_list,
            "x_cog_list": x_cog_list,
            "y_cog_list": y_cog_list,
            "area_list": area_list,
            "volume_list": volume_list,
        }
        return map_caracteristics

    # def generate_map(
    #     self,
    #     signal_data=None,
    #     position=None,
    #     stimulation_time=1,
    #     n_point_grid=50,
    #     plot=True,
    #     windows=((50, 5), (10, 50)),
    #     idx_axis_1=None,
    # ):
    #     position = position if position is not None else self.position
    #     signal_data = signal_data if signal_data is not None else self.signal_data
    #     if not isinstance(position, list):
    #         position = [position]
    #     if not isinstance(signal_data, list):
    #         signal_data = [signal_data]
    #     assert len(signal_data) == len(position)
    #     x_list_global, y_list_global, z_list_global = [], [], []
    #     for i in range(len(position)):
    #         points, idx_excluded_brainsight = self.get_local_projected_points(
    #             position[i][:, 3, :3], idx_axis_1=idx_axis_1
    #         )
    #         baseline, mep_data = self._get_baseline_mep(signal_data[i], stimulation_time, windows)
    #         # signal_data = np.delete(signal_data, idx_excluded_brainsight[0], axis=-1)
    #         # baseline = np.delete(baseline, idx_excluded_brainsight[0], axis=-1)
    #         x_list, y_list, z_list = self.generate_single_map(mep_data, baseline, points, n_point_grid)
    #         if plot:
    #             plot_map((x_list, y_list, z_list), n_point_grid)
    #         x_list_global.append(x_list)
    #         y_list_global.append(y_list)
    #         z_list_global.append(z_list)
    #     return x_list_global, y_list_global, z_list_global

    def get_local_projected_points(self, points, idx_axis_1=None):

        (x, y, z), com = get_plane_from_points(points)
        # create plane coordinates system
        local = np.array([to_plane_coordinates(p, (0, 0, 0), x, y, z) for p in points - com])

        z_threshold = 2
        mean_z = np.mean(local[:, 2])
        std_z = np.std(local[:, 2])
        idx_excluded_z = np.where(np.abs(local[:, 2]) > mean_z + z_threshold * std_z)
        mask = np.ones(points.shape[0], dtype=bool)
        mask[idx_excluded_z[0]] = False
        point_cleaned = points

        plane = get_plane_from_points(point_cleaned)
        local = np.array([to_plane_coordinates(p, (0, 0, 0), x, y, z) for p in point_cleaned - com])
        rotated_local = rotate_points(local[:, :2], idx_axis_1=idx_axis_1)
        rotated_local[:, 0] = -rotated_local[:, 0]

        return rotated_local, idx_excluded_z

    def _load_data(self, max_lines=None, idx_list=None):
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

        self.brainsight_data = [new_dict["brainsight_data"] for new_dict in self.all_data]
        self.position = [brainsight_data["position"].reshape(4, 4, -1).T for brainsight_data in self.brainsight_data]
        self.signal_data = [new_dict["signal_data"]["data"] for new_dict in self.all_data]
        # center signal data
        self.signal_data = [
            signal_data - np.mean(signal_data, axis=0, keepdims=True) for signal_data in self.signal_data
        ]
        self.target_position = [
            brainsight_data["target_position"].reshape(4, 4, -1).T for brainsight_data in self.brainsight_data
        ]
        # position, signal_data, target_position = self._exclude_data(position, signal_data, target_position)
