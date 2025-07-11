from map_generator.map_generator import MapGenerator
import os
# import matplotlib.pyplot as plt

# from plot_utils import plot_2d_points, plot_single_map, plot_heatmap
# from utils import Participant, get_mep_from_excel, get_plane_from_points, rotate_points, to_plane_coordinates

# import numpy as np


class GridBasedGenerator(MapGenerator):
    def __init__(self, data_dir_path, data_name_base, trial_list, output_path=None, data_rate=2000):
        super().__init__(data_dir_path, data_name_base, output_path, data_rate)
        self.data_rate = data_rate
        self.data_name_list = [f"{data_name_base}{trial}.pkl" for trial in trial_list]
        self.data_path_list = [os.path.join(data_dir_path, data_name) for data_name in self.data_name_list]
        self.n_map = len(trial_list)
        self._load_data()


# if __name__ == "__main__":
#     participant = Participant("NH")
#     mep_data, frame_file = participant.excel_mep(pseudo_trial=False)
#     mep_data = mep_data
#     data_dir = participant.return_dir_path()
#     file_name = participant.return_pkl_file_base()
#     trials = participant.return_trials(pseudo=False)

#     p2p_from_file = True
#     map_gen = GridBasedGenerator(data_dir, file_name, trials, data_rate=2148)

#     if mep_data is not None:
#         mep_data = [mep_data[i][:, -map_gen.signal_data[i].shape[-1] :] for i in range(len(map_gen.signal_data))]

#     signal_data = map_gen.signal_data
#     idx = min([si.shape[0] for si in signal_data])
#     signal_mat = [si[:idx, ...] * 1e6 for si in signal_data]
#     position_mat = map_gen.position
#     to_concat = list(range(1, len(trials) + 1))
#     signal_mat = [np.concatenate(([si[:idx, ...] for si in signal_data[:c]]), axis=-1) for c in to_concat]
#     position_mat = [np.vstack(map_gen.position[:c]) for c in to_concat]

#     if p2p_from_file and mep_data is not None:
#         mep_data_mat = [np.hstack([si for si in mep_data[:c]]) for c in to_concat]
#     else:
#         mep_data_mat = [None for _ in range(len(signal_mat))]

#     # signal_mat = mep_data_file
#     target_names = map_gen.brainsight_data[0]["target_name"]
#     grid_name = map_gen.brainsight_data[0]["target_name"][0][0].split(" ")[0]
#     idx_axis_1 = (
#         np.where(target_names == f"{grid_name} (6, 0)")[1][0],
#         np.where(target_names == f"{grid_name} (0, 0)")[1][0],
#     )
#     to_plot = (
#         np.where(target_names == f"{grid_name} (6, 0)")[1][0],
#         np.where(target_names == f"{grid_name} (0, 0)")[1][0],
#         np.where(target_names == f"{grid_name} (0, 5)")[1][0],
#         np.where(target_names == f"{grid_name} (6, 5)")[1][0],
#     )
#     colors = ["r", "g", "b", "c"]
#     fig, ax = plt.subplots(len(signal_mat), signal_mat[0].shape[1], num=f"Maps grid", sharey=True, sharex=True)
#     for i in range(len(signal_mat)):
#         points = position_mat[i][:, 3, :3]
#         signal_data = signal_mat[i]
#         if i == 0:
#             (x, y, z), com = get_plane_from_points(points)
#         y = -y if z[0] > 0 else y
#         z = -z if z[0] > 0 else z
#         # plot_plane(points, com, z, f'3D map {i}')
#         local = np.array([to_plane_coordinates(p, (0, 0, 0), x, y, z) for p in points - com])
#         rotated_points = rotate_points(local[:, :2], idx_axis_1=idx_axis_1, additional_rot=0)
#         # rotated_points = local
#         baseline, mep_data = map_gen._get_baseline_mep(signal_data, stimulation_time=1, windows=([50, 5], [14, 50]))
#         # signal_data = np.delete(signal_data, idx_excluded_brainsight[0], axis=-1)
#         # baseline = np.delete(baseline, idx_excluded_brainsight[0], axis=-1)
#         map_caracteristics = map_gen.generate_single_map(
#             mep_data, baseline, rotated_points, 50, p2p=mep_data_mat[i], exclude_outliers_mep=False, tiled=True
#         )
#         x_list, y_list, z_list = (
#             map_caracteristics["x_list"],
#             map_caracteristics["y_list"],
#             map_caracteristics["z_list"],
#         )
#         x_cog, y_cog = map_caracteristics["x_cog_list"], map_caracteristics["y_cog_list"]
#         area, volume = map_caracteristics["area_list"], map_caracteristics["volume_list"]
#         for j in range(len(x_list)):
#             # plot_2d_points(local, ax[0, j], colorized_points=(to_plot, colors))
#             # plot_2d_points(rotated_points, ax[1, j], colorized_points=(to_plot, colors))
#             # plot_heatmap(rotated_points, mep_data[:, j, :], ax[2, j])
#             plot_single_map(x_list[j], y_list[j], z_list[j], ax[i, j], 50, x_cog[j], y_cog[j], area[j], volume[j])
#         _ = [a.set_aspect("equal", adjustable="box") for a in ax.flatten()]
#     plt.show()
    # position = mapp_gen.position
    # signal_data = [d[2000:2400, 1:, :] for d in mapp_gen.signal_data]
    # baseline = [d[1900:1990, 1:, :] for d in mapp_gen.signal_data]

    # trans = [p[:, 3, :3] for p in position]
    # trans.append(np.concatenate(trans, axis=0))
    # signal_data.append(np.concatenate(signal_data, axis=-1))
    # baseline.append(np.concatenate(baseline, axis=-1))
    # planes = [MapGenerator.get_plane_from_points(p) for p in trans]
    # # _ = [mapp_gen.plot_3D_points(t, plane = planes[idx], show=True) for idx,  t in enumerate(trans)]

    # points = [mapp_gen.get_local_projected_points(trans[idx], planes[idx]) for idx in range(len(trans))]
    # _ = [mapp_gen.generate_map(signal_data[idx], baseline[idx], points[idx]) for idx in range(len(trans))]

    # plt.show()
