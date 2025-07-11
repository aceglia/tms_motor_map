from map_generator.map_generator import MapGenerator
import os
# import matplotlib.pyplot as plt

# from plot_utils import plot_2d_points, plot_map, plot_plane, plot_single_map
# from utils import (
#     Participant,
#     exclude_outliers,
#     get_mep_from_excel,
#     get_plane_from_points,
#     get_random_points,
#     rotate_points,
#     to_plane_coordinates,
# )



class PseudoRandomGenerator(MapGenerator):
    def __init__(self, data_dir_path, data_name_base, list_points_idx=(), output_path=None, data_rate=2000, **kwargs):
        super().__init__(self, data_dir_path, data_name_base, output_path, data_rate)
        self.data_rate = data_rate
        self.data_path_list = [os.path.join(data_dir_path, data_name_base)]
        self.n_map = len(list_points_idx)
        self._load_data(idx_list=list_points_idx)


# if __name__ == "__main__":
#     participant = Participant("NH")
#     mep_data, frame_file = participant.excel_mep(pseudo_trial=True)
#     data_dir = participant.return_dir_path()
#     file_name = participant.return_pkl_file_name()
#     p2p_from_file = True
#     map_gen = PseudoRandomGenerator(data_dir, file_name, data_rate=2148)

#     if mep_data is not None:
#         mep_data = [mep_data[i][:, :] for i in range(len(map_gen.signal_data))]

#     nb_stim_list = [24, 64, 94, 124, 154, map_gen.signal_data[0].shape[-1]]
#     rdm_points = get_random_points(nb_stim_list, nb_total_points=map_gen.signal_data[0].shape[-1] - 4)
#     colors = ["r", "g", "b", "c"]
#     target_names = map_gen.brainsight_data[0]["target_name"]
#     target_names_roll = np.roll(target_names, 4)
#     grid_name = map_gen.brainsight_data[0]["target_name"][0][0].split(" ")[0]
#     idx_limit_map = (
#         np.where(target_names_roll == f"{grid_name} (6, 0)")[1][0],
#         np.where(target_names_roll == f"{grid_name} (0, 0)")[1][0],
#         np.where(target_names_roll == f"{grid_name} (0, 5)")[1][0],
#         np.where(target_names_roll == f"{grid_name} (6, 5)")[1][0],
#     )

#     signal_data_roll = np.roll(map_gen.signal_data[0], 4, axis=-1)
#     position_roll = np.roll(map_gen.position[0], 4, axis=0)

#     idx_axis_1 = idx_limit_map[:2]
#     signal_mat = [signal_data_roll[..., rdm_points[i]] for i in range(len(rdm_points))]
#     position_mat = [position_roll[rdm_points[i]] for i in range(len(rdm_points))]

#     if p2p_from_file and mep_data is not None:
#         mep_data_file_roll = np.roll(mep_data[0], 4, axis=-1)
#         mep_data_mat = [mep_data_file_roll[..., rdm_points[i]] for i in range(len(rdm_points))]
#     else:
#         mep_data_mat = [None for _ in range(len(signal_mat))]

#     to_plot = idx_limit_map
#     fig, ax = plt.subplots(len(signal_mat),
#                             signal_mat[0].shape[1], num=f"Pseudo grid", sharey=True, sharex=True)

#     for i in range(len(signal_mat)):
#         points = position_mat[i][:, 3, :3]
#         signal_data = signal_mat[i]
#         (x, y, z), com = get_plane_from_points(points)
#         y = -y if z[0] > 0 else y
#         z = -z if z[0] > 0 else z
#         local = np.array([to_plane_coordinates(p, (0, 0, 0), x, y, z) for p in points - com])
#         rotated_points = rotate_points(local[:, :2], idx_axis_1=idx_axis_1, additional_rot=0)
#         baseline, mep_data = map_gen._get_baseline_mep(signal_data, stimulation_time=1, windows=([50, 5], [10, 80]))
#         # signal_data = np.delete(signal_data, idx_excluded_brainsight[0], axis=-1)
#         # baseline = np.delete(baseline, idx_excluded_brainsight[0], axis=-1)
#         map_caracteristics = map_gen.generate_single_map(
#             mep_data,
#             baseline,
#             rotated_points,
#             50,
#             p2p=mep_data_mat[i],
#             idx_limit_map=idx_limit_map,
#             exclude_outliers_mep=False,
#         )
#         x_list, y_list, z_list = (
#             map_caracteristics["x_list"],
#             map_caracteristics["y_list"],
#             map_caracteristics["z_list"],
#         )
#         x_cog, y_cog = map_caracteristics["x_cog_list"], map_caracteristics["y_cog_list"]
#         area, volume = map_caracteristics["area_list"], map_caracteristics["volume_list"]
#         # fig, ax = plt.subplots(3, len(x_list), num=f"Map number {i}")
#         for j in range(len(x_list)):
#             # plot_2d_points(local, ax[0, j], colorized_points=(to_plot, colors))
#             # plot_2d_points(rotated_points, ax[1, j], colorized_points=(to_plot, colors))
#             plot_single_map(x_list[j], y_list[j], z_list[j], ax[i, j], 50, x_cog[j], y_cog[j], area[j], volume[j])
#         _ = [a.set_aspect("equal", adjustable="box") for a in ax.flatten()]
#     plt.show()
