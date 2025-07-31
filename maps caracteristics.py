import matplotlib.pyplot as plt
import pandas as pd
from map_generator.plot_utils import plot_2d_points, plot_single_map
from map_generator.utils import (
    Participant,
    get_random_points,
    get_plane_from_points,
    rotate_points,
    to_plane_coordinates,
)
from map_generator.grid_based_generator import GridBasedGenerator
from map_generator.pseudo_random_generator import PseudoRandomGenerator
from scipy.stats import pearsonr
import numpy as np


def get_data_from_pseudo(map_gen, mep_data=None):
    nb_stim_list = [24, 64, 94, 124, 154, map_gen.signal_data[0].shape[-1]]
    rdm_points = get_random_points(nb_stim_list, nb_total_points=map_gen.signal_data[0].shape[-1] - 4)
    target_names = map_gen.brainsight_data[0]["target_name"]
    target_names_roll = np.roll(target_names, 4)
    signal_data_roll = np.roll(map_gen.signal_data[0], 4, axis=-1)
    position_roll = np.roll(map_gen.position[0], 4, axis=0)
    target_position = np.roll(map_gen.target_position[0], 4, axis=0)
    signal_mat = [signal_data_roll[..., rdm_points[i]] for i in range(len(rdm_points))]
    position_mat = [position_roll[rdm_points[i]] for i in range(len(rdm_points))]
    target_position_mat = [target_position[rdm_points[i]] for i in range(len(rdm_points))]
    if p2p_from_file and mep_data is not None:
        mep_data_file_roll = np.roll(mep_data[0], 4, axis=-1)
        mep_data_mat = [mep_data_file_roll[..., rdm_points[i]] for i in range(len(rdm_points))]
    else:
        mep_data_mat = [None for _ in range(len(signal_mat))]

    return target_names_roll, position_mat, signal_mat, mep_data_mat, target_position_mat


def get_data_from_grid(map_gen, trials, mep_data=None):
    to_concat = list(range(1, len(trials) + 1))
    target_names = map_gen.brainsight_data[0]["target_name"]
    idx = min([si.shape[0] for si in map_gen.signal_data])
    signal_mat = [np.concatenate(([si[:idx, ...] for si in map_gen.signal_data[:c]]), axis=-1) for c in to_concat]
    position_mat = [np.vstack(map_gen.position[:c]) for c in to_concat]
    target_position_mat = [np.vstack(map_gen.target_position[:c]) for c in to_concat]

    if p2p_from_file and mep_data is not None:
        mep_data_mat = [np.hstack([si for si in mep_data[:c]]) for c in to_concat]
    else:
        mep_data_mat = [None for _ in range(len(signal_mat))]
    return target_names, position_mat, signal_mat, mep_data_mat, target_position_mat


def get_data(map_gen, mep_data, trial_list=None, pseudo=False):
    if pseudo:
        return get_data_from_pseudo(map_gen, mep_data)
    return get_data_from_grid(map_gen, trial_list, mep_data)


def compute_maps(participant, pseudo=False, data_rate=2148, p2p_from_file=False):
    map_instance = PseudoRandomGenerator if pseudo else GridBasedGenerator
    file_name = participant.return_pkl_file_name() if pseudo else participant.return_pkl_file_base()
    map_gen = map_instance(
        participant.return_dir_path(), file_name, data_rate=data_rate, trial_list=participant.return_grid_trials()
    )

    mep_data, frame_file = participant.excel_mep(pseudo_trial=pseudo)

    if mep_data is not None:
        mep_data = [mep_data[i][:, -map_gen.signal_data[0].shape[-1] :] for i in range(len(map_gen.signal_data))]
    else:
        if p2p_from_file:
            print("WARNING: Peak to peak ask from file but none was found.")

    target_names, position_mat, signal_mat, mep_data_mat, target_position = get_data(
        map_gen, mep_data, participant.return_grid_trials(), pseudo
    )

    grid_name = map_gen.brainsight_data[0]["target_name"][0][0].split(" ")[0]
    idx_axis_1 = (
        np.where(target_names == f"{grid_name} (6, 0)")[1][0],
        np.where(target_names == f"{grid_name} (0, 0)")[1][0],
    )
    to_plot = (
        np.where(target_names == f"{grid_name} (6, 0)")[1][0],
        np.where(target_names == f"{grid_name} (0, 0)")[1][0],
        np.where(target_names == f"{grid_name} (0, 6)")[1][0],
        np.where(target_names == f"{grid_name} (6, 6)")[1][0],
    )
    colors = ["r", "g", "b", "c"]
    maps_characteristics = []
    fig, ax = plt.subplots(len(signal_mat), 2, num="points projection_" + name, sharey=True, sharex=True)
    for i in range(len(signal_mat)):
        points = position_mat[i][:, 3, :3]
        # remove points where position == (0,0,0)
        idx_zero = np.where(np.all(points == 0, axis=1))[0]
        if len(idx_zero) > 0:
            points = np.delete(points, idx_zero, axis=0)
            signal_data = np.delete(signal_mat[i], idx_zero, axis=-1)
        else:
            signal_data = signal_mat[i]
        # if i == 0:
        (x, y, z), com = get_plane_from_points(points)
        y = -y if z[0] > 0 else y
        z = -z if z[0] > 0 else z
        local = np.array([to_plane_coordinates(p, (0, 0, 0), x, y, z) for p in points - com])
        rotated_points = rotate_points(local[:, :2], idx_axis_1=idx_axis_1, additional_rot=0)
        baseline, mep_data = map_gen._get_baseline_mep(signal_data, stimulation_time=1, windows=([50, 5], [10, 40]))
        # plot_2d_points(local, ax[i, 0], colorized_points=(to_plot, colors))
        # plot_2d_points(rotated_points, ax[i, 1], colorized_points=(to_plot, colors))
        map_characteristics_tmp = map_gen.generate_single_map(
            mep_data,
            baseline,
            rotated_points,
            50,
            p2p=mep_data_mat[i],
            tiled=True,
        )
        maps_characteristics.append(map_characteristics_tmp)
    return maps_characteristics


def plot_maps(characteristics_list, name=""):
    fig, ax = plt.subplots(
        len(characteristics_list), len(characteristics_list[0]["x_list"]), num=name, sharey=True, sharex=True
    )
    muscle_names = ['fdi', 'ext_comm', 'supinator', 'tri', 'delt_post']
    for i, characteristics in enumerate(characteristics_list):
        x_list, y_list, z_list = (
            characteristics["x_list"],
            characteristics["y_list"],
            characteristics["z_list"],
        )
        x_cog, y_cog = characteristics["x_cog_list"], characteristics["y_cog_list"]
        area, volume = characteristics["area_list"], characteristics["volume_list"]
        for j in range(len(x_list)):
            # plot_2d_points(local, ax[0, j], colorized_points=(to_plot, colors))
            # plot_2d_points(rotated_points, ax[1, j], colorized_points=(to_plot, colors))
            # plot_heatmap(rotated_points, mep_data[:, j, :], ax[2, j])
            # plot_single_map(x_list[j], y_list[j], z_list[j], ax[i, j], 50, 0,0, area[j], volume[j])
            plot_single_map(x_list[j], y_list[j], z_list[j], ax[i, j], 50, x_cog[j], y_cog[j], area[j], volume[j])
            if j == 0 and i == len(x_list) // 2:
                ax[i, j].set_ylabel(f"latero-medial (mm)\n Map {i}")
            elif j == 0:
                ax[i, j].set_ylabel(f"\nMap {i}")
            if i == 0:
                ax[i, j].set_title(f"{muscle_names[j]}")
        _ = [a.set_aspect("equal") for a in ax.flatten()]
        ax[-1, len(x_list) // 2].set_xlabel("antero-posterior (mm)")
        # ax[len(characteristics_list) // 2, 0].set_ylabel("latero-medial (mm)")



    fig.suptitle(f"Map {name}")
    # save figure
    plt.savefig(f"maps_characteristics_{name}.png", dpi=300)


def plot_characteristics(characteristics_list, name="", absolutes=False):
    x_cog = [char["x_cog_list"] for char in characteristics_list]
    y_cog = [char["y_cog_list"] for char in characteristics_list]
    area = [char["area_list"] for char in characteristics_list]
    volume = [char["volume_list"] for char in characteristics_list]
    z_list = [char["z_list"] for char in characteristics_list]
    fig, ax = plt.subplots(5, len(z_list[0]), num=name)
    for i in range(len(z_list[0])):
        n_map = len(x_cog)
        x_plot = list(range(n_map))
        cor = [pearsonr(z_list[p][i].flatten(), z_list[p + 1][i].flatten())[0] for p in range(0, n_map - 1)]
        if absolutes:
            cor = [0] + cor
            _ = [ax[0, i].scatter(x_plot[p], x_cog[p][i]) for p in range(n_map)]
            _ = [ax[1, i].scatter(x_plot[p], y_cog[p][i]) for p in range(n_map)]
            _ = [ax[2, i].scatter(x_plot[p], area[p][i]) for p in range(n_map)]
            _ = [ax[3, i].scatter(x_plot[p], volume[p][i]) for p in range(n_map)]
            _ = [ax[4, i].scatter(x_plot[p], cor[p]) for p in range(n_map)]
        else:
            _ = [ax[0, i].scatter(x_plot[p], x_cog[p][i] - x_cog[p + 1][i]) for p in range(n_map - 1)]
            _ = [ax[1, i].scatter(x_plot[p], y_cog[p][i] - y_cog[p + 1][i]) for p in range(n_map - 1)]
            _ = [ax[2, i].scatter(x_plot[p], area[p][i] - area[p + 1][i]) for p in range(n_map - 1)]
            _ = [ax[3, i].scatter(x_plot[p], volume[p][i] - volume[p + 1][i]) for p in range(n_map - 1)]
            _ = [ax[4, i].scatter(x_plot[p], cor[p]) for p in range(n_map - 1)]


def add_to_dataframe(maps, data_frame, participant, condition, muscle_list):
    cog_err_x = [[np.nan for _ in range(len(muscle_list))]]
    cog_err_x = cog_err_x + [
        np.abs(np.array(maps[i + 1]["x_cog_list"]) - np.array(maps[i]["x_cog_list"])) for i in range(len(maps) - 1)
    ]
    cog_err_y = [[np.nan for _ in range(len(muscle_list))]]
    cog_err_y = cog_err_y + [
        np.abs(np.array(maps[i + 1]["y_cog_list"]) - np.array(maps[i]["y_cog_list"])) for i in range(len(maps) - 1)
    ]
    cog_err_eucl = [[np.nan for _ in range(len(muscle_list))]]
    cog_err_eucl = cog_err_eucl + [
        np.linalg.norm(np.array([maps[i + 1]["x_cog_list"], maps[i + 1]["y_cog_list"]]) - np.array([maps[i]["x_cog_list"], maps[i]["y_cog_list"]]), axis=0)
        for i in range(len(maps) - 1)
    ]
    cor_coef = [[np.nan for _ in range(len(muscle_list))]]
    cor_coef = cor_coef + [
        [
            pearsonr(maps[i + 1]["z_list"][m].flatten(), maps[i]["z_list"][m].flatten())[0]
            for m in range(len(muscle_list))
        ]
        for i in range(len(maps) - 1)
    ]

    for m, char in enumerate(maps):
        data_frame_tmp = pd.DataFrame(
            {
                "participant": [participant] * len(muscle_list),
                "condition": [condition] * len(muscle_list),
                "map_number": [m] * len(muscle_list),
                "x_cog": char["x_cog_list"],
                "y_cog": char["y_cog_list"],
                "area": char["area_list"],
                "volume": char["volume_list"],
                "muscle": muscle_list,
                "x_cog_error": cog_err_x[m],
                "y_cog_error": cog_err_y[m],
                "euclid_cog_error": cog_err_eucl[m],
                "correlation_coefficient": cor_coef[m],
            }
        )
        if data_frame.empty:
            data_frame = data_frame_tmp
        else:
            data_frame = pd.concat([data_frame, data_frame_tmp])
    return data_frame


if __name__ == "__main__":
    paticipants = list(range(2, 7))
    participants = [f"P{p:03d}_TN" for p in paticipants]
    p2p_from_file = True
    condition = ["pseudo", "grid"]
    muscle_list = ["fdi", "ext_comm", "sup", "tri", "delt_post"]
    data_frame = pd.DataFrame()
    for part_name in participants:
        participant = Participant(part_name)
        for m, name in enumerate(condition):
            maps = compute_maps(participant, pseudo=name == "pseudo", data_rate=2148, p2p_from_file=p2p_from_file)
            data_frame = add_to_dataframe(maps, data_frame, part_name, name, muscle_list)
            # plot_characteristics(maps, name + '_' + part_name, absolutes=True)
            # plot_maps(maps, name=name + f' maps participant {part_name}')
    data_frame.to_csv("maps_characteristics.csv")
    # plt.show()
