import os
from matplotlib import axis
import matplotlib.pyplot as plt
import pandas as pd
from biosiglive import load
from map_generator.plot_utils import plot_2d_points, plot_3D_points, plot_single_map
from map_generator.utils import (
    Participant,
    get_random_points,
    get_plane_from_points,
    rotate_points,
    to_plane_coordinates,
)
from biosiglive import save
from map_generator.grid_based_generator import GridBasedGenerator
from map_generator.pseudo_random_generator import PseudoRandomGenerator
from scipy.stats import pearsonr
import numpy as np


def get_data_from_pseudo(map_gen, mep_data=None, frame_for_file=None, seed=None):
    nb_stim_list = [24, 44, 64, 94, 124, 154, map_gen.signal_data[0].shape[-1]]
    rdm_points = get_random_points(nb_stim_list, nb_total_points=map_gen.signal_data[0].shape[-1], seed=seed)
    target_names = map_gen.brainsight_data[0]["target_name"]
    target_names_roll = np.roll(target_names, 4)
    signal_data_roll = np.roll(map_gen.signal_data[0], 4, axis=-1)
    position_roll = np.roll(map_gen.position[0], 4, axis=0)
    target_position = np.roll(map_gen.target_position[0], 4, axis=0)
    signal_mat = [signal_data_roll[..., rdm_points[i]] for i in range(len(rdm_points))]
    position_mat = [position_roll[rdm_points[i]] for i in range(len(rdm_points))]
    target_position_mat = [target_position[rdm_points[i]] for i in range(len(rdm_points))]
    target_names_mat = [target_names_roll[:, rdm_points[i]] for i in range(len(rdm_points))]
    if p2p_from_file and mep_data is not None:
        mep_data = check_frame_numbers(map_gen, mep_data, frame_for_file)
        mep_data_file_roll = np.roll(mep_data[0], 4, axis=-1)
        mep_data_mat = [mep_data_file_roll[..., rdm_points[i]] for i in range(len(rdm_points))]
    else:
        mep_data_mat = [None for _ in range(len(signal_mat))]

    return target_names_mat, position_mat, signal_mat, mep_data_mat, target_position_mat


def get_data_from_grid(map_gen, trials, mep_data=None, frame_for_file=None):
    to_concat = list(range(1, len(trials) + 1))
    target_names = [brain["target_name"] for brain in map_gen.brainsight_data]
    target_names = [np.hstack(target_names[:c]) for c in to_concat]
    idx = min([si.shape[0] for si in map_gen.signal_data])
    signal_mat = [np.concatenate(([si[:idx, ...] for si in map_gen.signal_data[:c]]), axis=-1) for c in to_concat]
    position_mat = [np.vstack(map_gen.position[:c]) for c in to_concat]
    target_position_mat = [np.vstack(map_gen.target_position[:c]) for c in to_concat]
    if p2p_from_file and mep_data is not None:
        mep_data = check_frame_numbers(map_gen, mep_data, frame_for_file)
        mep_data_mat = [np.hstack([si for si in mep_data[:c]]) for c in to_concat]
    else:
        mep_data_mat = [None for _ in range(len(signal_mat))]
    return target_names, position_mat, signal_mat, mep_data_mat, target_position_mat

def check_frame_numbers(map_gen, mep_data, frame_from_file):
    ordered_mep_data = []
    for i in range(len(frame_from_file)):
        frame_from_pkl = [int(frame.split(' ')[1]) for frame in map_gen.all_data[i]['signal_data']["frame_number"]]
        idx_list = [np.where(frame_from_pkl[j] == frame_from_file[i][0])[0][0] for j in range(len(frame_from_pkl))]
        frame_from_file[i][0][idx_list]
        ordered_mep_data.append(mep_data[i][:, idx_list])
    return ordered_mep_data


def get_data(map_gen, mep_data, trial_list=None, pseudo=False, frame_from_file=None, seed=None):
    if pseudo:
        return get_data_from_pseudo(map_gen, mep_data, frame_from_file, seed)
    return get_data_from_grid(map_gen, trial_list, mep_data, frame_from_file)

def get_idx_to_rotate(target_names):
    grid_name = target_names[0][0].split(" ")[0]
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
    return idx_axis_1, to_plot, colors

def compute_maps(participant, pseudo=False, data_rate=2148, p2p_from_file=False, grid_target_pos=None, seed=None):
    map_instance = PseudoRandomGenerator if pseudo else GridBasedGenerator
    file_name = participant.return_pkl_file_name() if pseudo else participant.return_pkl_file_base()
    map_gen = map_instance(
        participant.return_dir_path(), file_name, data_rate=data_rate, trial_list=participant.return_grid_trials()
    )
    mep_data, frame_file = participant.excel_mep(pseudo_trial=pseudo)

    target_names, position_mat, signal_mat, mep_data_mat, target_position = get_data(
        map_gen, mep_data, participant.return_grid_trials(), pseudo, frame_file, seed
    )
    maps_characteristics = []
    # plt.plot(signal_mat[-1][:, 0, :])

    # fig, ax = plt.subplots(2, len(signal_mat),num="points projection_" + name, sharey=True, sharex=True)
    for i in range(len(signal_mat)):
        tic = time.time()
        points = position_mat[i][:, 3, :3].copy()
        idx_zero = np.where(np.all(points == 0, axis=1))[0]
        if len(idx_zero) > 0:
            points = np.delete(points, idx_zero, axis=0)
            signal_data = np.delete(signal_mat[i], idx_zero, axis=-1)
            target_names_tmp = np.delete(target_names[i], idx_zero, axis=-1)
            if mep_data_mat[i] is not None:
                mep_data_tmp = np.delete(mep_data_mat[i], idx_zero, axis=-1)
            else:
                mep_data_tmp = None
        else:
            signal_data = signal_mat[i]
            target_names_tmp = target_names[i]
            mep_data_tmp = mep_data_mat[i] if mep_data_mat[i] is not None else None
        
        idx_axis_1, to_plot, colors = get_idx_to_rotate(target_names_tmp)
        # if i == 0:
            # plot 3d
        # position = grid_target_pos[0][:, 3, :3] if pseudo else target_position[0][:, 3, :3]
        # (x, y, z), com = get_plane_from_points(position) #, position_mat[-1][:, 3, :3])
        # increase weigthing at the corner at index "to_plot"
        if pseudo:
            points_weighted = points
            # points_to_add = points[to_plot, :]
            # repeat point to add
            # points_weighted = np.vstack((points_weighted, np.repeat(points_to_add, 10, axis=0)))
            (x, y, z), com = get_plane_from_points(points_weighted) #, position_mat[-1][:, 3, :3])
        else:
            (x, y, z), com = get_plane_from_points(points) #, position_mat[-1][:, 3, :3])
        # from map_generator.plot_utils import plot_plane
        # plot_plane(points, com, z, fig_name='global')
        # (x, y, z), com = get_plane_from_points(points_weighted) #, position_mat[-1][:, 3, :3])
        # # local = np.array([to_plane_coordinates(p, (0, 0, 0), x, y, z) for p in points - com])
        # plot_plane(points_weighted, com, z, fig_name='local')
        # # project back to the global coordinate system
        # plt.show()
        if np.dot(z, [0, 0, 1]) < 0:
            y = -y
            z = -z

        local = np.array([to_plane_coordinates(p, (0, 0, 0), x, y, z) for p in points - com])
        rotated_points = rotate_points(local[:, :2], idx_axis_1=idx_axis_1, additional_rot=0)
        baseline, mep_data = map_gen._get_baseline_mep(signal_data, stimulation_time=1, windows=([50, 5], [15, 35]))
        # plot_2d_points(local, ax[0, i], colorized_points=(to_plot, colors))
        # plot_2d_points(rotated_points, ax[1, i], colorized_points=(to_plot, colors))
        # check if the point are evenly spread on the area

        mep_data_tmp = None if not p2p_from_file else mep_data_tmp
        map_characteristics_tmp = map_gen.generate_single_map(
            mep_data,
            baseline,
            rotated_points,
            50,
            p2p=mep_data_tmp,
            tiled=False,
            pseudo=pseudo
        )

        map_characteristics_tmp.update({'time_to_compute': time.time() - tic})
        maps_characteristics.append(map_characteristics_tmp)
    target_pos_to_return = target_position if not pseudo else None
    return maps_characteristics, target_pos_to_return


def plot_maps(characteristics_list, name="", fold='', muscle_names=None):
    init_muscle_names = ['fdi', 'ext_comm', 'sup', 'tri', 'delt_post']
    muscle_names = init_muscle_names if not muscle_names else muscle_names
    idx_muscles = [init_muscle_names.index(name) for name in muscle_names]
    fig, ax = plt.subplots(
        len(characteristics_list), len(muscle_names), num=name, sharey=True, sharex=True
    )

    for i, characteristics in enumerate(characteristics_list):
        
        x_list, y_list, z_list = (
            characteristics["xgf_list"],
            characteristics["ygf_list"],
            characteristics["zgf_list"],
        )
        x_real, y_real = (
            characteristics["x_list"],
            characteristics["y_list"],
        )
        x_cog, y_cog = characteristics["x_cog_list"], characteristics["y_cog_list"]
        area, volume = characteristics["area_list"], characteristics["volume_list"]
        for j in idx_muscles:
            # plot_2d_points(local, ax[0, j], colorized_points=(to_plot, colors))
            # plot_2d_points(rotated_points, ax[1, j], colorized_points=(to_plot, colors))
            # plot_heatmap(rotated_points, mep_data[:, j, :], ax[2, j])
            # plot_single_map(x_list[j], y_list[j], z_list[j], ax[i, j], 50, 0,0, area[j], volume[j])
            plot_single_map(x_list[j], y_list[j], z_list[j], ax[i, j], 50, x_cog[j], y_cog[j], area[j], volume[j], x_real[j], y_real[j])
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
    plt.savefig(fr"D:\Documents\Programmation\tms_motor_map\results\{fold}\maps_characteristics_{name}.png")


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


def add_to_dataframe(maps, data_frame, participant, condition, muscle_list, fold='', seed=''):
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
            pearsonr(maps[i + 1]["zgf_list"][m].flatten(),
                      maps[i]["zgf_list"][m].flatten())[0]
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
                "time_to_compute": [char['time_to_compute']] * len(muscle_list),
            }
        )
        name = f"results\{fold}\maps_values.bio" 
        save({"participant": [participant], 
              "condition": [condition], 
              "map_number": [m],
              "muscle": muscle_list,
                "xgf_list": char["xgf_list"],
                "ygf_list": char["ygf_list"],
                "zgf_list": char["zgf_list"],
                "x_list": char["x_list"],
                "y_list": char["y_list"],
                "z_list": char["z_list"]},  name, add_data=True)
        
        if data_frame.empty:
            data_frame = data_frame_tmp
        else:
            data_frame = pd.concat([data_frame, data_frame_tmp])
    return data_frame


if __name__ == "__main__":
    paticipants = list(range(2, 14))
    participants = [f"P{p:03d}_TN" for p in paticipants]
    # participants = ['P004_TN_SCI']
    p2p_from_file = False
    condition = ["grid", "pseudo"]
    muscle_list = ["fdi", "ext_comm", "sup", "tri", "delt_post"]

    seed = np.random.randint(0, 100000)
    seed = 10
    # seed = 'test'
    fold = f'smooth_5_5_{seed}'
    os.makedirs(f'results\{fold}', exist_ok=True)
    # for i in range(batch_number):
    import time
    target = None
    data_frame = pd.DataFrame()
    for part_name in participants:
        participant = Participant(part_name)
        for m, name in enumerate(condition):
            maps, target = compute_maps(participant, pseudo=name == "pseudo", data_rate=2148, p2p_from_file=p2p_from_file, grid_target_pos=target, seed=seed)
            plot_maps(maps, name=name + f' maps participant {part_name}', fold=fold, muscle_names=muscle_list[:3])
            data_frame = add_to_dataframe(maps, data_frame, part_name, name, muscle_list, fold, seed)
            # plt.show()
    dir_to_save = f"maps_characteristics.csv"
    data_frame.to_csv(f"results\{fold}\{dir_to_save}")
    # plt.show()
