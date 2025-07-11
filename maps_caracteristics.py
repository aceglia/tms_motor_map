import matplotlib.pyplot as plt
from map_generator.plot_utils import plot_single_map
from map_generator.utils import Participant, get_random_points, get_plane_from_points, rotate_points, to_plane_coordinates
from map_generator.grid_based_generator import GridBasedGenerator
from map_generator.pseudo_random_generator import PseudoRandomGenerator
import numpy as np

def get_data_from_pseudo(map_gen, mep_data=None):
    nb_stim_list = [24, 64, 94, 124, 154, map_gen.signal_data[0].shape[-1]]
    rdm_points = get_random_points(nb_stim_list, nb_total_points=map_gen.signal_data[0].shape[-1] - 4)
    target_names = map_gen.brainsight_data[0]["target_name"]

    target_names_roll = np.roll(target_names, 4)
    mep_data_file_roll = np.roll(mep_data[0], 4, axis=-1)
    signal_data_roll = np.roll(map_gen.signal_data[0], 4, axis=-1)
    position_roll = np.roll(map_gen.position[0], 4, axis=0)
    signal_mat = [signal_data_roll[..., rdm_points[i]] for i in range(len(rdm_points))]
    position_mat = [position_roll[rdm_points[i]] for i in range(len(rdm_points))]
    if p2p_from_file and mep_data is not None:
        mep_data_mat = [mep_data_file_roll[..., rdm_points[i]] for i in range(len(rdm_points))]
    else:
        mep_data_mat = [None for _ in range(len(signal_mat))]

    return target_names_roll, position_mat, signal_mat, mep_data_mat

def get_data_from_grid(map_gen, trials, mep_data=None):
    to_concat = list(range(1, len(trials) + 1))
    target_names = map_gen.brainsight_data[0]["target_name"]
    idx = min([si.shape[0] for si in map_gen.signal_data])
    signal_mat = [np.concatenate(([si[:idx, ...] for si in map_gen.signal_data[:c]]), axis=-1) for c in to_concat]
    position_mat = [np.vstack(map_gen.position[:c]) for c in to_concat]

    if p2p_from_file and mep_data is not None:
        mep_data_mat = [np.hstack([si for si in mep_data[:c]]) for c in to_concat]
    else:
        mep_data_mat = [None for _ in range(len(signal_mat))]
    return target_names, position_mat, signal_mat, mep_data_mat

def get_data(map_gen, mep_data, trial_list=None, pseudo=False):
    if pseudo:
        return get_data_from_pseudo(map_gen, mep_data)
    return get_data_from_grid(map_gen, trial_list, mep_data)


def compute_maps(participant, pseudo=False, data_rate=2148, p2p_from_file=False):
    map_instance = PseudoRandomGenerator if pseudo else GridBasedGenerator
    file_name = participant.return_pkl_file_name() if pseudo else participant.return_pkl_file_base()
    map_gen = map_instance(participant.return_dir_path(), file_name, data_rate=data_rate, trial_list=participant.return_grid_trials())
    
    mep_data, frame_file = participant.excel_mep(pseudo_trial=pseudo)

    if mep_data is not None:
        mep_data = [mep_data[i][:, -map_gen.signal_data[0].shape[-1]:] for i in range(len(map_gen.signal_data))]
    else:
        if p2p_from_file:
            raise RuntimeError('Peak to peak ask from file but none was found.')
    
    target_names, position_mat, signal_mat, mep_data_mat = get_data(map_gen, mep_data, participant.return_grid_trials(), pseudo)
    grid_name = map_gen.brainsight_data[0]["target_name"][0][0].split(" ")[0]
    idx_axis_1 = (
        np.where(target_names == f"{grid_name} (6, 0)")[1][0],
        np.where(target_names == f"{grid_name} (0, 0)")[1][0],
    )
    maps_characteristics = []
    for i in range(len(signal_mat)):
        points = position_mat[i][:, 3, :3]
        signal_data = signal_mat[i]
        if i == 0:
            (x, y, z), com = get_plane_from_points(points)
        y = -y if z[0] > 0 else y
        z = -z if z[0] > 0 else z
        local = np.array([to_plane_coordinates(p, (0, 0, 0), x, y, z) for p in points - com])
        rotated_points = rotate_points(local[:, :2], idx_axis_1=idx_axis_1, additional_rot=0)
        baseline, mep_data = map_gen._get_baseline_mep(signal_data, stimulation_time=1, windows=([50, 5], [10, 80]))
        map_characteristics_tmp = map_gen.generate_single_map(
            mep_data,
            baseline,
            rotated_points,
            50,
            p2p=mep_data_mat[i],
            idx_limit_map=None,
            exclude_outliers_mep=False,
        )
        maps_characteristics.append(map_characteristics_tmp)
    return maps_characteristics


def plot_maps(characteristics_list, name=""):
    fig, ax = plt.subplots(len(characteristics_list), 
                           len(characteristics_list[0]["x_list"]),
                           num=name, sharey=True, sharex=True)

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
            plot_single_map(x_list[j], y_list[j], z_list[j], ax[i, j], 50, x_cog[j], y_cog[j], area[j], volume[j])
        _ = [a.set_aspect("equal", adjustable="box") for a in ax.flatten()]
        ax[-1, len(x_list)//2].set_xlabel("antero-posterior (mm)")
        ax[len(characteristics_list) // 2, 0].set_ylabel("latero-medial (mm)")

def plot_characteristics(characteristics_list, name=''):
    cog = [(char['x_cog_list'], char['y_cog_list']) for char in characteristics_list]
    area = [char['area_list'] for char in characteristics_list]
    volume = [char['volume_list'] for char in characteristics_list]
    fig, ax = plt.subplots(3, len(characteristics_list[0]['x_list']), num=name)
    for i in range(len(characteristics_list)):
        pass


        


if __name__ == '__main__':
    participants = ['HB', 'NH']
    p2p_from_file = True
    all_maps = {'pseudo':[], 'grid':[]}
    for part_name in participants:
        participant = Participant(part_name)
        pseudo_maps = compute_maps(participant, pseudo=True, data_rate=2148, p2p_from_file=p2p_from_file)
        # plot_maps(pseudo_maps, name=f'Pseudo maps participant {part_name}')
        grid_maps = compute_maps(participant, pseudo=False, data_rate=2148, p2p_from_file=p2p_from_file)
        # plot_maps(grid_maps, name=f'Grid maps participant {part_name}')
        for maps in [pseudo_maps, grid_maps]:
            plot_characteristics(maps, '')


        all_maps['pseudo'].append(pseudo_maps)
        all_maps['grid'].append(grid_maps)
    
    # plot scatter plot for x and y cog for all participants and all maps

    

    
    plt.show()






