from csv import excel
import os
import numpy as np
from scipy.spatial import ConvexHull
import random

import pandas as pd


def project_points_to_plane(points, normal, com):
    projected_points = np.zeros_like(points)
    for p in range(points.shape[0]):
        projected_points[p, :] = points[p, :] - np.dot(points[p, :] - com, normal) * normal
    return projected_points


def angle_between_vectors_cross(u, v):
    """
    Calculate the angle between two vectors using the cross product.

    Parameters:
    u, v : array-like
        Input vectors.

    Returns:
    tuple
        The angle between the vectors in radians and degrees.
    """
    u = (*u, 0)
    v = (*v, 0)
    u = np.array(u)
    v = np.array(v)

    cross_product = np.cross(u, v)
    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)
    magnitude_cross = np.linalg.norm(cross_product)

    sin_theta = magnitude_cross / (magnitude_u * magnitude_v)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)  # Ensure sin_theta is within the valid range

    angle_radians = np.arcsin(sin_theta)
    angle_degrees = np.degrees(angle_radians)

    return angle_radians, angle_degrees


def apply_rotation(angle, points):
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    rotated_local_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    local = rotated_local_hom @ R.T
    return local[:, :2]


def rotate_points(points, idx_axis_1=(0, -6), additional_rot=0, ref_axis=[1, 0]):
    idx_axis_1 = [-1, -6] if idx_axis_1 is None else idx_axis_1
    axis_1 = points[idx_axis_1[0], :] - points[idx_axis_1[1], :]
    ref_axis = np.array(ref_axis)
    # if self.angle is None:
    angle, _ = angle_between_vectors_cross(axis_1, ref_axis)
    rotated_local = apply_rotation(angle, points.copy())
    axis_1 = rotated_local[idx_axis_1[0], :] - rotated_local[idx_axis_1[1], :]
    angle_bis, _ = angle_between_vectors_cross(axis_1, ref_axis)
    if abs(angle_bis) > 0.001:
        rotated_local = apply_rotation(-angle, points.copy())

    # check if same direction
    rotated_local_x = rotated_local[idx_axis_1[0], :] - rotated_local[idx_axis_1[1], :]
    norm_rotated = np.linalg.norm(rotated_local_x)
    rot_normilized = rotated_local_x / norm_rotated
    if np.dot(rot_normilized, ref_axis) < 0:
        return apply_rotation(np.pi, rotated_local)
    return rotated_local


def to_plane_coordinates(p_proj, origin, x_axis, y_axis, z_axis=None):
    if z_axis is not None:
        return to_plane_coordinates_3d(p_proj, origin, x_axis, y_axis, z_axis)
    return to_plane_coordinates_2d(p_proj, origin, x_axis, y_axis)


def to_plane_coordinates_2d(p_proj, origin, x_axis, y_axis):
    vec = p_proj - origin
    return np.array([np.dot(vec, x_axis), np.dot(vec, y_axis)])


def to_plane_coordinates_3d(p_proj, origin, x_axis, y_axis, z_axis):
    vec = p_proj - origin
    return np.array([np.dot(vec, x_axis), np.dot(vec, y_axis), np.dot(vec, z_axis)])


def project_point_onto_plane(point, plane_point, plane_normal):
    vec = point - plane_point
    distance = np.dot(vec, plane_normal)
    return point - distance * plane_normal


def get_plane_from_points(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered)

    normal = Vt[2]
    x_axis = Vt[0]  # first principal direction
    y_axis = Vt[1]  # second principal direction
    z_axis = normal  # third is the normal

    return (x_axis, y_axis, z_axis), centroid


def exclude_signal_data(p2p, baseline, n_map):
    p2p_threshold = 3.5
    p2p_mean = np.mean(p2p, axis=-1)
    p2p_std = np.std(p2p, axis=-1)
    idx_excluded_p2p = [np.where(p2p[i, :] > p2p_mean[i] + p2p_threshold * p2p_std[i])[0] for i in range(p2p.shape[0])]

    baseline_threshold = 2
    baseline = baseline.swapaxes(0, 1)
    baseline_mean_window = np.mean(baseline, axis=1)
    baseline_conc = baseline.reshape(n_map + 1, -1)
    baseline_mean = np.mean(baseline_conc, axis=-1)
    baseline_std = np.std(baseline_conc, axis=-1)
    idx_excluded_baseline = [
        np.where(baseline_mean_window[i, :] > baseline_mean[i] + baseline_threshold * baseline_std[i])[0]
        for i in range(baseline_mean_window.shape[0])
    ]

    idx_general = [
        np.array(idx_excluded_baseline[i].tolist() + idx_excluded_p2p[i].tolist())
        for i in range(len(idx_excluded_baseline))
    ]
    idx_general = [np.unique(idx_general[i]) for i in range(len(idx_general))]

    return idx_general


def get_random_points(list_nb_points, nb_total_points):
    random.seed(42)
    initial_list = np.arange(0, nb_total_points)
    random.shuffle(initial_list[4:])
    return [initial_list[0 : list_nb_points[i]] for i in range(len(list_nb_points))]


def get_mep_from_excel(dir_path, trials, base_name, channel_names=None, exclude_mep=False, reverse=True):
    all_mep_data = []
    all_frames = []
    nb_files = 0
    for trial in trials:
        data_mat = []
        frames = []
        for channel in channel_names:
            excel_names = f"{base_name}{trial}_{channel.lower()}_MatLabResults.xlsx"
            if not os.path.exists(os.path.join(dir_path, excel_names)):
                excel_names = f"{base_name}{trial}_{channel.upper()}_MatLabResults.xls"

            if os.path.exists(os.path.join(dir_path, excel_names)):
                data_tmp = pd.read_excel(os.path.join(dir_path, excel_names), sheet_name=1)
                headers = data_tmp.values[18]
                mep_found_idx = headers.tolist().index("Found")
                data_glob = data_tmp.values[21:-1]
                frame_idx = 0
                mep_idx = 1
                frames_tmp = data_glob[:, frame_idx]
                mep_tmp = data_glob[:, mep_idx].astype(float)
                is_mep = data_glob[:, mep_found_idx] > 0
                mep_tmp[is_mep == False] = 0
                mep_tmp = mep_tmp if not reverse else mep_tmp[::-1]
                data_mat.append(mep_tmp)
                frames.append(frames_tmp)
                nb_files += 1
            else:
                data_mat.append(None)
                frames.append(None)
        if nb_files == 0:
            return None, None

        sizes = [mat.shape[0] for mat in data_mat if mat is not None]
        frames_tmp = [frame for frame in frames if frame is not None][0]
        for i in range(len(data_mat)):
            if data_mat[i] is None:
                data_mat[i] = np.zeros((sizes[0]))
                frames[i] = frames_tmp
        all_mep_data.append(np.array(data_mat))
        all_frames.append(np.array(frames))
        if exclude_mep:
            all_mep_data = [exclude_outliers(mep_data) for mep_data in all_mep_data]
    return all_mep_data, all_frames


def exclude_outliers(data, threshold=3.5):
    for i in range(data.shape[0]):
        if data[i].sum() == 0:
            continue
        mean = data[i][data[i] != 0].mean()
        std = data[i][data[i] != 0].std()
        idx_excluded = np.where(data[i] > mean + threshold * std)[0]
        data[i][idx_excluded] = np.nan
    return data


def get_cog(x, y, p2p):
    if p2p.sum() == 0:
        return 0, 0
    x_cog = np.sum(x * p2p) / np.sum(p2p)
    y_cog = np.sum(y * p2p) / np.sum(p2p)
    return x_cog, y_cog


def get_area_and_volume(x, y, z):
    x = x[z > np.max(z) * 0.1]
    y = y[z > np.max(z) * 0.1]
    z = z[z > np.max(z) * 0.1]
    if x.size < 4:
        return 0, 0
    hull_area = ConvexHull(np.array([x, y]).T)
    area = hull_area.volume
    hull_volume = ConvexHull(np.array([x, y, z]).T)
    volume = hull_volume.volume
    return area, volume


def check_order(name):
    import csv

    number = int(name.split("_")[0][-3:])
    with open("participant_numbers.txt", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if int(row[0].split("(")[-1]) == number:
                return str(row[1].split(")")[0])[2:-1] == "pseudo first"


class ParticipantTest:
    def __init__(self, name):
        self.name = name

    def part_dict(self):
        return {
            "base_name": "UA",
            "dir_path": self._return_dir_path(self.name),
            "pkl_file_name": self._return_pkl_file_name(self.name),
            "excel_file_name": self._return_excel_file_name(self.name),
        }

    def return_pkl_file_name(self):

        return rf"data_trial_test_mapping_{self.name}007.pkl"

    def return_pkl_file_base(self):
        return rf"data_trial_test_mapping_{self.name}00"

    def return_dir_path(self):
        return rf"D:\Documents\Udem\Postdoctorat\Projet transfert nerveux\data\test_{self.name}_001"

    def return_excel_file_name(self):
        return rf"test_mapping_{self.name}_00"

    def return_pseudo_trial(self):
        return ["7"]

    def return_grid_trials(self):
        return ["3", "4", "5", "6"]

    def return_trials(self, pseudo=False):
        if pseudo:
            return self.return_pseudo_trial()
        return self.return_grid_trials()

    def excel_mep(self, pseudo_trial=False):
        trials = self.return_trials(pseudo_trial)

        mep_data_file, frame_file = get_mep_from_excel(
            self.return_dir_path(),
            trials,
            channel_names=["FDI", "ext_comm", "sup", "tri", "delt_post"],
            exclude_mep=True,
            base_name=self.return_excel_file_name(),
        )
        return mep_data_file, frame_file


class Participant:
    def __init__(self, name):
        self.name = name
        self.pseudo_first = False
        if not "SCI" in name:
            self.pseudo_first = check_order(name)
        self.trials = ["2", "3", "4", "5", "6", "7"]

    def part_dict(self):
        return {
            "base_name": "UA",
            "dir_path": self._return_dir_path(self.name),
            "pkl_file_name": self._return_pkl_file_name(self.name),
            "excel_file_name": self._return_excel_file_name(self.name),
        }

    def return_pkl_file_name(self):
        return rf"data_trial_{self.name}00{self.return_pseudo_trial()[0]}.pkl"

    def return_pkl_file_base(self):
        return rf"data_trial_{self.name}00"

    def return_dir_path(self):
        return rf"D:\Documents\Udem\Postdoctorat\Projet transfert nerveux\data\{self.name}"

    def return_excel_file_name(self):
        return rf"{self.name}_00"

    def return_pseudo_trial(self):
        trial = self.trials[-1] if not self.pseudo_first else self.trials[0]
        return [trial]

    def return_grid_trials(self):
        trial = self.trials[:-1] if not self.pseudo_first else self.trials[1:]
        return trial

    def return_trials(self, pseudo=False):
        if pseudo:
            return self.return_pseudo_trial()
        return self.return_grid_trials()

    def excel_mep(self, pseudo_trial=False):
        trials = self.return_trials(pseudo_trial)

        mep_data_file, frame_file = get_mep_from_excel(
            self.return_dir_path(),
            trials,
            channel_names=["FDI", "ext_comm", "sup", "tri", "delt_post"],
            exclude_mep=True,
            base_name=self.return_excel_file_name(),
        )
        return mep_data_file, frame_file
