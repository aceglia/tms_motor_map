import os
import numpy as np
import random
import pandas as pd

def get_idx_to_rotate(target_names):
    grid_name = target_names[0].split(" ")[0]
    idx_axis_1 = (
        np.where(target_names == f"{grid_name} (6, 0)")[0][-1],
        np.where(target_names == f"{grid_name} (0, 0)")[0][-1],
    )
    to_plot = (
        np.where(target_names == f"{grid_name} (6, 0)")[0][-1],
        np.where(target_names == f"{grid_name} (0, 0)")[0][-1],
        np.where(target_names == f"{grid_name} (0, 6)")[0][-1],
        np.where(target_names == f"{grid_name} (6, 6)")[0][-1],
    )
    colors = ["r", "g", "b", "c"]
    return idx_axis_1, to_plot, colors


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


def rotate_points(points, idx_axis_1, ref_axis=[1, 0]):
    axis_1 = points[idx_axis_1[0], :] - points[idx_axis_1[1], :]
    ref_axis = np.array(ref_axis)
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


def ransac_plane(points, threshold=0.01, max_iterations=1000):
    best_inliers = []
    best_plane = None

    for _ in range(max_iterations):
        # Randomly sample 3 points to define a plane
        sample_indices = np.random.choice(points.shape[0], 3, replace=False)
        p1, p2, p3 = points[sample_indices]

        # Compute the plane normal
        normal = np.cross(p2 - p1, p3 - p1)
        normal /= np.linalg.norm(normal)

        # Compute the plane equation: ax + by + cz + d = 0
        d = -np.dot(normal, p1)

        # Compute distances of all points to the plane
        distances = np.abs(np.dot(points, normal) + d)

        # Identify inliers
        inliers = np.where(distances < threshold)[0]

        # Update best plane if this one has more inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)

    return best_plane, best_inliers


# import open3d as o3d

# def o3d_ransac_plane(points, threshold=0.01, max_iterations=1000):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     plane_model, inliers = pcd.segment_plane(distance_threshold=threshold, ransac_n=3, num_iterations=max_iterations)
#     return plane_model, inliers


def get_plane_from_points(points, to_center=None, **kwargs):

    centroid_all = np.mean(points, axis=0)

    _, inliers = ransac_plane(points, **kwargs)

    points_plane = points[inliers]

    centroid = np.mean(points_plane, axis=0)
    centered = points_plane - centroid

    _, _, Vt = np.linalg.svd(centered)

    x_axis = Vt[0]
    y_axis = Vt[1]
    z_axis = Vt[2]

    # consistent orientation
    if np.dot(z_axis, [0, 0, 1]) < 0:
        z_axis *= -1
        y_axis *= -1

    # enforce right-handed frame
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # project global centroid to plane
    dist = np.dot(z_axis, centroid_all - centroid)
    to_center = centroid_all if to_center is None else to_center
    com_new = to_center - dist * z_axis

    return (x_axis, y_axis, z_axis), com_new, inliers


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


def get_random_points(list_nb_points, nb_total_points, seed=None):
    # test avec 50, 100, 2504, 26(pasmal), 58, 925
    if seed is not None:
        random.seed(seed)
    initial_list = np.arange(0, nb_total_points)
    random.shuffle(initial_list[4:])
    return [initial_list[0 : list_nb_points[i]] for i in range(len(list_nb_points))]


def get_mep_from_excel(dir_path, trials, base_name, channel_names=None, exclude_mep=True, reverse=True):
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
                frame_idx = headers.tolist().index("Frame")
                mep_idx = headers.tolist().index("Pk to Pk")
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


def exclude_outliers(data, threshold=3.5, mep_limit=50):
    for i in range(data.shape[0]):
        if data[i].sum() == 0:
            continue
        data[i][data[i] <= mep_limit] = 0  # np.nan
        mean = np.nanmean(data[i])
        std = np.nanstd(data[i])
        data[i][data[i] > mean + threshold * std] = 0  # np.nan
    return data


def get_cog(x, y, p2p):
    if p2p.sum() == 0:
        return 0, 0
    x_cog = np.sum(x * p2p) / np.sum(p2p)
    y_cog = np.sum(y * p2p) / np.sum(p2p)
    return x_cog, y_cog


def get_area_and_volume(x, y, z, n_tot=2500):
    area_tot = (abs(x.min()) + abs(x.max())) * (abs(y.min()) + abs(y.max()))
    area = (len(np.where(z > z.max() * 0.1)[0]) / n_tot) * area_tot
    volume = np.sum(z[z > z.max() * 0.1]) - 0.1 * len(np.where(z > z.max() * 0.1)[0]) * z.max()
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
        if "SCI" in name:
            self.trials = self.trials[:]

    def return_pkl_file_name(self):
        return rf"data_trial_P{self.name}00{self.return_pseudo_trial()[0]}.pkl"

    def return_pkl_file_base(self):
        return rf"data_trial_P{self.name}00"

    def return_dir_path(self):
        return rf"D:\Documents\Udem\Postdoctorat\Projet transfert nerveux\data_serveur\{self.name}\TMS_mapping"
        # return rf"D:\Documents\Udem\Postdoctorat\Projet transfert nerveux\data\{self.name}"

        # return rf"\\IURDPM\Synapse\1_Recherche\Equipe_Barthelemy\1-Projets\DB_TN24H_TransfertNerveux\3- Raw Data\Healthy\{self.name}\TMS_mapping"

    def return_excel_file_name(self):
        return rf"P{self.name}_00"

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
            exclude_mep=False,
            base_name=self.return_excel_file_name(),
            reverse=pseudo_trial,
        )
        return mep_data_file, frame_file
