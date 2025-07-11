from hmac import new
import math
import pickle
import signal
import matplotlib.pyplot as plt
import numpy as np
import os
from biosiglive.file_io.save_and_load import dic_merger, _read_all_lines

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def exclude_signal_data(data):
    pass

def exclude_brainsight_data(data):
    pass


def project_points_to_plane(points):
    points = points[:, :3]
    centered_points = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(points)
    normal = vh[2]
    projected_point = points.copy()
    for p in range(points.shape[0]):
         projected_point[p, :] = points[p, :] - np.dot(centered_points[p, :], normal) * normal

    # plot the plane and the points
    # Draw plane
    centroid = np.mean(points, axis=0)
    d = -centroid.dot(normal)
    X = centered_points
    xx, yy = np.meshgrid(np.arange(np.min(X[:, 0]), np.max(X[:, 0])), np.arange(np.min(X[:, 1]), np.max(X[:, 1])))

    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    # plot the surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx, yy, z, alpha=0.5)
    # ax.scatter(centered_points[:, 0], centered_points[:, 1], centered_points[:, 2])
    # ax.quiver(centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2], length = 10)
    # ax.scatter(projected_point[:, 0], projected_point[:, 1], projected_point[:, 2])

    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    C, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    xx, yy = np.meshgrid(
       np.linspace(points[:, 0].min(), points[:, 0].max(), 10),
       np.linspace(points[:, 1].min(), points[:, 1].max(), 10)
    )
    normal = (C[0], C[1], -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    zz = C[0] * xx + C[1] * yy + C[2]
    ax.plot_surface(xx, yy, zz, alpha=0.5, label='Fitted Plane')
    # prjection of 3D points 
    
    projected_point = points.copy()
    for p in range(points.shape[0]):
         projected_point[p, :] = points[p, :] - np.dot(centered_points[p, :], normal) * normal
    
    # ax.scatter(projected_point[:, 0], projected_point[:, 1], projected_point[:, 2])
    ax.scatter(points[: , 0], points[:, 1], points[:, 2])
    set_axes_equal(ax)

    plt.figure("2d")
    projected_point = projected_point - np.mean(projected_point, axis=0)
    plt.scatter(projected_point[:, 0], projected_point[:, 1])
    # set x and y axis between +/- 30
    plt.xlim((-30, 30))
    plt.ylim((-30, 30))
    
    return projected_point




def least_square_plane(points, target_names):
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    C, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    xx, yy = np.meshgrid(
       np.linspace(points[:, 0].min(), points[:, 0].max(), 10),
       np.linspace(points[:, 1].min(), points[:, 1].max(), 10)
    )
    zz = C[0] * xx + C[1] * yy + C[2]
    a, b, c = C
    # Plane normal: [-a, -b, 1]
    z_axis = np.array([-a, -b, 1.0])
    z_axis /= np.linalg.norm(z_axis)
    # Choose in-plane x_axis (e.g., projection of [1, 0, 0])
    x_guess = np.array([1.0, 0.0, 0.0])
    x_axis = x_guess - np.dot(x_guess, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    # Compute orthogonal y_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    CSFit = np.stack([x_axis, y_axis, z_axis], axis=1) 
    origin = np.mean(points, axis=0).tolist()

    projected_points = (points - origin)[:, :2] @ CSFit[:, :2].T
    plt.figure('2d')
    plt.scatter(points[:, 0], points[:, 1], c='r', marker='o')
    plt.scatter(projected_points[:, 0], projected_points[:, 1], c='g', marker='o')
    plt.plot(xx, yy, 'k-')
    plt.title('2D Plane Fitting')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', label='Points')
    ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], color='g', label='Projected Points')

    ax.plot_surface(xx, yy, zz, alpha=0.5, label='Fitted Plane')
    #plot x, y, z axis
    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r', label='X-axis'
              , length=10)
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g', label='Y-axis'
             , length=10 )
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b', label='Z-axis'
              , length=10)    # plot x, y and z vectors
    
    
    # reproject points on the 2D plane
    # projected_points = np.dot(points[:, :2], C[:2])

    # ax.scatter(projected_points[:, 0], projected_points[:, 1], points[:, 2], color='g', label='Projected Points')
    # plot x and y vectors
    set_axes_equal(ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Plane Fitting to Point Cloud")
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    file_dir = r'D:\Documents\Udem\Postdoctorat\Projet transfert nerveux\data\test_HB_001'
    trials = ['2', '3', '4', '5', '6']
    all_data = []
    for t in trials:
        trial = r'data_trial_test_mapping_HB00' + t+ '.pkl'
        data = _read_all_lines(os.path.join(file_dir, trial), data=[], merge=False)
        # if t in ['2', "3"]:
        data = data[:42]
        min_signal_shape = min([d["signal_data"]["data"].shape[0] for d in data if d["signal_data"]["data"].shape[0] > 2000])
        for d in data:
            d["signal_data"]["data"] = d["signal_data"]["data"][:min_signal_shape, :, None]
            d['brainsight_data']['position'] = np.array(d['brainsight_data']['position'])[:, None]
            d['brainsight_data']['target_position'] = np.array(d['brainsight_data']['target_position'])[:, None]

        idx_wrong = [i for i, d in enumerate(data) if d["signal_data"]["data"].shape[0] != min_signal_shape]
        if idx_wrong:
            shape_value = data[idx_wrong[0]]["signal_data"]["data"].shape[0]
            to_fill = min_signal_shape - shape_value
            data[idx_wrong[0]]["signal_data"]["data"] = np.concatenate([data[idx_wrong[0]]["signal_data"]["data"], np.zeros((to_fill, 6, 1))], axis=0)
        
        new_dict = None
        for d in data:
            new_dict = dic_merger(d, new_dict)
        all_data.append(new_dict)

    fig, ax = plt.subplots(3, 3, figsize=(10, 7))
    brainsight_data = [new_dict["brainsight_data"] for new_dict in all_data]
    position = [brainsight_data['position'].reshape(4, 4, -1).T for brainsight_data in brainsight_data] 
    signal_data = [new_dict["signal_data"]['data'][2100:2500, ...] for new_dict in all_data]
    # target_names = brainsight_data["target_name"]
    for k in range(len(position)):
        trans_list = [p[:, 3, :] for p in position[k:k+1]]
        rot_list = [p[:, :3, :] for p in position[k:k+1]]
        trans, r = np.concatenate(trans_list, axis=0), np.concatenate(rot_list, axis=0)
        # target_pos = brainsight_data['target_position'].reshape(4, 4, -1).T
        # trans_target = target_pos[:, 3, :]
        signal_data_tmp = np.concatenate(signal_data[k:k+1], axis=-1)
        # compute the peak to peak amplitude for each channel according x and y position of trans
        peak_to_peak = np.ptp(signal_data_tmp, axis=0)
        projected = project_points_to_plane(trans)
        # least_square_plane(trans, target_names)
        # eigen_vectors = compute_normal(trans)
        # plot plane and vectors 
        # plot_points(trans, eigen_vectors)

        # plot the 2d plane of the signal data
        x, y, z = projected[:, 0], projected[:, 1], projected[:, 2]
        n_point_grid = 50
        for i in range(1, 4):#peak_to_peak.shape[0]):
            # plt.subplot(3, 3, i)
            z = peak_to_peak[i, :]

            # Define grid
            xi = np.linspace(x.min(), x.max(), n_point_grid)
            yi = np.linspace(y.min(), y.max(), n_point_grid)
            xi, yi = np.meshgrid(xi, yi)

            from scipy.interpolate import griddata
            # Interpolate
            zi = griddata((x, y), z, (xi, yi), method='cubic') 
            zi = zi
            # plot interpolated 2d plane
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.plot_surface(xi, yi, zi, cmap='jet')
            # plt.contourf(xi, yi, zi, n_point_grid, cmap='jet')
            # plt.colorbar()
            # plt.scatter(x, y, c='k', marker='o')
            # plt.show()
            ax[k, i-1].contourf(xi, yi, zi, n_point_grid, cmap='jet')

            # target_trans, target_R = target_pos[:, 3, :], target_pos[:, :3, :]
            # # plot trans in 3D
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(trans[:, 0], trans[:, 1], trans[:, 2], c='b', marker='o')
            # ax.scatter(target_trans[:, 0], target_trans[:, 1], target_trans[:, 2], c='r', marker='o')
            # # plot vector x, y and z
            # # ax.quiver(trans[:, 0], trans[:, 1], trans[:, 2], R[:, 0, :], R[:, 1, :], R[:, 2, :], length=0.1, normalize=True)

            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
        plt.show()

        # time = signal_data[:, 0, :]
        # for j in range(signal_data.shape[-1]):
        #     for i in range(1, signal_data.shape[1]):
        #         plt.subplot(3, 3, i+1)
        #         plt.plot(time[:, j], signal_data[:, i, j])
        # plt.show()