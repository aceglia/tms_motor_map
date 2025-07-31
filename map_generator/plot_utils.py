import matplotlib.pyplot as plt
import numpy as np


def plot_3D_points(points, plane=None, show=True, ax_3d=None):
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # if plane is not None:
    #     (x, y, z), com = plane
    #     # create plane coordinates system
    #     projected = np.array([self._project_point_onto_plane(p, com, z) for p in points])
    #     local_projeted = np.array([self._to_plane_coordinates(p_proj, com, x, y) for p_proj in projected])
    #     rotated_local = self._rotate_points(local_projeted)
    # create local coordinates system using the points -1 and -6
    # plt.scatter(local_projeted[:, 0], local_projeted[:, 1], c='r')
    # plt.scatter(np.mean(local_projeted[:, 0]), np.mean(local_projeted[:, 1]), c='g', marker='x')
    # bounding box square
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], c="k")
    # ideal square
    x_min, x_max = -30, 30
    y_min, y_max = -30, 30
    # move to points center
    x_min, x_max = x_min - np.mean(points[:, 0]), x_max - np.mean(points[:, 0])
    y_min, y_max = y_min - np.mean(points[:, 1]), y_max - np.mean(points[:, 1])
    plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], c="k", linestyle="--")

    # ideal points
    plt.scatter(points[:, 0], points[:, 1], c="k")
    plt.scatter(points[0, 0], points[0, 1], c="r")

    plt.scatter(points[-1, 0], points[-1, 1], c="g")
    plt.scatter(points[-6, 0], points[-6, 1], c="b")
    # plt.scatter(ideal_points[:, 0], ideal_points[:, 1], c='b')

    # make axes equal
    x_min, x_max = plt.gca().get_xlim()
    y_min, y_max = plt.gca().get_ylim()
    plt.xlim(min(x_min, y_min), max(x_max, y_max))
    plt.ylim(min(x_min, y_min), max(x_max, y_max))

    # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # self.set_axes_equal(ax)
    plt.show()

    if show:
        plt.show()


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


def plot_plane(points, com, normal, fig_name=None):
    fig = plt.figure(figsize=(10, 7), num=fig_name)
    ax = fig.add_subplot(111, projection="3d")
    xx, yy = np.meshgrid(
        np.linspace(points[:, 0].min(), points[:, 0].max(), 30), np.linspace(points[:, 1].min(), points[:, 1].max(), 30)
    )
    d = -com.dot(normal)
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1.0 / normal[2]
    ax.plot_wireframe(xx, yy, zz, alpha=0.5, color="k")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="k")
    colors = ["r", "g", "b", "c"]
    for i, idx in enumerate([0, 5, -6, -1]):
        ax.scatter(points[:42, :][idx, 0], points[:42, :][idx, 1], points[:42, :][idx, 2], c=colors[i], s=70)
    set_axes_equal(ax)


def plot_heatmap(points, mep, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    x, y = points[:, 0], points[:, 1]
    z = np.ptp(mep, axis=0)
    # x=np.unique(x)
    # y=np.unique(y)
    x = np.arange(0, 7)
    y = np.arange(0, 6)
    X, Y = np.meshgrid(x, y)
    Z = z.reshape(len(x), len(y))
    Z = np.transpose(Z)
    data = Z
    # plt.contour(X, Y, Z, 7, linewidths = 0.5, colors = 'k')
    im = ax.imshow(data, cmap="jet", origin="lower", aspect="equal", extent=[min(x), max(x), min(y), max(y)])
    plt.colorbar(im, ax=ax)


def plot_single_map(x, y, zgf, ax=None, n_point_grid=50, x_cog=None, y_cog=None, area=None, volume=None):

    if ax is None:
        fig, ax = plt.subplots()
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)

    xi = np.linspace(x_min, x_max, n_point_grid)
    yi = np.linspace(y_min, y_max, n_point_grid)
    xi, yi = np.meshgrid(xi, yi)

    ax.contourf(xi, yi, zgf, n_point_grid, cmap="jet")
    ax.scatter(x, y, c="k", s=4, alpha=0.1, marker="o")
    # if x_cog is not None and y_cog is not None:
    #     ax.scatter(x_cog, y_cog, c="k", s=150, marker="x")
    # ax.text(
    #     0.05,
    #     0.95,
    #     f"area: {area:.2f} mm^2\nvolume: {volume:.2f} mm^3",
    # )
    # ax.set_xlabel("antero-posterior (mm)")
    # ax.set_ylabel("latero-medial (mm)")


def plot_map(x_y_z, n_point_grid=50):
    x, y, zgf = x_y_z[0], x_y_z[1], x_y_z[2]
    fig, ax = plt.subplots(1, len(zgf), sharex=True, sharey=True)
    for i in range(len(zgf)):
        plot_single_map(x[i], y[i], zgf[i], ax[i], n_point_grid)


def plot_2d_points(points, ax=None, color="k", colorized_points=None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(points[:, 0], points[:, 1], color=color)
    if colorized_points is not None:
        idxs, colors = colorized_points
        _ = [ax.scatter(points[idxs[i], 0], points[idxs[i], 1], color=colors[i]) for i in range(len(idxs))]
    ax.set_aspect("equal")
    return ax
