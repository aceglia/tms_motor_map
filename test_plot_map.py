import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # create a rectangular 2D point cloud with points spaced of 10 mm appart and of size 60x60 mm
    x = np.arange(0, 60, 10).astype(float)
    y = np.arange(0, 60, 10).astype(float)
    # add noise in the x and y coordinates
    x += np.random.normal(0, 1, x.shape)
    y += np.random.normal(0, 1, y.shape)

    xx, yy = np.meshgrid(x, y)
    z = np.zeros_like(xx).astype(float)
    # add some random noise to the point cloud
    z += np.random.normal(0, 1, xx.shape)

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

    # plot 3d

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")  # plot the point cloud using matplotlib
    ax.scatter(xx, yy, z, c="r", marker="o")
    set_axes_equal(ax)
    plt.show()
