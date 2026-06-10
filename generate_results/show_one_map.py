from biosiglive import load
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np



def plot_single_map(x, y, zgf, ax=None, n_point_grid=50, x_cog=None, y_cog=None, x_real=None, y_real=None):
    if ax is None:
        _, ax = plt.subplots()
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    xi = np.linspace(x_min, x_max, n_point_grid)
    yi = np.linspace(y_min, y_max, n_point_grid)
    xi, yi = np.meshgrid(xi, yi)
    x_flat = xi.flatten()
    y_flat = yi.flatten()
    z_flat = zgf.flatten()
    x = x_flat[z_flat > np.max(z_flat) * 0.1]
    y = y_flat[z_flat > np.max(z_flat) * 0.1]
    if x.size < 4:
        return 0, 0
    ax.contourf(xi, yi, zgf, n_point_grid, cmap="jet")
    if x_real is not None and y_real is not None:
        ax.scatter(x_real, y_real, s=8, alpha=0.5, marker="o", facecolors="none", edgecolors="black", linewidths=0.8)
    # if x_cog is not None and y_cog is not None:
    #     ax.scatter(x_cog, y_cog, c="k", s=150, marker="x")


if __name__ == "__main__":
    font_base = 14
    big = font_base + 2
    small = font_base - 2
    bigger = font_base + 4
    sci = True
    result_dir = (
        r"D:\Documents\Programmation\tms_motor_map\results\smooth_6_6_0_ransac"
        if not sci
        else r"D:\Documents\Programmation\tms_motor_map\results\smooth_6_6_0_ransac_sci"
    )
    data_frame = pd.read_csv(os.path.join(result_dir, "maps_characteristics.csv"))

    data_maps = load(os.path.join(result_dir, "maps_values.bio"), merge=False)
    data_frame_tot = pd.DataFrame()
    for i in range(len(data_maps)):
        data_frame_tmp = pd.DataFrame(
            {
                "participant": data_maps[i]["participant"] * 5,
                "map_number": data_maps[i]["map_number"] * 5,
                "x_list": data_maps[i]["x_list"],
                "y_list": data_maps[i]["y_list"],
                "zgf_list": data_maps[i]["zgf_list"],
                "condition": data_maps[i]["condition"] * 5,
                "xgf_list": data_maps[i]["xgf_list"],
                "ygf_list": data_maps[i]["ygf_list"],
                "muscle": list(data_frame["muscle"][:5]),
            }
        )
        if data_frame_tot.empty:
            data_frame_tot = data_frame_tmp
        else:
            data_frame_tot = pd.concat([data_frame_tot, data_frame_tmp])

    min_maps = pd.read_csv(os.path.join(result_dir, "maps_min_map.csv"))
    list_points_tot = [
        [49, 98, 147, 196, 245],
        # [34, 64, 94, 124, 154, 184],
        # [34, 64, 94, 124, 154, 184]
        [24, 44, 64, 84, 104, 124, 144, 164, 184],
        [24, 44, 64, 84, 104, 124, 144, 164, 184],
    ] if not sci else [
        [49, 98, 147, 196],
        [24, 44, 64, 84, 104, 124, 144, 164, 184],
        [24, 44, 64, 84, 104, 124, 144, 164, 184],
    ]
    # for participant in data_frame_tot['participant'].unique():
    participant_to_plot = "008_TN" if not sci else "004_TN_SCI"
    n_to_plot = [[49, -1, 245], [24, -1, 184]] if not sci else [[49, -1, 196], [24, -1, 184]]
    map_num = [[0, -1, 4], [0, -1, 8]]

    fig = plt.figure(constrained_layout=True, num=participant_to_plot)
    cond_y_label = ["Grid", "Pseudo-random"]
    subfigs = fig.subfigures(nrows=2, ncols=2)
    titles = ['First', 'Tailored', 'Full-size']
    for c, cond in enumerate(["grid", "pseudo"]):
        for r, muscle in enumerate(["sup", "ext_comm"]):
            if r == 0:
                subfigs[r, c].suptitle(f"{cond_y_label[c]}", fontsize=bigger + 2)
            fig.supylabel(f"POST-ANT (mm)", fontsize=bigger)
            fig.supxlabel(f"LAT-MED (mm)" , fontsize=bigger)
            axes = subfigs[r, c].subplots(nrows=1, ncols=len(n_to_plot[c]))
            prev_maps = None
            for m, map_number in enumerate(n_to_plot[c]):
                if map_number == -1:
                    map_number = min_maps.loc[
                        (min_maps["participant"] == participant_to_plot)
                        & (min_maps["muscle"] == muscle)
                        & (min_maps["condition"] == cond)
                    ]["min_map_number"].values[0]
                data_frame_tmp_map = data_frame_tot.loc[
                    (data_frame_tot["participant"] == participant_to_plot)
                    & (data_frame_tot["muscle"] == muscle)
                    & (data_frame_tot["condition"] == cond)
                    & (data_frame_tot["map_number"] == list_points_tot[c].index(int(map_number)))
                ]
                x_list = data_frame_tmp_map["xgf_list"].values[0]
                y_list = data_frame_tmp_map["ygf_list"].values[0]
                z_list = data_frame_tmp_map["zgf_list"].values[0]
                x_real = data_frame_tmp_map["x_list"].values[0]
                y_real = data_frame_tmp_map["y_list"].values[0]
                ax = axes[m]
                plot_single_map(x_list, y_list, z_list, ax, 50, x_real=x_real, y_real=y_real)
                if r == 0:
                    ax.set_title(titles[m], fontsize=big)
                if r == 0 and c == 0 and m == 0:
                    ax.set_ylabel("SUP", fontsize=big)
                if r == 1 and c == 0 and m == 0:
                    ax.set_ylabel("EDC", fontsize=big)
                for tick in ax.get_xticklabels():
                    tick.set_fontsize(font_base)
                for tick in ax.get_yticklabels():
                    tick.set_fontsize(font_base)
                ax.set_aspect("equal")
    plt.show(block=True)
