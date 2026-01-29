import os
from matplotlib import colors, markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from biosiglive import load


def min_map(grid_data_frame, part, euclid=True, c="grid"):
    if euclid:
        cor_coef = grid_data_frame.loc[grid_data_frame['participant'] == part].loc[grid_data_frame['euclid_cog_error'] <= 3.6]
        # cor_coef = grid_data_frame.loc[grid_data_frame["participant"] == part].loc[
        #     grid_data_frame["kl_divergence"] <= 2
        # ]
    else:
        cor_coef = grid_data_frame.loc[grid_data_frame["participant"] == part].loc[
            grid_data_frame["correlation_coefficient"] >= 0.9
        ]
    map_list = [np.nan, np.nan, np.nan]
    if not cor_coef.empty:
        for m, muscle in enumerate(cor_coef["muscle"].unique()):
            pd_tmp_muscle = cor_coef.loc[cor_coef["muscle"] == muscle]
            min_map = pd_tmp_muscle["map_number"].min()
            map_list[m] = min_map
    else:
        return map_list
    return map_list


def recompute_correlation(participants, grid_data_frame, muscle_list):
    for p in participants:
        cor_coef = [[np.nan for _ in range(len(muscle_list))]] + [
            [
                pearsonr(
                    grid_data_frame.loc[grid_data_frame["participant"] == p]
                    .loc[grid_data_frame["map_number"] == i + 1]
                    .loc[grid_data_frame["muscle"] == m]["zgf_list"]
                    .values[0]
                    .flatten(),
                    grid_data_frame.loc[grid_data_frame["participant"] == p]
                    .loc[grid_data_frame["map_number"] == i]
                    .loc[grid_data_frame["muscle"] == m]["zgf_list"]
                    .values[0]
                    .flatten(),
                )[0]
                for m in muscle_list
            ]
            for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "correlation_coefficient"] = sum(cor_coef, [])
    return grid_data_frame


def recompute_euclid_dist(participants, grid_data_frame, muscle_list):
    for p in participants:
        cog_err_eucl = [[np.nan for _ in range(len(muscle_list))]]
        cog_err_eucl = cog_err_eucl + [
            [
                np.linalg.norm(
                    np.array(
                        [
                            grid_data_frame.loc[grid_data_frame["participant"] == p]
                            .loc[grid_data_frame["map_number"] == i + 1]
                            .loc[grid_data_frame["muscle"] == m]["x_cog"],
                            grid_data_frame.loc[grid_data_frame["participant"] == p]
                            .loc[grid_data_frame["map_number"] == i + 1]
                            .loc[grid_data_frame["muscle"] == m]["y_cog"],
                        ]
                    )
                    - np.array(
                        [
                            grid_data_frame.loc[grid_data_frame["participant"] == p]
                            .loc[grid_data_frame["map_number"] == i]
                            .loc[grid_data_frame["muscle"] == m]["x_cog"],
                            grid_data_frame.loc[grid_data_frame["participant"] == p]
                            .loc[grid_data_frame["map_number"] == i]
                            .loc[grid_data_frame["muscle"] == m]["y_cog"],
                        ]
                    ),
                    axis=0,
                )[0]
                for m in muscle_list
            ]
            for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "euclid_cog_error"] = sum(cog_err_eucl, [])
    return grid_data_frame


def rmse(ref, pred):
    return np.sqrt(np.mean((ref - pred) ** 2))


def recompute_area_error(participants, grid_data_frame, muscle_list):
    for p in participants:
        cog_err_eucl = [[np.nan for _ in range(len(muscle_list))]]
        cog_err_eucl = cog_err_eucl + [
            [
                rmse(
                    np.array(
                        grid_data_frame.loc[grid_data_frame["participant"] == p]
                        .loc[grid_data_frame["map_number"] == i + 1]
                        .loc[grid_data_frame["muscle"] == m]["area"]
                    ),
                    np.array(
                        grid_data_frame.loc[grid_data_frame["participant"] == p]
                        .loc[grid_data_frame["map_number"] == i]
                        .loc[grid_data_frame["muscle"] == m]["area"]
                    ),
                )
                for m in muscle_list
            ]
            for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "area_error"] = sum(cog_err_eucl, [])
    return grid_data_frame


def recompute_volume_error(participants, grid_data_frame, muscle_list):
    for p in participants:
        cog_err_eucl = [[np.nan for _ in range(len(muscle_list))]]
        cog_err_eucl = cog_err_eucl + [
            [
                rmse(
                    np.array(
                        grid_data_frame.loc[grid_data_frame["participant"] == p]
                        .loc[grid_data_frame["map_number"] == i + 1]
                        .loc[grid_data_frame["muscle"] == m]["volume"]
                    ),
                    np.array(
                        grid_data_frame.loc[grid_data_frame["participant"] == p]
                        .loc[grid_data_frame["map_number"] == i]
                        .loc[grid_data_frame["muscle"] == m]["volume"]
                    ),
                )
                for m in muscle_list
            ]
            for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "volume_error"] = sum(cog_err_eucl, [])
    return grid_data_frame



def plot(df, x, y, hue=None, name="", ax=None, pseudo=False, legend=True, palette="rocket"):
    # fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=3, num=name)
    muscle_name = ["fdi", "ext_comm", "sup"]
    muscle_name_title = ["FDI", "Extenseur commun", "Supinateur"]
    # replace muscles name in dataframe by ['FDI', 'Extenseur commun', 'Supinateur']
    # df['muscle'] = df['muscle'].replace({'fdi': 'FDI', 'ext_comm': 'Extenseur commun','sup': 'Supinateur'})

    # for j in range(3):
    if ax is None:
        plt.figure(name)
        ax = plt.gca()
    sns.lineplot(
        df,  # .loc[df["muscle"] == muscle_name[j]],
        x=x,
        y=y,
        hue=hue,
        marker="o",
        ax=ax,
        legend=legend,
        palette=palette,
        err_style="band",
        errorbar=("sd", 1),
    )
    # if j != 2:
    #     ax[j].get_legend().remove()
    if "correlation" in y:
        ax.axhline(y=0.9, color="r", linestyle="--")
        ax.set_ylabel("Correlation coefficient")
        ax.set_xlabel("Number of stimulations")
    if "euclid" in y:
        ax.axhline(y=3.6, color="g", linestyle="--")
        ax.set_ylabel("Euclidian distance")
    if "divergence" in y:
        ax.axhline(y=0.02, color="g", linestyle="--")
        ax.set_ylabel("KL divergence")
    # if pseudo:
    #     ax.set_xticks(list(range(1, 7)))
    #     ax.set_xticklabels([str(i) for i in [44, 64, 94, 124, 154, 184]])
    #     ax.set_xlabel('Number of points')
    # else:
    #     ax.set_xticks(list(range(1, 5)))
    #     ax.set_xticklabels([str(i) for i in [98, 147, 196, 245]])
    #     ax.set_xlabel('Map number')

    # plt.savefig(f"{name}.png")


if __name__ == "__main__":
    result_dir = "smooth_5_5_10"
    data_frame = pd.read_csv(os.path.join(result_dir, "maps_characteristics.csv"))

    data_maps = load(os.path.join(result_dir, "maps_values.bio"), merge=False)[:144]
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
    # merge dataframe
    data_frame = pd.merge(data_frame, data_frame_tot, on=["participant", "map_number", "condition", "muscle"])

    data_frame = data_frame.loc[data_frame["participant"] != "P006_TN"]
    participants = data_frame["participant"].unique()

    condition = ["grid", "pseudo"]
    # sns.set(font_scale=1.5)

    df_cond = pd.DataFrame()
    for c, cond in enumerate(condition):
        grid_data_frame = data_frame.loc[data_frame["condition"] == cond].loc[
            data_frame["participant"].isin(participants)
        ]
        grid_data_frame = grid_data_frame.loc[grid_data_frame["muscle"].isin(list(grid_data_frame["muscle"][:3]))]
        muscle_list = list(data_frame["muscle"][:3])
        # , 'area_error', 'volume_error']
        grid_data_frame = recompute_correlation(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_euclid_dist(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_area_error(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_volume_error(participants, grid_data_frame, muscle_list)
        grid_data_frame = recompute_kl_divergence(participants, grid_data_frame, muscle_list)

        df_cond = pd.concat([df_cond, grid_data_frame], ignore_index=True)
        # pointplot wit no line
        # sns.pointplot(y='participant', x='min_map_number', hue='muscle', data=pd_maps, ax=axes[2, c], join=False)
        # sns.pointplot(y='participant', x='min_map_number', hue='muscle', data=pd_maps, ax=axes[2, c], join=False, dodge=1)
        # same colors for the swarmplot
        # black in rgbda
        # color_palette = {'fdi': 'k', 'ext_comm': 'k','sup': 'k'}
        # different markers for the swarmplot
        # markers = ['o', 'v', 'D']
        # sns.pointplot(y='participant',
        #               orient='v', join=False,
        #               x='min_map_number', hue='muscle', data=pd_maps, ax=axes[2, c],
        #                   dodge=True, palette=color_palette, markers=markers, scale=0.5,
        #                   errorbar=None)

        # for k, key in enumerate(keys):
        #     ax = axes[k, c]
        #     df_tmp = grid_data_frame[["map_number", key, "muscle", 'participant']]
        #     plot(df_tmp, "map_number", key, None, ax=ax, pseudo=cond=='pseudo')
        #     if k == 0:
        #         ax.set_title(f"{cond}")
    df_cond = df_cond[
        [
            "map_number",
            "correlation_coefficient",
            "euclid_cog_error",
            "kl_divergence",
            "muscle",
            "participant",
            "condition",
        ]
    ].dropna()
    df_cond.loc[df_cond["condition"] == "grid", "map_number"] = (
        df_cond.loc[df_cond["condition"] == "grid", "map_number"].values + 1
    ) * 49
    # replace 1 by 44, 2 by 64, 3 by 94, 4 by 124, 5 by 154, 6 by 184
    list_number = [44, 64, 94, 124, 154, 184]
    df_cond.loc[df_cond["condition"] == "pseudo", "map_number"] = df_cond.loc[
        df_cond["condition"] == "pseudo", "map_number"
    ].apply(lambda x: list_number[x - 1])
    # set svg font to none
    plt.rcParams["svg.fonttype"] = "none"
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(nrows=1, ncols=2)
    colors = [[0.38092887, 0.12061482, 0.32506528], [0.7965014, 0.10506637, 0.31063031]]
    keys = ["euclid_cog_error", "correlation_coefficient"]
    keys = ["kl_divergence", "correlation_coefficient"]
    for i in range(2):
        if i == 0:
            axes = subfigs[i].subplots(nrows=2, ncols=1, sharex=True)
            for k, key in enumerate(keys):
                ax = axes[k]
                if k == 0:
                    ax.set_title("a) Evolution of errors")

                # create palette with two colors
                plot(df_cond, "map_number", key, "condition", ax=ax, legend=k == 0, palette=sns.color_palette(colors))
                if k == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, ["Grid", "Pseudo"], title="Method", frameon=False)

                # if k == 0:
                #     colors = [patch.get_edgecolor()[0][:-1].tolist() for patch in ax.collections]
                if k == 0:
                    ax_twin = ax.twiny()
                    ticks_location = ax.get_xticks()
                    # set ticks as 4 * ticks
                    ax_twin.set_xticks(np.linspace(df_cond.map_number.min(), df_cond.map_number.max(), 10))
                    ticks_location = ax_twin.get_xticks()
                    ax_twin.set_xticklabels([str(np.round((i * 4) / 60, 1)) for i in ticks_location])
                    # set x ticks label from ax min to ax max
                    ax_twin.set_xlabel("Time (min)")
                    ax_twin.set_xlim(ax.get_xlim())
        else:
            pd_maps = pd.DataFrame()
            for part in participants:
                for c in condition:
                    maps_euclid = min_map(df_cond.loc[df_cond["condition"] == c], part, euclid=True, c=cond)
                    maps_corr = min_map(df_cond.loc[df_cond["condition"] == c], part, euclid=False, c=cond)
                    pd_tmp = pd.DataFrame(
                        {
                            "participant": [part] * 3,
                            "condition": [c] * 3,
                            "map_number_euclid": maps_euclid,
                            "correlation_coefficient": maps_corr,
                            "muscle": ["fdi", "ext_comm", "sup"],
                        }
                    )
                    if pd_maps.empty:
                        pd_maps = pd_tmp
                    else:
                        pd_maps = pd.concat([pd_maps, pd_tmp], ignore_index=True)
            pd_maps["min_map_number"] = pd_maps[["map_number_euclid", "correlation_coefficient"]].max(axis=1)
            # pd_maps["min_map_number"] = pd_maps[["kl_divergence", "correlation_coefficient"]].max(axis=1)

            axes = subfigs[i].subplots(nrows=1, ncols=1)
            max_value_axis = pd_maps.min_map_number.max() + 50
            ax = axes
            ax.set_title("b) Optimal number of stimulation needed")
            sns.violinplot(y="min_map_number", x="condition", hue="muscle", data=pd_maps, ax=ax, legend=False, cut=0)
            colors_violin = [
                colors[0] + [1],
                colors[0] + [0.6],
                colors[0] + [0.2],
                colors[1] + [1],
                colors[1] + [0.6],
                colors[1] + [0.2],
            ]
            for p, patch in enumerate(ax.collections):
                patch.set_facecolor(colors_violin[p])
            ax.set_ylim(ax.get_ylim()[0], max_value_axis)
            ax.set_ylabel("Optimal stimulation number")
            yticks = ["Grid", "Pseudo-random"]
            ax.set_xticklabels([yticks[i] for i in ax.get_xticks()])
            pos_x = [-0.27, 0, 0.27, 0.73, 1, 1.27]
            pos_y = [260, 260, 260, 130, 130, 130]
            text = ["FDI", "EXT", "SUP", "FDI", "EXT", "SUP"]
            for t, te in enumerate(text):
                ax.text(pos_x[t], pos_y[t], te, ha="center", color=colors_violin[t][:-1] + [1])
            ax.set_xlabel("Method")
            # ax_ytwin = ax.twinx()
            # # set ticks as 4 * ticks
            # ax_ytwin.set_yticks(np.linspace(pd_maps.min_map_number.min(), pd_maps.min_map_number.max() + 50, 10))
            # ticks_location = ax_ytwin.get_yticks()
            # ax_ytwin.set_yticklabels([str(np.round((i * 4)/60, 1)) for i in ticks_location])
            # # set x ticks label from ax min to ax max
            # ax_ytwin.set_ylabel('Time (min)')
    plt.show()
