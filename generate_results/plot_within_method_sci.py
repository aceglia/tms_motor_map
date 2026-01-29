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
        #     grid_data_frame["kl_divergence"] <= 0.02
        # ]
    else:
        cor_coef = grid_data_frame.loc[grid_data_frame["participant"] == part].loc[
            grid_data_frame["correlation_coefficient"] >= 0.9
        ]
    map_list = []
    if not cor_coef.empty:
        for muscle in cor_coef["muscle"].unique():
            pd_tmp_muscle = cor_coef.loc[cor_coef["muscle"] == muscle]
            min_map = pd_tmp_muscle["map_number"].min()
            map_list.append(min_map)
    else:
        return [-1] * 3
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


def kl_divergence(img_new, img_old, eps=1e-12, log_fct="2", jsd=False, root=False):
    if log_fct == "2":
        log_fct = np.log2
    elif log_fct == "10":
        log_fct = np.log10
    elif log_fct == "e":
        log_fct = np.log
    else:
        raise RuntimeError("Not recognized log fucntion")
    p = np.clip(img_new.astype(float), eps, 1) / np.sum(img_new.astype(float))
    q = np.clip(img_old.astype(float), eps, 1) / np.sum(img_old.astype(float))
    m = 0.5 * (p + q)
    if jsd:
        value = 0.5 * (np.sum(p * log_fct(p / m)) + np.sum(q * log_fct(q / m)))
        return np.sqrt(value) if root else value
    
    return np.sum(p * log_fct(p / q))


def recompute_kl_divergence(participants, grid_data_frame, muscle_list):
    for p in participants:
        kl_div = [[np.nan for _ in range(len(muscle_list))]]
        kl_div = kl_div + [
            [
                kl_divergence(
                    np.array(
                        grid_data_frame.loc[grid_data_frame["participant"] == p]
                        .loc[grid_data_frame["map_number"] == i]
                        .loc[grid_data_frame["muscle"] == m]["zgf_list"]
                    )[0],
                    np.array(
                        grid_data_frame.loc[grid_data_frame["participant"] == p]
                        .loc[grid_data_frame["map_number"] == i + 1]
                        .loc[grid_data_frame["muscle"] == m]["zgf_list"]
                    )[0],
                    log_fct="e",
                    jsd=True,
                    root=False
                )
                for m in muscle_list
            ]
            for i in range(len(np.unique(grid_data_frame["map_number"])) - 1)
        ]
        grid_data_frame.loc[grid_data_frame["participant"] == p, "kl_divergence"] = sum(kl_div, [])
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
    plt.margins(0)
    # if j != 2:
    #     ax[j].get_legend().remove()
    ax_xlim = ax.get_xlim()
    if "correlation" in y:
        ax.axhline(y=0.9, color="r", linestyle="--")
        ax.set_ylabel("Correlation coefficient")
        ax.set_xlabel("Number of stimulations")
    if "euclid" in y:
        ax.axhline(y=3.6, color="g", linestyle="--")
        ax.set_ylabel("Euclidian distance")
    if "divergence" in y:
        ax.axhline(y=0.02, color="g", linestyle="--")
        ax.set_ylabel("Euclidian distance")
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
    result_dir = "sci_smooth_7_3_10"
    result_dir = "sci_smooth_10_5_2543"
    data_frame = pd.read_csv(os.path.join(result_dir, "maps_characteristics.csv"))

    data_maps = load(os.path.join(result_dir, "maps_values.bio"), merge=False)[-144:]
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

    list_number = [44, 64, 94, 124, 154, 184]
    df_cond.loc[df_cond["condition"] == "pseudo", "map_number"] = df_cond.loc[
        df_cond["condition"] == "pseudo", "map_number"
    ].apply(lambda x: list_number[x - 1])
    # df_cond = df_cond.loc[~(df_cond['muscle'] == 'fdi')]
    muscle_list = ["ext_comm", "sup"]
    df_cond = df_cond.loc[df_cond["muscle"].isin(muscle_list)]
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
                plot(df_cond, "map_number", key, "condition", ax=ax, legend=k == 0, palette=sns.color_palette(colors))

                if k == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, ["Grid", "Pseudo"], title="Method", frameon=False)
                    ax.set_title("a) Evolution of errors")

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
                            "participant": [part] * len(muscle_list),
                            "condition": [c] * len(muscle_list),
                             'map_number_euclid': maps_euclid,
                            # "kl_divergence": maps_euclid,
                            "correlation_coefficient": maps_corr,
                            "muscle": muscle_list,
                        }
                    )
                    if pd_maps.empty:
                        pd_maps = pd_tmp
                    else:
                        pd_maps = pd.concat([pd_maps, pd_tmp], ignore_index=True)
            pd_maps["min_map_number"] = pd_maps[["map_number_euclid", "correlation_coefficient"]].max(axis=1)
            axes = subfigs[i].subplots(nrows=1, ncols=1)
            ax = axes
            ax.set_title("b) Optimal number of stimulation needed")

            colors_violin = [colors[0] + [0.6], colors[1] + [0.6], colors[0] + [0.2], colors[1] + [0.2]]
            sns.barplot(
                y="min_map_number",
                x="condition",
                hue="muscle",
                data=pd_maps,
                ax=ax,
                legend=False,
                gap=0.2,
                palette=sns.color_palette(colors_violin),
            )
            count = 0
            for c, cont in enumerate(ax.containers):
                for patch in cont.patches:
                    patch.set_facecolor(colors_violin[count])
                    count += 1

            ax.set_ylabel("Optimal stimulation number")
            yticks = ["Grid", "Pseudo-random"]
            ax.set_xticklabels([yticks[i] for i in ax.get_xticks()])
            pos_x = [-0.2, 0.2, 0.8, 1.2]
            pos_y = [100, 100, 70, 70]
            text = ["EXT", "SUP", "EXT", "SUP"]
            colors_violin = [colors[0] + [0.6], colors[0] + [0.2], colors[1] + [0.6], colors[1] + [0.2]]
            for t, te in enumerate(text):
                ax.text(pos_x[t], pos_y[t], te, ha="center", color=colors_violin[t][:-1] + [1])
            ax.set_xlabel("Method")
            # ax_ytwin = ax.twinx()
            # # set ticks as 4 * ticks
            # ax_ytwin.set_yticks(np.linspace(0, pd_maps.min_map_number.max(), 10))
            # ticks_location = ax_ytwin.get_yticks()
            # ax_ytwin.set_yticklabels([str(np.round((i * 4)/60, 1)) for i in ticks_location])
            # set x ticks label from ax min to ax max

            # ax_ytwin.set_ylabel('Time (min)')

    plt.show()
