import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from biosiglive import load
from utils import *


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
        df, #.loc[df["muscle"] == muscle_name[j]],
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
        ax.set_ylabel("Correlation coefficient", fontsize=16)
        ax.set_xlabel("Number of stimulations", fontsize=18)
    if "euclid" in y:
        ax.axhline(y=3.6, color="g", linestyle="--")
        ax.set_ylabel("Euclidian distance", fontsize=14)
    if "divergence" in y:
        ax.axhline(y=0.15, color="g", linestyle="--")
        ax.set_ylabel("KL divergence", fontsize=16)
    # if pseudo:
    #     ax.set_xticks(list(range(1, 7)))
    #     ax.set_xticklabels([str(i) for i in [44, 64, 94, 124, 154, 184]])
    #     ax.set_xlabel('Number of points')
    # else:
    #     ax.set_xticks(list(range(1, 5)))
    #     ax.set_xticklabels([str(i) for i in [98, 147, 196, 245]])
    #     ax.set_xlabel('Map number')

    # plt.savefig(f"{name}.png")


def main(args):
    (path_tmp, df_path, save_path, csv_path) = args
    if not os.path.exists(path_tmp):
        path_tmp = path_tmp.replace(".bio", "_test.bio")
    data_maps = load(path_tmp, merge=False)
    data_frame = pd.read_csv(df_path)
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

    data_frame = data_frame.loc[data_frame["participant"] != "006_TN"]
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
    # list_number = [44, 64, 94, 124, 154, 184]
    list_number = [64, 94, 124, 154, 184]
    list_number = [44, 64, 84, 104, 124, 144, 164, 184]
    df_cond.loc[df_cond["condition"] == "pseudo", "map_number"] = df_cond.loc[
        df_cond["condition"] == "pseudo", "map_number"
    ].apply(lambda x: list_number[x - 1])
    # set svg font to none
    plt.rcParams["svg.fonttype"] = "none"
    font_base = 16
    big = font_base + 2
    bigger = font_base + 4
    small = font_base - 2
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
                    ax.set_title("a) Criteria evolution", fontsize=bigger)

                # create palette with two colors
                plot(df_cond, "map_number", key, "condition", ax=ax, legend=k == 0, palette=sns.color_palette(colors))
                if k == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, ["Grid", "Pseudo"], title="", frameon=False, fontsize=font_base)
                # ax.set_yticklabels([np.round(i, 2) for i in ax.get_yticks()], fontsize=small)
                # ax.set_xticklabels([np.round(i, 2) for i in ax.get_xticks()], fontsize=small)
                ax.tick_params(axis='x', labelrotation=0, labelsize=small)
                ax.tick_params(axis='y', labelrotation=0, labelsize=small)
                # if k == 0:
                #     colors = [patch.get_edgecolor()[0][:-1].tolist() for patch in ax.collections]
                # if k == 0:
                #     ax_twin = ax.twiny()
                #     ticks_location = ax.get_xticks()
                #     # set ticks as 4 * ticks
                #     ax_twin.set_xticks(np.linspace(df_cond.map_number.min(), df_cond.map_number.max(), 10))
                #     ticks_location = ax_twin.get_xticks()
                #     ax_twin.set_xticklabels([str(np.round((i * 4) / 60, 1)) for i in ticks_location])
                #     # set x ticks label from ax min to ax max
                #     ax_twin.set_xlabel("Time (min)", fontsize = )
                #     ax_twin.set_xlim(ax.get_xlim())
        else:
            pd_maps = pd.DataFrame()
            for part in participants:
                for c in condition:
                    maps_euclid = min_map(df_cond.loc[df_cond["condition"] == c], part, key="euclid_cog_error")
                    maps_kl = min_map(df_cond.loc[df_cond["condition"] == c], part, key="kl_divergence")
                    maps_corr = min_map(df_cond.loc[df_cond["condition"] == c], part, key="correlation_coefficient")
                    pd_tmp = pd.DataFrame(
                        {
                            "participant": [part] * 3,
                            "condition": [c] * 3,
                            "map_number_euclid": maps_euclid,
                            "map_number_kl": maps_kl,
                            "correlation_coefficient": maps_corr,
                            "muscle": ["fdi", "ext_comm", "sup"],
                        }
                    )
                    if pd_maps.empty:
                        pd_maps = pd_tmp
                    else:
                        pd_maps = pd.concat([pd_maps, pd_tmp], ignore_index=True)
            # pd_maps["min_map_number"] = pd_maps[["map_number_euclid", "correlation_coefficient"]].max(axis=1)
            pd_maps["min_map_number"] = pd_maps[["map_number_kl", "correlation_coefficient"]].max(axis=1)
            pd_maps.to_csv(csv_path)

            axes = subfigs[i].subplots(nrows=1, ncols=1)
            # max_value_axis = pd_maps.min_map_number.max() + 50
            ax = axes
            ax.set_title("b) Optimal number of stimulation needed", fontsize=bigger)
            colors_violin = [
                colors[0] + [0.8],
                colors[0] + [0.6],
                colors[0] + [0.2],
                colors[1] + [1],
                colors[1] + [0.6],
                colors[1] + [0.2],
            ]
            if 'sci' in save_path:
                sns.barplot(
                y="min_map_number",
                x="condition",
                hue="muscle",
                data=pd_maps.loc[pd_maps.muscle != 'fdi'],
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
                pos_x = [-0.2, 0.2, 0.8, 1.2]
                # pos_y = [175, 175, 130, 130]
                offset = 20
                pseudo_pos = pd_maps.loc[pd_maps['condition'] == "pseudo"].min_map_number.max() + offset
                grid_pos = pd_maps.loc[pd_maps['condition'] == "grid"].min_map_number.max() + offset
                pos_y = [grid_pos, grid_pos, pseudo_pos, pseudo_pos]
                text = ["EDC", "SUP", "EDC", "SUP"]
            else:
                sns.violinplot(y="min_map_number", x="condition", hue="muscle", data=pd_maps, ax=ax, legend=False, cut=0)
                for p, patch in enumerate(ax.collections):
                    patch.set_facecolor(colors_violin[p])
                pos_x = [-0.27, 0, 0.27, 0.73, 1, 1.27]
                offset = 20
                pseudo_pos = pd_maps.loc[pd_maps['condition'] == "pseudo"].min_map_number.max() + offset
                grid_pos = pd_maps.loc[pd_maps['condition'] == "grid"].min_map_number.max() + offset
                pos_y = [grid_pos, grid_pos, grid_pos, pseudo_pos, pseudo_pos, pseudo_pos]
                text = ["FDI", "EDC", "SUP", "FDI", "EDC", "SUP"]


            ax.set_ylim(ax.get_ylim()[0], max(pos_y) + 20)
            ax.set_ylabel("Optimal stimulation number", fontsize=big)
            yticks = ["Grid", "Pseudo-random"]
            ax.set_xticklabels([yticks[i] for i in ax.get_xticks()], fontsize=big)
            ax.tick_params(axis='y', labelrotation=0, labelsize=small)

            
            for t, te in enumerate(text):
                ax.text(pos_x[t], pos_y[t], te, ha="center", color=colors_violin[t][:-1] + [1], fontsize=font_base)
            ax.set_xlabel("")
            # ax_ytwin = ax.twinx()
            # # set ticks as 4 * ticks
            # ax_ytwin.set_yticks(np.linspace(pd_maps.min_map_number.min(), pd_maps.min_map_number.max() + 50, 10))
            # ticks_location = ax_ytwin.get_yticks()
            # ax_ytwin.set_yticklabels([str(np.round((i * 4)/60, 1)) for i in ticks_location])
            # # set x ticks label from ax min to ax max
            # ax_ytwin.set_ylabel('Time (min)
    # plt.show()
    plt.savefig(save_path)
    print("Figure saved to", save_path)
    plt.show(block=True)


if __name__ == "__main__":
    seeds = [0]
    smooth_1 = [6]
    smooth_2 =  [6]
    all_folder = []
    for s in seeds:
        for s1 in smooth_1:
            for s2 in smooth_2:
                all_folder.append(rf"D:\Documents\Programmation\tms_motor_map\results\smooth_{s1}_{s2}_{s}_ransac_sci")
    # use pool to perform parallel processing
    args = [
        (
            os.path.join(result_dir, "maps_values.bio"),
            os.path.join(result_dir, "maps_characteristics.csv"),
            os.path.join(result_dir, "maps_results.png"),
            os.path.join(result_dir, "maps_min_map.csv")
        )
        for result_dir in all_folder
    ]
    # ctx = mp.get_context("spawn")
    # with ctx.Pool(processes=4) as pool:
    #     pool.map(main, args)
    main(args[0])
