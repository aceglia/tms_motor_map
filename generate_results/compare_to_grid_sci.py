import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib.gridspec as gridspec

def compute_blandt_altman(
    mean,
    diff,
    units="mm",
    title="Bland-Altman Plot",
    show=True,
    color=None,
    x_axis=None,
    markers=None,
    ax=None,
    threeshold=np.inf,
    no_y_label=False,
    plot=True,
    icc=None
):

    # Average difference (aka the bias)
    bias = np.mean(diff)

    # Sample standard deviation
    s = np.std(diff, ddof=1) 
    # print(f"For the differences, μ = {bias:.4f} {units} and s = {s:.4f} {units} ")

    # Limits of agreement (LOAs)
    upper_loa = bias + 1.96 * s
    lower_loa = bias - 1.96 * s
    # print(f"The limits of agreement are {upper_loa:.2f} {units} and {lower_loa:.2f} {units} ")

    # Confidence level
    C = 0.95  # 95%
    # Significance level, α
    alpha = 1 - C
    # Number of tails
    tails = 2
    # Quantile (the cumulative probability)
    q = 1 - (alpha / tails)
    # Critical z-score, calculated using the percent-point function (aka the
    # quantile function) of the normal distribution
    z_star = st.norm.ppf(q)
    # print(f"95% of normally distributed data lies within {z_star}σ of the mean")
    # Limits of agreement (LOAs)
    loas = (bias - z_star * s, bias + z_star * s)
    # print(f"The limits of agreement are {loas} {units} ")
    # Limits of agreement (LOAs)
    loas = st.norm.interval(C, bias, s)
    print(np.round(loas, 2))
    # Sample size
    n = diff.shape[0]
    # Degrees of freedom
    dof = n - 1
    # Standard error of the bias
    se_bias = s / np.sqrt(n)
    # Standard error of the LOAs
    se_loas = np.sqrt(3 * s**2 / n)

    # Confidence interval for the bias
    ci_bias = st.t.interval(C, dof, bias, se_bias)
    # Confidence interval for the lower LOA
    ci_lower_loa = st.t.interval(C, dof, loas[0], se_loas)
    # Confidence interval for the upper LOA
    ci_upper_loa = st.t.interval(C, dof, loas[1], se_loas)

    # print(
    #     f" Lower LOA = {np.round(lower_loa, 2)}, 95% CI {np.round(ci_lower_loa, 2)}\n",
    #     f"Bias = {np.round(bias, 2)}, 95% CI {np.round(ci_bias, 2)}\n",
    #     f"Upper LOA = {np.round(upper_loa, 2)}, 95% CI {np.round(ci_upper_loa, 2)}",
    # )
    if plot:
        if ax is None:
            plt.figure(title)
        ax = plt.axes() if ax is None else ax
        markers = markers if markers is not None else "o"
        if color is not None:
            for i in range(len(color)):
                # mean_tmp = mean_to_plot[i * len(color[i]) * 4 : (i + 1) * len(color[i]) * 4]
                # diff_tmp = diff_to_plot[i * len(color[i]) * 4 : (i + 1) * len(color[i]) * 4]
                mean_tmp = mean[i * len(color[i]) : (i + 1) * len(color[i])]
                diff_tmp = diff[i * len(color[i]) : (i + 1) * len(color[i])]
                # ax.scatter(mean_tmp, diff_tmp, color=color[i][0], s=100, alpha=0.6, marker=markers)
                color_markers = plt.cm.viridis(np.linspace(0, 1, diff_tmp.shape[0]))
                color_tmp = color_markers if "marker" in title else color[i]
                for j in range(len(mean_tmp)):
                    #if np.abs(diff_tmp[j]) > threeshold:
                    #    continue
                    ax.scatter(mean_tmp[j], diff_tmp[j], c=color_tmp[j], s=100, alpha=0.6, marker=markers)
            

        else:
            ax.scatter(mean, diff, c='k', s=20, alpha=0.6, marker='o')
    # Plot the zero line
        ax.axhline(y=0, color="k", lw=0.5)
        # Plot the bias and the limits of agreement
        ax.axhline(y=loas[1], color="grey", ls="--")
        ax.axhline(y=bias, color="grey", ls="--")
        ax.axhline(y=loas[0], color="grey", ls="--")

        # Labels
        font = 18
        ax.set_title(title, fontsize=font + 2)
        # ax.set_ylabel(f'Difference ({units} )', fontsize=font)
        # if x_axis is not None:
        #     ax.set_xlabel(x_axis, fontsize=font)
        # else:
        #     ax.set_xlabel(f"Mean ({units})", fontsize=font)
        # ax.tick_params(axis="y", labelsize=font)
        # ax.tick_params(axis="x", labelsize=font)
        # if not no_y_label:
        #     ax.set_ylabel(f"Difference ({units})", fontsize=font)
        # else:
        #     ax.set_ylabel("", fontsize=font)
    # ax.xticks(fontsize=font)
    # ax.yticks(fontsize=font)
    # Confidence intervals
    # ax.plot([left] * 2, list(ci_upper_loa), color="grey", ls="--", alpha=0.5)
    # ax.plot([left] * 2, list(ci_bias), color="grey", ls="--", alpha=0.5)
    # ax.plot([left] * 2, list(ci_lower_loa), color="grey", ls="--", alpha=0.5)
    # Confidence intervals' caps
        left, right = ax.get_xlim()

    # Set x-axis limits
        domain = right - left
        ax.set_xlim(left, left + domain)
        x = np.linspace(left, right, 100)
    # ax.plot(x_range, [ci_upper_loa[1]] * 2, color="grey", ls="--", alpha=0.5)
    # ax.plot(x_range, [ci_upper_loa[0]] * 2, color="grey", ls="--", alpha=0.5)
    # ax.plot(x_range, [ci_bias[1]] * 2, color="grey", ls="--", alpha=0.5)
    # ax.plot(x_range, [ci_bias[0]] * 2, color="grey", ls="--", alpha=0.5)
    # ax.plot(x_range, [ci_lower_loa[1]] * 2, color="grey", ls="--", alpha=0.5)
    # ax.plot(x_range, [ci_lower_loa[0]] * 2, color="grey", ls="--", alpha=0.5)
    # fill between confidence intervals for loa
        ax.fill_between(x, ci_lower_loa[0], ci_lower_loa[1], color="grey", alpha=0.2)
        ax.fill_between(x, ci_upper_loa[0], ci_upper_loa[1], color="grey", alpha=0.2)

    # Get axis limits
        bottom, top = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
    # Set y-axis limits
    # max_y = max(abs(bottom), abs(top))
        max_y = top
        min_y = abs(bottom)
        ax.text(x_min, max_y, icc)

        ax.set_ylim(-min_y, max_y)
        # Annotations
        # ax.annotate("+LOA", (right, upper_loa), (0, 7), textcoords="offset pixels", fontsize=font)
        # ax.annotate(f"{upper_loa:+4.2f}", (right, upper_loa), (0, -25), textcoords="offset pixels", fontsize=font)
        # ax.annotate("Bias", (right, bias), (0, 7), textcoords="offset pixels", fontsize=font)
        # ax.annotate(f"{bias:+4.2f}", (right, bias), (0, -25), textcoords="offset pixels", fontsize=font)
        # ax.annotate("-LOA", (right, lower_loa), (0, 7), textcoords="offset pixels", fontsize=font)
        # ax.annotate(f"{lower_loa:+4.2f}", (right, lower_loa), (0, -25), textcoords="offset pixels", fontsize=font)

    if show:
        plt.show()

    return bias, lower_loa, upper_loa, (ci_lower_loa, ci_upper_loa)


import os
results_dir = 'sci_smooth_10_5_2543'
data_frame = pd.read_csv(os.path.join(results_dir, "maps_characteristics.csv"))

data_frame = data_frame.loc[data_frame["participant"] != 'P006_TN']

muscle_list = list(data_frame['muscle'][1:3])

data_frame = data_frame.loc[data_frame['muscle'].isin(muscle_list)]

list_points_tot = [[49, 98, 147, 196],
                    [24, 44, 64, 94, 124, 154, 184],
                    [24, 44, 64, 94, 124, 154, 184]
               ] 
min_maps = pd.read_csv(os.path.join(results_dir, "maps_min_map.csv"))
pd_reduced = pd.DataFrame()
for i, participant in enumerate(min_maps['participant'].unique()):
    for j, muscle in enumerate(min_maps['muscle'].unique()):
        if muscle not in muscle_list:
            continue
        for c, cond in enumerate(['grid', 'pseudo']):
            min_map = min_maps.loc[(min_maps['participant'] == participant) & (min_maps['muscle'] == muscle) & (min_maps['condition'] == cond)]['min_map_number'].values[0]
            pd_tmp = data_frame.loc[(data_frame['participant'] == participant) & (data_frame['muscle'] == muscle) & (data_frame['condition'] == cond) & (data_frame['map_number'] == list_points_tot[c].index(min_map))]
            pd_reduced = pd.concat([pd_reduced, pd_tmp], ignore_index=True)


import seaborn as sns
# fig, axes = plt.subplots(2, 2, figsize=(15, 5))
# fig, axes_min = plt.subplots(2, 2, figsize=(15, 5))
# fig_tot = plt.figure(figsize=(15, 5), num='subplots')

# axes_tot = []
# locator_list = [5,5,350, 0.05]
col = [[(0, 0), (0, 3)], [(0, 0), (0, 3)],
        [(1, 0), (1, 3)], [(1, 0), (1, 3)]]
col_span = [[3, 1], [3, 1], [3, 1], [3, 1]]
# fig, axs = plt.subplot_mosaic([['x_cog_line', 'x_cog_bar', 'y_cog_line', 'y_cog_bar'],
#                                ['area_line', 'area_bar', 'volume_line', 'volume_bar']],
#                               figsize=(6, 6),
#                               width_ratios=(4, 1), height_ratios=(1, 4),
#                               layout='constrained')
fig_tot = plt.figure(layout='constrained', figsize=(10, 4))
subfigs = fig_tot.subfigures(2, 2, wspace=0.05)
comp = ['grid-grid', 'pseudo-pseudo', 'pseudo-grid']
rating_title = ['X-COG', 'Y-COG', 'Area', 'Volume']

for r, rating in enumerate(['x_cog', 'y_cog', 'area', 'volume']):
    pd_icc = pd.DataFrame()
    pd_icc_min = pd.DataFrame()
    for c, cond in enumerate(['grid', 'pseudo', 'pseudo']):
        list_points = list_points_tot[c][:-1] if c != 2 else list_points_tot[c]
        for m, map in enumerate(list_points):
            pseudo = data_frame.loc[data_frame['condition'] == cond].loc[data_frame['map_number'] == list_points.index(map)]
            pseudo['ref'] = 0
            if m == 0:
                min_map_tmp = pd_reduced.loc[pd_reduced['condition'] == cond]
                min_map_tmp['ref'] = 0
            if c == 2:
                ref = data_frame.loc[data_frame['condition'] == 'grid'].loc[data_frame['map_number'] == list_points_tot[0].index(list_points_tot[0][-1])]
            else:
                ref = data_frame.loc[data_frame['condition'] == cond].loc[data_frame['map_number'] == list_points_tot[c].index(list_points_tot[c][-1])] 
            ref['ref'] = 1
            merged = pseudo.reset_index(drop=True).merge(on=['participant', 'muscle'], right=ref.reset_index(drop=True))
            diff = pseudo[rating].values - ref[rating].values
            rmse = np.sqrt(np.mean(diff**2))
            mean = (pseudo[rating].values + ref[rating].values)  / 2
            raters = 'map_number' if c != 2 else 'condition'
            rmse_dic = {'participant': participant, 'muscle': muscle, 
                        'condition': cond, 'rmse': rmse, 'rating': rating, 'map': map, 'comp': comp[c]}
            # rename column 'ICC' into 'ICC_all'
            # icc_df.rename(columns={'ICC': 'ICC_all'}, inplace=True)
            if m == 0:
                diff = min_map_tmp[rating].values - ref[rating].values
                rmse = np.sqrt(np.mean(diff**2))
                mean = (min_map_tmp[rating].values + ref[rating].values)  / 2
                rmse_dic_min = {'participant': participant, 'muscle': muscle, 
                            'condition': cond, 'rmse': rmse, 'rating': rating, 'map': 'min', 'comp': comp[c]}
                pd_icc_min = pd.concat([pd_icc_min, pd.DataFrame(rmse_dic_min, index=[0])], ignore_index=True)
            pd_icc = pd.concat([pd_icc, pd.DataFrame(rmse_dic, index=[0])], ignore_index=True)
    # ax = axes.flatten()[r]
    loc = col[r]
    span = col_span[r]
    # icc3 = pd_icc.loc[pd_icc['Type'] == 'ICC3']
    # icc3_min = pd_icc_min.loc[pd_icc_min['Type'] == 'ICC3']
    icc3 = pd_icc
    icc3_min = pd_icc_min
    subfig = subfigs.flatten()[r]
    subfig.suptitle(rating_title[r], fontsize=16)
    gs = subfig.add_gridspec(1, 4, wspace=0.02)

    # axs[rating + '_line']
    # gs = gridspec.GridSpec(1, 4)
    # gs.update(left=0.05, right=0.48, wspace=0.05) if r % 2 == 0 else gs.update(left=0.55, right=0.98, wspace=0.05)
    axes_tmp = []
    ax_ref = None
    for i in range(2):
        loc_tmp = loc[i]
        span_tmp = span[i]
        # axes_tot.append(plt.subplot2grid((2, 8), loc=loc_tmp, colspan=span_tmp, fig=fig_tot, rowspan=1))
        if i == 0:
            axes_tmp.append(fig_tot.add_subplot(gs[:, loc_tmp[1]:loc_tmp[1]+span_tmp]))
            ax = axes_tmp[i]
            sns.lineplot(x='map', y='rmse', hue='comp', data=icc3, ax=ax, marker='o', palette="rocket",legend=r==0)
            if r==0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, ['Grid-Grid', 'Pseudo-Pseudo', 'Pseudo-Grid'], title='Method comparison', frameon=False)
            ax.set_title('Overall error', fontsize=12)
            ax_ref = ax
        else:
            axes_tmp.append(fig_tot.add_subplot(gs[:, loc_tmp[1]:loc_tmp[1]+span_tmp]))
            ax = axes_tmp[i]
            sns.barplot(x='comp', y='rmse', data=icc3_min, ax=ax, palette="rocket")
            ax.set_title('Optimal map RMSE', fontsize=12)
            # share y from an axis
            ax.set_ylim(ax_ref.get_ylim())
        # ax.hlines(0.8, ax.get_xlim()[0], ax.get_xlim()[1], colors='gray', linestyles='dashed')
        # ax.set_ylim(0, 1 + 0.1)
        if r in [0, 2] and i == 0:
            ax.set_ylabel('RMSE')
        elif r in [0, 2] and i == 1:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        elif r in [1, 3] and i == 0:
            ax.set_ylabel('')

        elif r in [1, 3] and i == 1:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        if r in [2, 3] and i == 0:
            ax.set_xlabel('Stimulation number')
        elif r in [2, 3] and i == 1:
            ax.set_xlabel('Method comparison')
            ax.set_xticklabels(['Grid-\nGrid', 'Pseudo-\nPseudo', 'Pseudo-\nGrid'], rotation=0)
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])

    # ax.set_title(rating)
    # ax_min = axes_min.flatten()[r]
    # ax_min.set_title(rating)
    # # fig, ax = plt.subplots(4, 3, sharey='row')

    # # plt.figure(cond + str(c))
    # # sns.barplot(x='map', y='ICC', hue='map', data=icc3, ax=ax)
    # # markers = ['o', 'v', 'd']
    # # for c, comp in enumerate(icc3['comp'].unique()):
    # #     icc3_tmp = icc3.loc[icc3['comp'] == comp]
    # #     icc3_tmp = icc3_tmp.sort_values(by='map')
    # #     array = np.array(icc3_tmp['ICC'])
    # #     ax.plot(icc3_tmp['map'], array, label=comp, marker=markers[c], color='k', alpha=0.5)

    # sns.lineplot(x='map', y='ICC', hue='comp', data=icc3, ax=ax, marker='o')

    # ax.set_ylim(0, ax.get_ylim()[1] + 0.1)
    # # sns.scatterplot(x='map', y='ICC', hue='comp', data=icc3, ax=ax, legend=False)
    # ax.hlines(0.8, ax.get_xlim()[0], ax.get_xlim()[1], colors='gray', linestyles='dashed')

    # sns.barplot(x='comp', y='ICC', data=icc3_min, ax=ax_min)

plt.show(block=True)
            