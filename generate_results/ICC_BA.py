import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib.ticker as mticker


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
results_dir = 'smooth_10_5_10'
data_frame = pd.read_csv(os.path.join(results_dir, "maps_characteristics.csv"))

data_frame = data_frame.loc[data_frame["participant"] != 'P006_TN']

muscle_list = list(data_frame['muscle'][:3])

data_frame = data_frame.loc[data_frame['muscle'].isin(muscle_list)]

fig, ax = plt.subplots(4, 3, sharey='row')
list_points = [24, 44, 64, 94, 124]
locator_list = [5,5,350, 0.05]
for m, map in enumerate([24, 64,  124]):
    ref = data_frame.loc[data_frame['condition'] == 'grid'].loc[data_frame['map_number'] == 4]
    pseudo = data_frame.loc[data_frame['condition'] == 'pseudo'].loc[data_frame['map_number'] == list_points.index(map)]
    for r, rating in enumerate(['x_cog', 'y_cog', 'area', 'volume']):
        merged = pseudo.reset_index(drop=True).merge(on=['participant', 'muscle'], right=ref.reset_index(drop=True))
        diff = pseudo[rating].values - ref[rating].values
        mean = (pseudo[rating].values + ref[rating].values)  / 2
        icc_df = pg.intraclass_corr(data=pd.concat([ref, pseudo], ignore_index=True), targets='participant', raters='condition', ratings=rating).round(2)
        ba_result=compute_blandt_altman(mean, diff, show=False, title=rating + '_' + str(map), ax=ax[r, m],
                               icc='ICC: ' + str(icc_df.loc[icc_df.Type == 'ICC3'].ICC.values[0]))
        loc = mticker.MultipleLocator(base=locator_list[r]) 
        ax[r, m].yaxis.set_major_locator(loc)
        if rating == 'volume':
            ax[r, m].set_xlabel('mean')
        print(rating, map, ba_result)
plt.show()
        