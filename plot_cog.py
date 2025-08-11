import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot(df, x, y, hue):
    fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=3, num=y)
    muscle_name = ['fdi', 'ext_comm', 'sup']
    for j in range(3):
        ax[j].set_title(f"{muscle_name[j]}")
        sns.lineplot(
            df.loc[df[y] != 0].loc[df["muscle"] == muscle_name[j]],
            x=x,
            y=y,
            hue=hue,
            marker="o",
            ax=ax[j], 
        )
        if j != 2:
            ax[j].get_legend().remove()
        if 'correlation' in y:
            ax[j].axhline(y=0.9, color='r', linestyle='--')
        if 'euclid' in y:
            ax[j].axhline(y=3.6, color='g', linestyle='--')
        ax[j].set_xticks(list(range(1, 5)))
        ax[j].set_xticklabels([str(i) for i in range(2, 6)])


data_frame = pd.read_csv("maps_characteristics.csv")

paticipants = list(range(2, 11))
participants = [f"P{p:03d}_TN" for p in paticipants]

grid_data_frame = data_frame.loc[data_frame["condition"] == "grid"].loc[data_frame["participant"].isin(participants)]
df_cog_x = grid_data_frame[["map_number", "x_cog_error", "muscle", 'participant']]
df_cog_y = grid_data_frame[["map_number", "y_cog_error", "muscle", 'participant']]
df_cog_euclid = grid_data_frame[["map_number", "euclid_cog_error", "muscle", 'participant']]
df_cor_coef = grid_data_frame[["map_number", "correlation_coefficient", "muscle", 'participant']]

plot(df_cog_x, "map_number", "x_cog_error", "participant")
plot(df_cog_y, "map_number", "y_cog_error", "participant")
plot(df_cog_euclid, "map_number", "euclid_cog_error", "participant")
plot(df_cor_coef,"map_number", "correlation_coefficient", "participant")


plt.show()
