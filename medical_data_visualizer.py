import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data
df = pd.read_csv('medical_examination.csv')

# 2. Add an overweight column
# BMI = weight (kg) / height (m)^2. If BMI > 25, overweight = 1, else 0.
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3. Normalize data by making 0 always good and 1 always bad. 
# If the value of cholesterol or gluc is 1, set the value to 0. If more than 1, set to 1.
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Draw Categorical Plot
def draw_cat_plot():
    # 5. Create DataFrame for cat plot using pd.melt with values from cholesterol, gluc, smoke, alco, active, and overweight
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Group and reformat the data in df_cat to split it by cardio. Show the counts of each feature.
    # (Note: renaming column is necessary for catplot to work)
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size().rename(columns={'size': 'total'})
    
    # 7. Convert the data into long format and create a chart that shows the value counts of the categorical features using sns.catplot()
    fig = sns.catplot(
        x='variable', y='total', hue='value', col='cardio',
        data=df_cat, kind='bar'
    ).fig

    # 8. Get the figure for the output and store it in the fig variable
    # (fig is already obtained above)

    # 9. Do not modify the next two lines.
    fig.savefig('catplot.png')
    return fig


# 10. Draw Heat Map
def draw_heat_map():
    # 11. Clean the data in the df_heat variable
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculate the correlation matrix and store it in the corr variable
    corr = df_heat.corr(method='pearson')

    # 13. Generate a mask for the upper triangle and store it in the mask variable
    mask = np.triu(corr)

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Plot the correlation matrix using sns.heatmap()
    sns.heatmap(corr, annot=True, fmt='.1f', linewidths=1, mask=mask, square=True, cbar_kws={'shrink': 0.5}, center=0, vmin=-0.1, vmax=0.25, ax=ax)

    # 16. Do not modify the next two lines.
    fig.savefig('heatmap.png')
    return fig