import matplotlib.pyplot as plt

def df_eda(df, var):
    s = df[var].value_counts()
    print(s/s.sum())
    df = pd.DataFrame(df[var].value_counts()).reset_index()
    return df.rename(columns={'index': var, var: 'Count'})


def cat_eda(df, var_cat):
    fig, axs = plt.subplots(8, 2, squeeze=True)
    axs = axs.flatten()

    for i, j in zip(var_cat, axs):

        (df[i].value_counts()*100.0 / len(df)).plot.pie(autopct='%.1f%%',
                                                        figsize=(14, 36), fontsize=12, ax=j)
        j.yaxis.label.set_size(15)


def num_eda(df, var_num):
    fig, axs = plt.subplots(1, 3, squeeze=True)
    axs = axs.flatten()

    for i, j in zip(var_num, axs):

        df[i].plot.hist(color='#0504aa', alpha=0.6,
                        rwidth=0.96, figsize=(20, 6), ax=j)
        j.set_title(i)
        # j.yaxis.label.set_size(15)


def cat_targ(df, var_cat):
    fig, axs = plt.subplots(8, 2, squeeze=True)
    axs = axs.flatten()

    for i, j in zip(var_cat, axs):

        df.groupby([i, 'Churn']).size().unstack().plot(
            kind='bar', alpha=0.6, figsize=(14, 36), ax=j)
        j.set_title(i)
        j.set_xlabel("")
