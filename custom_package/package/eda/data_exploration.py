import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def correlation(df, method, figsize, annot = True):

    corr = df.corr(method = method)
    mask  = np.triu(corr)

    plt.figure(figsize = figsize)
    sns.heatmap(corr, annot = True, mask = mask);plt.show()