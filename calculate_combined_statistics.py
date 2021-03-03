import numpy as np
def combined_mean_std(means: np.array, stds: np.array, counts:np.array)->tuple:
    """[calculates combined statistics]
    Args:
        means (np.array): [mean array of all distributions]
        stds (np.array): [standart deviations array of all distributions]
        counts (np.array): [list of num of elements for all distributions ]
    Returns:
        [tuple]: [combined_means, combined_stds]
    """
    N = np.sum(counts)
    # combined Mean
    Mc = np.sum([m*counts[i] for i, m in enumerate(means)]) / np.sum(counts)
    SumX = np.sum(means*counts)
    SumX2 = np.sum(counts*means**2 + (counts)*stds**2)
    SDc = np.sqrt(np.abs((SumX2 - SumX**2/N))/(N))
    return Mc, SDc

