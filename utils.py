import numpy as np

def conf_mat_report(conf_mat, label_names):
    report = {}
    for i in range(conf_mat.shape[0]):
        report[label_names[i]] = {
            'precision' : float(conf_mat[i][i]/conf_mat[:, i].sum()*100),
            'recall': float(conf_mat[i][i]/conf_mat[i].sum()*100)
        }
    return report


def split_data(df, ratio=0.8):
    indices = np.array(df.index)
    np.random.shuffle(indices)
    return df.iloc[indices[:int(len(df)*ratio)]], df.iloc[indices[int(len(df)*ratio):]]
