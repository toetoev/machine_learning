import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def preprocessing(data, y, perform_ohe=False, drop_first=False, perform_scale=False, scaler=StandardScaler(), output_Path=None, index=False):
    # split columns as categorical and numerical, don't perform scale to numerical y
    df = data.copy()
    is_numerical = y.dtypes == 'float64' or y.dtypes == 'int64'
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    if is_numerical:
        numerical_cols.remove(y.name)
    if len(numerical_cols) > 0:
        # perform standardization(zero-mean, unit variance) to numerical columns
        if perform_scale:
            scaler = scaler
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # perform label encoding to categorical columns
    if len(categorical_cols) > 0:
        le = LabelEncoder()
        df[categorical_cols] = df[categorical_cols].apply(
            lambda col: le.fit_transform(col))
        # perform one hot key encoding to categorical columns and drop original columns
        if perform_ohe:
            if not is_numerical:
                categorical_cols.remove(y.name)
            df = df.join(pd.get_dummies(
                df[categorical_cols], columns=categorical_cols, prefix=categorical_cols, drop_first=drop_first))
            df = df.drop(categorical_cols, 1)

    # export as processed csv file
    if output_Path != None:
        df.to_csv(output_Path, index=index)
    return df


def pca(x, y, n_components, show_plot=False, show_result=False):
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(x)

    if show_plot:
        colors = 'rgbkcmy'

        unique_y = np.unique(y)
        for i in range(len(unique_y)):
            plt.scatter(pc[y == unique_y[i], 0], pc[y == unique_y[i], 1],
                        color=colors[i % len(colors)],
                        label=unique_y[i])

        plt.legend()
        plt.title('After PCA Transformation')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    if show_result:
        print(pca.explained_variance_ratio_, len(pca.explained_variance_))
        print(pca.explained_variance_ratio_.sum())

    columns = []
    for i in range(len(pca.components_)):
        columns.append('principal component ' + str(i + 1))
    pcDf = pd.DataFrame(data=pc, columns=columns)
    finalDf = pd.concat([pcDf, y], axis=1)
    return finalDf


def feature_selection(df, target, show_heat_map=False, show_process=False):
    corr_mat = df.corr()

    if show_heat_map:
        plt.figure(figsize=(13, 5))
        sns.heatmap(data=corr_mat, annot=True, cmap='GnBu')
        plt.show()

    target_name = target.name
    candidates = corr_mat.index[
        (corr_mat[target_name] > 0.5) | (corr_mat[target_name] < -0.5)
    ].values
    candidates = candidates[candidates != target_name]
    if show_process:
        print('Correlated to', target_name, ': ', candidates)

    removed = []
    for c1 in candidates:
        for c2 in candidates:
            if (c1 not in removed) and (c2 not in removed):
                if c1 != c2:
                    coef = corr_mat.loc[c1, c2]
                    if coef > 0.6 or coef < -0.6:
                        removed.append(c1)
    if show_process:
        print('Removed: ', removed)

    selected_features = [x for x in candidates if x not in removed]
    if show_process:
        print('Selected features: ', selected_features)
    if len(selected_features) == 0:
        return df
    else:
        return pd.concat([df[selected_features], target], axis=1)
