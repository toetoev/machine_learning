from supervised_learning import *
from unsupervised_learning import *
from processing import preprocessing, feature_selection, pca
import pandas as pd
import seaborn as sns


# compare different data engineering and feature engineering impact on the outcome
def input_comprison(data, y, model=None, log_matrix=True, out_path=None):
    res = []
    df = preprocessing(data=data, y=data[y])
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))
    res[0].append('Label Encoded')

    df = preprocessing(data=data, y=data[y], perform_ohe=True)
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))
    res[1].append('One Hot Encoded')

    df = preprocessing(
        data=data, y=data[y], perform_scale=True, perform_ohe=True)
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))
    res[2].append('Standard Scaled')

    df = preprocessing(data=data, y=data[y], perform_scale=True)
    df = feature_selection(df=df, target=df[y], show_process=False)
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))
    res[3].append('Feature Selection')

    df = preprocessing(data=data, y=data[y], perform_scale=False)
    df = pca(df.loc[:, df.columns.difference([y])],
             df[y], 0.9, show_result=True)
    res.append(model(
        df.loc[:, df.columns.difference([y])], df[y]))
    res[4].append('PCA')

    df = pd.DataFrame(res, columns=[
                      'Accuracy Score', 'Confusion Matrix', 'Training Time', 'Predict Time', 'Processing'])
    if log_matrix:
        print(model.__name__)
        print(df)
    if out_path != None:
        df.to_csv(out_path)
    return df


# perform classification comparing four model outcome
def classification_model_comparison(data, y, name, log_matrix=True, show_plot=False):
    res = input_comprison(data, y, model=logistic_regression, log_matrix=log_matrix,
                          out_path='reports/supervised_learning/' + name + '/logistic_regression_' + name + '.csv')

    res = res.append(input_comprison(data, y, model=knn, log_matrix=log_matrix,
                                     out_path='reports/supervised_learning/' + name + '/knn_' + name + '.csv'))
    res = res.append(input_comprison(data, y, model=decision_tree, log_matrix=log_matrix,
                                     out_path='reports/supervised_learning/' + name + '/decision_tree_' + name + '.csv'))
    res = res.append(input_comprison(data, y, model=neural_network, log_matrix=log_matrix,
                                     out_path='reports/supervised_learning/' + name + '/neural_network_' + name + '.csv'))
    if show_plot:
        model = ['logistic regression'] * 5 + ['knn'] * 5 + \
            ['decision_tree'] * 5 + ['neural_network'] * 5
        res['model'] = model

        fig, axs = plt.subplots(3)
        fig.suptitle(name)
        plt.subplots_adjust(hspace=0.3)

        sns.barplot(
            y=res['Processing'], x=res['Accuracy Score'], hue=res.model, data=res, orient='h', ax=axs[0])

        sns.barplot(
            y=res['Processing'], x=res['Training Time'], hue=res.model, data=res, orient='h', ax=axs[1])

        sns.barplot(
            y=res['Processing'], x=res['Predict Time'], hue=res.model, data=res, orient='h', ax=axs[2])
        plt.legend(fontsize='small')
        plt.show()


# Regression: Test with wine dataset using linear regression and neural network
#             Test with one hot encoded mushroom dataset using linear regression
df = pd.read_csv('data/wine/winequality-red.csv')
y = 'quality'
df = preprocessing(data=df, y=df[y], perform_scale=True)
print(linear_regression(
    df.loc[:, df.columns.difference(['quality'])], df['quality'], log_result=False))
print(neural_network(df, df[y], is_regression=True,
                     log_result=False, epochs=20))

df = pd.read_csv('data/mushroom/mushrooms.csv')
y = 'class'
df = df.dropna()
df = df.reset_index(drop=True)
df = preprocessing(
    data=df, y=df[y], perform_scale=True, perform_ohe=True, drop_first=True)
print(linear_regression(
    df.loc[:, df.columns.difference(['class'])], df['class']))

# Classification: Test with student-mat dataset using four models and compare with different data engineering and feature engineering
df = pd.read_csv('data/student/student-mat.csv')
df['G3'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
y = 'G3'
classification_model_comparison(
    df, y, 'student', log_matrix=True, show_plot=True)

# Clustering: Test with student-mat dataset
df = pd.read_csv('data/student/student-mat.csv')
df['G3'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
y = 'G3'
hierarchical(df, y, n_clusters=2, scaling=False, features=0)
kmeans(df, y, n_clusters=2, features=0, show_elbow=True)
dbscan(df, y, eps=1, min_samples=10)
