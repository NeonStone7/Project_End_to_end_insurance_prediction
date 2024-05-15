from package.data_retrieval.data_retrieval import s3_retrieval, write_to_s3
from package.eda.data_exploration import correlation
from sklearn.datasets import load_iris
import pandas as pd

data = pd.DataFrame(load_iris().data, columns = load_iris().feature_names)


correlation(data,
    method = 'pearson',
    figsize = (20,20),
    annot = True)