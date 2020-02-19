from fastai.tabular import *

path = untar_data(URLs.ADULT_SAMPLE)

df = pd.read_csv(path/'adult.csv')

