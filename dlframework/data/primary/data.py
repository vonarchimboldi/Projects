from fastai.tabular import *

procs = [FillMissing, Categorify, Normalize]

valid_idx = range(len(df)-2000, len(df))

dep_var = 'salary'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

data = TabularDataBunch.from_df(path, df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)
#print(data.train_ds.cont_names)
