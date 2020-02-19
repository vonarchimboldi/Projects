from fastai.tabular import *


learn = tabular_learner(data, layers=[200,100], emb_szs={'native-country': 10}, metrics=accuracy)
learn.fit_one_cycle(1, 1e-2)

print(learn.predict(df.iloc[0]))