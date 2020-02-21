from sklearn import preprocessing

def encode_df(df, cat_cols):
    
    if len(cat_cols) == 0:
        return df
    else: 
        for c in cat_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[c].values)
            df.loc[:, c] = lbl.transform(df[c].values)
            #self.label_encoders[c] = lbl
        return df
    

        