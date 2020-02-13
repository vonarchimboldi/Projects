from .feature_selection import Feature_Selection
from .categorical import CategoricalFeatures

def filter_df(df, selector):
    feat_selector = Feature_Selection(df, 
                             selector = "vif", 
                             vif_threshold =0.2,
                             mixed =False)

    selected_features = feat_selector.get_features()
    return selected_features

def encode_df(df):
    cols = list(df.columns)
    cat_feats = CategoricalFeatures(df, 
                                    categorical_features=cols, 
                                    encoding_type="label",
                                    handle_na=True)
    df = cat_feats.fit_transform()
    return df

