import pandas as pd
from sklearn.impute import KNNImputer
import copy
from sklearn.cluster import MiniBatchKMeans
from category_encoders.woe import  WOEEncoder
import random


def woe_stand_split_df(df, woe_feats, test_size = 0.3, scaler = None):
    X = df.drop(['labels'], axis=1)
    y = df['labels']
    random_index = random.sample(range(len(y)), len(y))
    trx = int((1-test_size) * len(y))
    X_train = X.iloc[random_index[:trx], : ]
    y_train = y.iloc[random_index[:trx]]
    X_test = X.iloc[random_index[trx:], : ]
    y_test = y.iloc[random_index[trx:]]
    if woe_feats is not None:
        encoder= WOEEncoder(cols = woe_feats)   
        X_train = encoder.fit_transform(X_train, y_train)
        X_test = encoder.transform(X_test)
    assert scaler is not None
    scaler_o = scaler()
    X_train = scaler_o.fit_transform(X_train)
    X_test = scaler_o.transform(X_test)
    return X_train, X_test, y_train, y_test


def sampling_data(X_train, y_train, sampler, seed = None):
    k, sf = sampler
    if k == "KMeansSMOTE":
        sm = sf(random_state=seed, 
                kmeans_estimator=MiniBatchKMeans(n_init="auto"),
                cluster_balance_threshold = min(y_train.mean(), 0.5)) # give lower cluster_balance_threshold
    else: 
        sm = sf(random_state=seed)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res

def load_data_D1(path):
    df = pd.read_csv(path)
    woe_feats = ['Contract', 'PaymentMethod']
    # label
    le_ls = [ "Partner", 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    # label
    for l in le_ls:
        df[l] = df[l].map({'No': 0, 'Yes': 1})
    le_ls = [ "OnlineSecurity", 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    # label
    for l in le_ls:
        df[l] = df[l].map({'No internet service': 0 , 'No': 1, 'Yes': 2})  
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['MultipleLines'] = df['MultipleLines'].map({'No phone service': 0 , 'No': 1, 'Yes': 2})
    df['InternetService'] = df['InternetService'].map({'No': 0 , 'Fiber optic': 1, 'DSL': 2})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"] , errors="coerce")
    # drop nan and duplicates
    df = df.dropna()
    df = df.drop_duplicates(subset=['customerID'], keep='last')
    ranking_key = list(df["customerID"])
    df = df.drop(columns=['customerID']) # delete the id
    df = df.rename(columns={"Churn": "labels"})
    print("dataset size", len(df))
    return df, ranking_key, woe_feats

def load_data_D2(path):
    df = pd.read_csv(path)
    df['ServiceArea'] = df['ServiceArea'].fillna('no info')
    df = df.drop(columns=["RVOwner", 'Homeownership'])
    woe_feats= ['ServiceArea', 'Occupation', 'PrizmCode']
    le_ls = [ "ChildrenInHH", "HandsetRefurbished", "HandsetWebCapable", 
             "TruckOwner", "BuysViaMailOrder", "RespondsToMailOffers",
             "OptOutMailings", "NonUSTravel", "OwnsComputer", "HasCreditCard", 
             "NewCellphoneUser", "NotNewCellphoneUser", "OwnsMotorcycle", "MadeCallToRetentionTeam"]
    # label
    for l in le_ls:
        df[l] = df[l].map({'No': 0, 'Yes': 1})
    df['MaritalStatus'] = df['MaritalStatus'].map({'Unknown':0, 'No':1, 'Yes':2})
    df['CreditRating'] = df['CreditRating'].str.split('-').str[0].astype(int)

    df = df.rename(columns={"Churn": "labels"})
    df["labels"] = df["labels"].map({'Yes': 1, 'No': 0})
    df.loc[df["HandsetPrice"]=="Unknown", "HandsetPrice"] = 0
    df["HandsetPrice"] =  df["HandsetPrice"].astype(int)
    df = df.drop_duplicates(subset=['CustomerID'], keep='last')
    df = df.dropna()
    ranking_key = list(df["CustomerID"])
    df = df.drop(columns=['CustomerID']) # delete the id
    print("dataset size", len(df))
    return df, ranking_key, woe_feats

def load_data_D3(path):
    df = pd.read_csv(path)
    woe_feats = ['MEMBER_MARITAL_STATUS', 'MEMBER_GENDER', 'PAYMENT_MODE' ]
    # label
    df['MEMBER_MARITAL_STATUS'] = df['MEMBER_MARITAL_STATUS'].fillna('unknown')
    df['MEMBER_GENDER'] = df['MEMBER_GENDER'].fillna('unknown')
    df['MEMBERSHIP_PACKAGE'] = df['MEMBERSHIP_PACKAGE'].map({'TYPE-A':0, 'TYPE-B':1})
    df = df.rename(columns={"MEMBERSHIP_STATUS": "labels"})
    df["labels"] = df["labels"].map({'INFORCE': 0, 'CANCELLED': 1})
    # fit nan
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    df[["MEMBER_ANNUAL_INCOME", "MEMBER_OCCUPATION_CD"]] = imputer.fit_transform(df[["MEMBER_ANNUAL_INCOME", "MEMBER_OCCUPATION_CD"]])
    # drop nan and duplicates
    df = df.drop_duplicates(subset=['MEMBERSHIP_NUMBER'], keep='last')
    ranking_key = list(df['START_DATE (YYYYMMDD)'])
    df = df.drop(columns=['MEMBERSHIP_NUMBER', "AGENT_CODE", 'START_DATE (YYYYMMDD)', 'END_DATE  (YYYYMMDD)']) # delete the id
    print("dataset size", len(df))
    return df, ranking_key, woe_feats


def load_data_T1_temporal(path, scaler):
    df = pd.read_csv(path)
    woe_feats = ['MEMBER_MARITAL_STATUS', 'MEMBER_GENDER', 'PAYMENT_MODE' ]
    # label
    df['MEMBER_MARITAL_STATUS'] = df['MEMBER_MARITAL_STATUS'].fillna('unknown')
    df['MEMBER_GENDER'] = df['MEMBER_GENDER'].fillna('unknown')
    df['MEMBERSHIP_PACKAGE'] = df['MEMBERSHIP_PACKAGE'].map({'TYPE-A':0, 'TYPE-B':1})
    df = df.rename(columns={"MEMBERSHIP_STATUS": "labels"})
    df["labels"] = df["labels"].map({'INFORCE': 0, 'CANCELLED': 1})
    # fit nan
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    df[["MEMBER_ANNUAL_INCOME", "MEMBER_OCCUPATION_CD"]] = imputer.fit_transform(df[["MEMBER_ANNUAL_INCOME", "MEMBER_OCCUPATION_CD"]])
    # drop nan and duplicates
    df = df.drop_duplicates(subset=['MEMBERSHIP_NUMBER'], keep='last')
    df_temporal_key = copy.deepcopy(df[["START_DATE (YYYYMMDD)", "END_DATE  (YYYYMMDD)"]])
    df = df.drop(columns=['MEMBERSHIP_NUMBER', "AGENT_CODE", 'START_DATE (YYYYMMDD)', 'END_DATE  (YYYYMMDD)']) # delete the id

    # split remainer and churner
    tr = (df_temporal_key["START_DATE (YYYYMMDD)"]<=20091131) & (df_temporal_key["START_DATE (YYYYMMDD)"]>=20061201)  & ((df_temporal_key["END_DATE  (YYYYMMDD)"].isna()) | ((df_temporal_key["END_DATE  (YYYYMMDD)"]<20101131) & (df_temporal_key["END_DATE  (YYYYMMDD)"]>=20091131)))
    te = (df_temporal_key["START_DATE (YYYYMMDD)"]>20091130) & (df_temporal_key["START_DATE (YYYYMMDD)"]<=20121130)  & ((df_temporal_key["END_DATE  (YYYYMMDD)"].isna()) | ((df_temporal_key["END_DATE  (YYYYMMDD)"]<20131130) & (df_temporal_key["END_DATE  (YYYYMMDD)"]>=20121130)) )
    y = df['labels']
    X_train = df[tr].drop(['labels'], axis=1)
    X_test = df[te].drop(['labels'], axis=1)
    y_train = y[tr]
    y_test = y[te]

    # woe
    encoder= WOEEncoder(cols = woe_feats)   
    X_train = encoder.fit_transform(X_train, y_train)
    X_test = encoder.transform(X_test)

    # scale
    assert scaler is not None
    scaler_o = scaler()
    X_train = scaler_o.fit_transform(X_train)
    X_test = scaler_o.transform(X_test) 
    print("dataset size", len(df))
    
    return X_train, X_test, y_train, y_test