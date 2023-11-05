import pandas as pd
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import random
from sklearn.impute import KNNImputer
from sqlite3 import connect
import copy
from sklearn.cluster import MiniBatchKMeans

def stand_split_df(df, test_size = 0.3, scaler = None):
    X = df.drop(['labels'], axis=1)
    y = df['labels']
    assert scaler is not None
    scaler_o = scaler()
    scaler_o.fit(X)
    X = scaler_o.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
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
    le_ls = ["gender", "Partner", "Dependents",
              "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity","OnlineBackup",
             'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract',
             'PaperlessBilling', 'PaymentMethod', "Churn"]
    # label
    for l in le_ls:
        le = LabelEncoder()
        df[l] = le.fit_transform(df[l])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"] , errors="coerce")
    # drop nan and duplicates
    df = df.dropna()
    df = df.drop_duplicates(subset=['customerID'], keep='last')
    df = df.drop(columns=['customerID']) # delete the id
    df = df.rename(columns={"Churn": "labels"})
    print("dataset size", len(df))
    return df

def load_data_D2(path):
    df = pd.read_csv(path)
    le_ls = ["ServiceArea", "ChildrenInHH", "HandsetRefurbished", "HandsetWebCapable", 
             "TruckOwner", "RVOwner", "Homeownership", "BuysViaMailOrder", "RespondsToMailOffers",
             "OptOutMailings", "NonUSTravel", "OwnsComputer", "HasCreditCard", 
             "NewCellphoneUser", "NotNewCellphoneUser", "OwnsMotorcycle", "MadeCallToRetentionTeam",
             "CreditRating", "PrizmCode", "Occupation", "MaritalStatus"]
    # label
    for l in le_ls:
        le = LabelEncoder()
        df[l] = le.fit_transform(df[l])

    df = df.rename(columns={"Churn": "labels"})
    df["labels"] = df["labels"].map({'Yes': 1, 'No': 0})
    df.loc[df["HandsetPrice"]=="Unknown", "HandsetPrice"] = 0
    df["HandsetPrice"] =  df["HandsetPrice"].astype(int)
    df = df.drop_duplicates(subset=['CustomerID'], keep='last')
    df = df.dropna()
    df = df.drop(columns=['CustomerID']) # delete the id
    print("dataset size", len(df))
    return df

def load_data_D3(path):
    df = pd.read_csv(path)
    le_ls = [ 'MEMBER_MARITAL_STATUS', 'MEMBER_GENDER', 'MEMBERSHIP_PACKAGE', 'PAYMENT_MODE' ]
    # label
    for l in le_ls:
        le = LabelEncoder()
        df[l] = le.fit_transform(df[l])
    df = df.rename(columns={"MEMBERSHIP_STATUS": "labels"})
    df["labels"] = df["labels"].map({'INFORCE': 0, 'CANCELLED': 1})
    # fit nan
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    df[["MEMBER_ANNUAL_INCOME", "MEMBER_OCCUPATION_CD"]] = imputer.fit_transform(df[["MEMBER_ANNUAL_INCOME", "MEMBER_OCCUPATION_CD"]])
    # drop nan and duplicates
    df = df.drop_duplicates(subset=['MEMBERSHIP_NUMBER'], keep='last')
    df = df.drop(columns=['MEMBERSHIP_NUMBER', "AGENT_CODE", 'START_DATE (YYYYMMDD)', 'END_DATE  (YYYYMMDD)']) # delete the id
    print("dataset size", len(df))
    return df


def load_data_D4_temporal(path, scaler):
    df = pd.read_csv(path)
    le_ls = [ 'MEMBER_MARITAL_STATUS', 'MEMBER_GENDER', 'MEMBERSHIP_PACKAGE', 'PAYMENT_MODE' ]
    # label
    for l in le_ls:
        le = LabelEncoder()
        df[l] = le.fit_transform(df[l])
    df = df.rename(columns={"MEMBERSHIP_STATUS": "labels"})
    df["labels"] = df["labels"].map({'INFORCE': 0, 'CANCELLED': 1})
    # fit nan
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    df[["MEMBER_ANNUAL_INCOME", "MEMBER_OCCUPATION_CD"]] = imputer.fit_transform(df[["MEMBER_ANNUAL_INCOME", "MEMBER_OCCUPATION_CD"]])
    # drop nan and duplicates
    df = df.drop_duplicates(subset=['MEMBERSHIP_NUMBER'], keep='last')
    df_temporal_key = copy.deepcopy(df[["START_DATE (YYYYMMDD)", "END_DATE  (YYYYMMDD)"]])
    df = df.drop(columns=['MEMBERSHIP_NUMBER', "AGENT_CODE", 'START_DATE (YYYYMMDD)', 'END_DATE  (YYYYMMDD)']) # delete the id
    
    # scale
    X = df.drop(['labels'], axis=1)
    y = df['labels']
    assert scaler is not None
    scaler_o = scaler()
    scaler_o.fit(X)
    X = scaler_o.transform(X) 

    # split 3 years remainer and 1 year churner
    tr = (df_temporal_key["START_DATE (YYYYMMDD)"]<=20091131) & (df_temporal_key["START_DATE (YYYYMMDD)"]>=20061201)  & ((df_temporal_key["END_DATE  (YYYYMMDD)"].isna()) | (df_temporal_key["END_DATE  (YYYYMMDD)"]<20101131) )
    te = (df_temporal_key["START_DATE (YYYYMMDD)"]>20091130) & (df_temporal_key["START_DATE (YYYYMMDD)"]<=20121130)  & ((df_temporal_key["END_DATE  (YYYYMMDD)"].isna()) | (df_temporal_key["END_DATE  (YYYYMMDD)"]<20131130) )
    X_train = X[tr]
    X_test = X[te]
    y_train = y[tr]
    y_test = y[te]
    print("dataset size", len(df))
    return X_train, X_test, y_train, y_test