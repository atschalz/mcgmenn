import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
import category_encoders as ce
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from category_encoders.glmm import GLMMEncoder
from sklearn.model_selection import KFold
from utils.utils import glmm5CV_encode_multiple_features_gpboost, TargetEncoderMultiClass

import pickle
import time

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def process_dataset(dataset_name, target="", mode="train_val_test", RS=42, hct=10, test_ratio=0.2, val_ratio=0.1, folds=5):
    if not os.path.exists(f"../data/prepared/{dataset_name}"):
        os.mkdir(f"../data/prepared/{dataset_name}")

    new_path = f"{mode}_RS{RS}_hct{hct}"
    if mode == "cv":
        new_path += f"_{folds}folds"
    elif mode == "train_test":
        new_path += f"_split{1-test_ratio*100}-{test_ratio*100}"
    elif mode == "train_val_test":
        new_path += f"_split{round(100-(test_ratio+val_ratio)*100)}-{round(test_ratio*100)}-{round(val_ratio*100)}"

    if not os.path.exists(f"../data/prepared/{dataset_name}/"+new_path):
        os.mkdir(f"../data/prepared/{dataset_name}/"+new_path)

    return_dict = {}

    # Load datasets
    try:
        df = pd.read_csv(f"../data/raw/{dataset_name}/{dataset_name}.csv")
        df = df.drop("Unnamed: 0", axis=1)
    except:
        if dataset_name=="wine-reviews":
            df = pd.read_csv(f"../data/raw/{dataset_name}/winemag-data-130k-v2.csv")
            df = df.drop("Unnamed: 0", axis=1)
        else:
            df = pd.read_excel(f"../data/raw/{dataset_name}/{dataset_name}.xlsx")

    # Drop columns with more than 5% missings (difference to Pargent et al.)
    df = df.drop(df.columns[df.isna().sum() / df.shape[0] > 0.95], axis=1)

    # Define dataset-specific column types
    if dataset_name == "churn":
        # Click dataset
        cat_cols = ["area_code"]
        y_col = "class"
        z_cols = ["number_customer_service_calls", "state"]
        bin_cols = ["voice_mail_plan", "international_plan"]
        numeric_cols = list(set(df.columns) - set([y_col]) - set(bin_cols + cat_cols + z_cols))

    elif dataset_name == "kdd_internet_usage":
        # Click dataset
        y_col = "Who_Pays_for_Access_Work"
        z_cols = list(df.nunique()[df.nunique() >= hct].index)
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set(['Who_Pays_for_Access_Work']))
        cat_cols = list(set(df.nunique()[df.nunique() < hct].index) - set([y_col]) - set(bin_cols))
        numeric_cols = []


    elif dataset_name == "Click_prediction_small":
        # Click dataset
        df = df.drop(["query_id", "url_hash"], axis=1)
        cat_cols = []
        y_col = "click"
        z_cols = ["ad_id", "advertiser_id", "keyword_id", "title_id", "description_id", "user_id"]
        bin_cols = []
        numeric_cols = ["impression", "depth", "position"]

    elif dataset_name == "Amazon_employee_access":
        # Amazon dataset
        cat_cols = []
        y_col = "target"
        z_cols = ['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME',
                  'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE']
        bin_cols = []
        numeric_cols = []

    elif dataset_name == "adult":
        # Click dataset
        y_col = "class"
        z_cols = ["occupation", "education", "native-country"]
        bin_cols = ["sex"]
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set(["sex", "class"] + z_cols))
        numeric_cols = list(df.columns[df.dtypes != "object"])
        # Label encode target
        le_ = LabelEncoder()
        df[y_col] = le_.fit_transform(df[y_col].astype(str))

    elif dataset_name == "KDDCup09_upselling":
        # 1. Identify target
        y_col = "UPSELLING"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))

        df.loc[df[y_col]==-1,y_col] = 0

    elif dataset_name == "kick":
        # 1. Identify target
        y_col = "IsBadBuy"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))
        # 6. Label encode dtypes==object

    elif dataset_name == "open_payments":
        # 1. Identify target
        y_col = "status"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))
        # 6. Label encode target
        le_ = LabelEncoder()
        df[y_col] = le_.fit_transform(df[y_col].astype(str))
    elif dataset_name == "road-safety-drivers-sex":
        # 1. Identify target
        y_col = "Sex_of_Driver"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))
        # 6. Label encode target
        le_ = LabelEncoder()
        df[y_col] = le_.fit_transform(df[y_col].astype(str))
    elif dataset_name == "porto-seguro":
        df = df.drop("id", axis=1)
        # 1. Identify target
        y_col = "target"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        cat_cols = [col for col in df.columns if
                    np.logical_and(np.logical_and(df[col].nunique() > 2, df[col].nunique() < hct), "_cat" in col)]
        z_cols = [col for col in df.columns if np.logical_and(df[col].nunique() >= hct, "_cat" in col)]
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols + cat_cols + z_cols))
        # 6. Label encode dtypes==object
    elif dataset_name=="academic_performance":
        df = df.drop(["COD_S11", "Cod_SPro"], axis=1)
        alternative_targets = ["CR_PRO", "QR_PRO", "CC_PRO", "WC_PRO", "FEP_PRO", "ENG_PRO", "QUARTILE", "PERCENTILE",
                               "2ND_DECILE", ]
        df = df.drop(alternative_targets, axis=1)

        z_cols = list(df.columns[list(np.logical_and(df.nunique() >= hct, df.dtypes == "object"))])
        bin_cols = list(df.columns[df.nunique() == 2])
        cat_cols = list(set(df.columns[df.dtypes == "object"]) - set(z_cols + bin_cols))
        numeric_cols = ["SEL", "SEL_IHE", "MAT_S11", "CR_S11", "CC_S11", "BIO_S11", "ENG_S11"]
        y_col = "G_SC"
        df[y_col] = (df[y_col]-df[y_col].mean())/df[y_col].std()
    elif dataset_name=="hpc-job-scheduling":
        # 1. Identify target
        y_col = "Class"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))
        # Label encode target
        le_ = LabelEncoder()
        df[y_col] = le_.fit_transform(df[y_col].astype(str))

    elif dataset_name=="eucalyptus":
        # 1. Identify target
        y_col = "Utility"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))
        # Label encode target
        le_ = LabelEncoder()
        df[y_col] = le_.fit_transform(df[y_col].astype(str))


    elif dataset_name=="Midwest_survey":
        # 1. Identify target
        y_col = "Location..Census.Region."
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = ['In.your.own.words..what.would.you.call.the.part.of.the.country.you.live.in.now.']
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))
        # Label encode target
        le_ = LabelEncoder()
        df[y_col] = le_.fit_transform(df[y_col].astype(str))
    elif dataset_name=="video-game-sales":
        df = df.drop(["Name", "Rank"], axis=1)
        # 1. Identify target
        y_col = "Genre"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = ['Platform', 'Publisher']
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))
        # Label encode target
        le_ = LabelEncoder()
        df[y_col] = le_.fit_transform(df[y_col].astype(str))
    elif dataset_name=="okcupid-stem":
        df = df.drop("last_online", axis=1)
        # 1. Identify target
        y_col = "job"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))
        # Label encode target
        le_ = LabelEncoder()
        df[y_col] = le_.fit_transform(df[y_col].astype(str))
    elif dataset_name=="Diabetes130US":
        df = df.drop(["examide", "citoglipton", "encounter_id", "patient_nbr"], axis=1)
        df.loc[df.readmitted=="NO","readmitted"] = 0
        df.loc[df.readmitted=="<30","readmitted"] = 1
        df.loc[df.readmitted==">30","readmitted"] = 2
        df["readmitted"] = df["readmitted"].astype(int)

        # 1. Identify target
        y_col = "readmitted"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = ["age",
                  "discharge_disposition_id", "admission_source_id",
                  "payer_code",
                  "medical_specialty", "diag_1", "diag_2", "diag_3",
                  ]

        # z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols)- set(z_cols))
    elif dataset_name=="ames-housing":
        # 1. Identify target
        y_col = "Sale_Price"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))
    elif dataset_name=="employee_salaries":
        # 1. Identify target
        y_col = "Current_Annual_Salary"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))
    elif dataset_name=="avocado-sales":
        df = df.drop("Date", axis=1)
        df["year"] = df["year"].astype("object")
        # 1. Identify target
        y_col = "AveragePrice"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))
    elif dataset_name=="wine-reviews":
        df = df.drop(["region_2", "designation", "description", "taster_twitter_handle", "title"], axis=1)
        # 1. Identify target
        y_col = "points"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))

    elif dataset_name=="medical_charges":
        # 1. Identify target
        y_col = "Average.Medicare.Payments"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))
    elif dataset_name=="particulate-matter-ukair-2017":
        df = df.drop(["datetime", "PM.sub.2.5..sub..particulate.matter..Hourly.measured."], axis=1)
        df["Month"] = df["Month"].astype("object")
        df["DayofWeek"] = df["Month"].astype("object")
        # 1. Identify target
        y_col = "PM.sub.10..sub..particulate.matter..Hourly.measured."
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))
    elif dataset_name=="flight-delay-usa-dec-2017":
        df = df.drop(['FL_DATE', 'CRS_DEP_TIME'], axis=1)
        df["DAY_OF_MONTH"] = df["DAY_OF_MONTH"].astype("object")
        df["DAY_OF_WEEK"] = df["DAY_OF_WEEK"].astype("object")
        # 1. Identify target
        y_col = "ARR_DELAY"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))
    elif dataset_name=="nyc-taxi-green-dec-2016":
        df = df.loc[df.payment_type == 1]
        df = df.drop(['total_amount', 'lpep_pickup_datetime', 'lpep_dropoff_datetime', 'trip_distance', 'fare_amount',
                      "payment_type"], axis=1)
        df["RatecodeID"] = df["RatecodeID"].astype("object")
        df["DOLocationID"] = df["DOLocationID"].astype("object")
        df["PULocationID"] = df["PULocationID"].astype("object")
        df["improvement_surcharge"] = df["improvement_surcharge"].astype("object")
        df["mta_tax"] = df["mta_tax"].astype("object")

        # 1. Identify target
        y_col = "tip_amount"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))



    assert len(cat_cols+[y_col]+z_cols+bin_cols+numeric_cols)==df.shape[1], "Column type definitions imply different dimensionality than dataset"

    return_dict["y_col"] = y_col
    return_dict["cat_cols"] = cat_cols
    return_dict["bin_cols"] = bin_cols
    return_dict["z_cols"] = z_cols

    # Split data and target
    y = df[y_col]
    X = df.drop(y_col, axis=1)

    if mode=="cv":
        kf = KFold(n_splits=folds, shuffle=True, random_state=RS)
        split = kf.split(X, y)
    elif mode in ["train_test", "train_val_test"]:
        test_indices = X.sample(frac=test_ratio, random_state=RS).index
        split = [(np.array(list(set(X.index).difference(test_indices))), np.array(test_indices))]

    for num, (train_indices, test_indices) in enumerate(split):
        X_train = X.loc[train_indices]
        y_train = y.loc[train_indices]
        X_test = X.loc[test_indices]
        y_test = y.loc[test_indices]
        if mode in ["train_val_test", "cv"]:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=RS, shuffle=True)

        # label encode categorical features
        bin_impute = {}
        for col in cat_cols + z_cols + bin_cols:# [i for i in bin_cols if df[i].dtype == "object"]:
            le_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-2, encoded_missing_value=-1)
            X_train[col] = le_.fit_transform(X_train[col].astype(str).values.reshape(-1,1))
            if mode in ["train_val_test", "cv"]:
                X_val[col] = le_.transform(X_val[col].astype(str).values.reshape(-1,1))
            X_test[col] = le_.transform(X_test[col].astype(str).values.reshape(-1,1))

            if col in z_cols+cat_cols:
                # Recode categorical column missings as new category
                X_train.loc[X_train[col]==-1,col] = X_train[col].max()+1
                if mode in ["train_val_test", "cv"]:
                    X_val.loc[X_val[col]==-1,col] = X_train[col].max()+1
                X_test.loc[X_test[col]==-1,col] = X_train[col].max()+1

                # Recode categorical column unknown categories as new category
                X_train.loc[X_train[col] == -2, col] = X_train[col].max() + 2
                if mode in ["train_val_test", "cv"]:
                    X_val.loc[X_val[col] == -2, col] = X_train[col].max() + 2
                X_test.loc[X_test[col] == -2, col] = X_train[col].max() + 2
            elif col in bin_cols:
                # Impute binary columns with train mode
                u, c = np.unique(X_train[col][X_train[col]!=-1], return_counts=True)
                bin_impute[col] = u[np.argmax(c)]
                X_train.loc[X_train[col]==-1,col] = bin_impute[col]
                if mode in ["train_val_test", "cv"]:
                    X_val.loc[X_val[col]==-1,col] = bin_impute[col]
                X_test.loc[X_test[col]==-1,col] = bin_impute[col]

            X_train[col] = X_train[col].astype(int)
            X_val[col] = X_val[col].astype(int)
            X_test[col] = X_test[col].astype(int)

        if mode in ["train_test", "train_val_test"]:
            str_num = ""
        elif mode == "cv":
            str_num = f"_{num}"

        # Pargent Pipeline
        ### Imputation 1
        # binary and cat already happened

        # Impute continuous columns with train mean & standardize
        cont_impute = {}
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                cont_impute[col] = X_train[col][~X_train[col].isna()].mean()
                X_train.loc[X_train[col].isna(),col] = cont_impute[col]
                if mode in ["train_val_test", "cv"]:
                    X_val.loc[X_val[col].isna(),col] = cont_impute[col]
                X_test.loc[X_test[col].isna(),col] = cont_impute[col]

            # Standardize
            scaler = StandardScaler()
            # fit and transform scaler on X_train and X_test
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            if mode in ["train_val_test", "cv"]:
                X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        # Standardize continuous targets
        if target =="continuous":
            target_scaler = StandardScaler()
            # fit and transform scaler on X_train and X_test
            y_train = pd.Series(target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel(),index=X_train.index)
            if mode in ["train_val_test", "cv"]:
                y_val = pd.Series(target_scaler.transform(y_val.values.reshape(-1, 1)).ravel(),index=X_val.index)
            y_test = pd.Series(target_scaler.transform(y_test.values.reshape(-1, 1)).ravel(),index=X_test.index)

            return_dict["target_scaler"] = target_scaler

        ### Encoding & Imputation 2
        # get encodings for high-card cat features
        z_ohe_encoded_train = pd.DataFrame(index=X_train.index)
        if mode in ["train_val_test", "cv"]:
            z_ohe_encoded_val = pd.DataFrame(index=X_val.index)
        z_ohe_encoded_test = pd.DataFrame(index=X_test.index)
        z_target_encoded_train = pd.DataFrame(index=X_train.index)
        if mode in ["train_val_test", "cv"]:
            z_target_encoded_val = pd.DataFrame(index=X_val.index)
        z_target_encoded_test = pd.DataFrame(index=X_test.index)
        if len(z_cols) > 0:
            start = time.time()
            for col in z_cols:
                # OHE
                # u, c = np.unique(X_train[col], return_counts=True)
                # collapse_ = u[c < hct]
                # col_collapsed_train = X_train[col].apply(lambda x: -2 if x in collapse_ else x)
                # if mode in ["train_val_test", "cv"]:
                #     col_collapsed_val = X_val[col].apply(lambda x: -2 if x in collapse_ else x)
                # col_collapsed_test = X_test[col].apply(lambda x: -2 if x in collapse_ else x)
                enc = OneHotEncoder(min_frequency=hct, handle_unknown='ignore')
                ohe_encoded_train = pd.DataFrame(enc.fit_transform(X_train[[col]]).toarray(),
                                                 columns=enc.get_feature_names_out([col]), index=X_train.index)
                if mode in ["train_val_test", "cv"]:
                    ohe_encoded_val = pd.DataFrame(enc.transform(X_val[[col]]).toarray(),
                                               columns=enc.get_feature_names_out([col]), index=X_val.index)
                ohe_encoded_test = pd.DataFrame(enc.transform(X_test[[col]]).toarray(),
                                                columns=enc.get_feature_names_out([col]), index=X_test.index)
                z_ohe_encoded_train = z_ohe_encoded_train.join(ohe_encoded_train)
                if mode in ["train_val_test", "cv"]:
                    z_ohe_encoded_val = z_ohe_encoded_val.join(ohe_encoded_val)
                z_ohe_encoded_test = z_ohe_encoded_test.join(ohe_encoded_test)
            end = time.time()
            ohe_encoding_time = end - start

            start = time.time()
            for col in z_cols:

                # Target encoding
                if target=="categorical":
                    encoder = TargetEncoderMultiClass(np.unique(y_train).shape[0])
                else:
                    encoder = TargetEncoder()
                re_encoded_train = encoder.fit_transform(X_train[col].astype(object), y_train)
                if mode in ["train_val_test", "cv"]:
                    re_encoded_val = encoder.transform(X_val[col].astype(object), y_val)
                re_encoded_test = encoder.transform(X_test[col].astype(object), y_test)
                z_target_encoded_train = z_target_encoded_train.join(re_encoded_train)
                if mode in ["train_val_test", "cv"]:
                    z_target_encoded_val = z_target_encoded_val.join(re_encoded_val)
                z_target_encoded_test = z_target_encoded_test.join(re_encoded_test)
            end = time.time()
            target_encoding_time = end - start
            # GLMM encoding (scales poorly by samples (no. of RE does not really matter): 721ms for 100, 9.04s for 500, 59.8s for 1000)
        #     encoder = GLMMEncoder()
        #     re_encoded_train = encoder.fit_transform(X_train[z_cols].astype(object), y_train)
        #     re_encoded_val = encoder.fit_transform(X_val[col].astype(object), y_val)
        #     re_encoded_test = encoder.fit_transform(X_test[col].astype(object), y_test)
        #     z_target_encoded_train = z_target_encoded_train.join(re_encoded_train)
        #     z_target_encoded_val = z_target_encoded_val.join(re_encoded_val)
        #     z_target_encoded_test = z_target_encoded_test.join(re_encoded_test)


        ### Drop constants (Drop features that are constant during training. As none of the original datasets includes constant columns, this step only removes constant features that are produced by the encoders or the CV splitting procedure)
        if any(X_train.nunique() == 1):
            drop_cols = X_train.columns[X_train.nunique() == 1]
            X_train = X_train.drop(drop_cols, axis=1)
            if mode in ["train_val_test", "cv"]:
                X_val = X_val.drop(drop_cols, axis=1)
            X_test = X_test.drop(drop_cols, axis=1)
        if any(z_ohe_encoded_train.nunique() == 1):
            drop_cols = z_ohe_encoded_train.columns[z_ohe_encoded_train.nunique() == 1]
            z_ohe_encoded_train = z_ohe_encoded_train.drop(drop_cols, axis=1)
            if mode in ["train_val_test", "cv"]:
                z_ohe_encoded_val = z_ohe_encoded_val.drop(drop_cols, axis=1)
            z_ohe_encoded_test = z_ohe_encoded_test.drop(drop_cols, axis=1)
        if any(z_target_encoded_train.nunique() == 1):
            drop_cols = z_target_encoded_train.columns[z_target_encoded_train.nunique() == 1]
            z_target_encoded_train = z_target_encoded_train.drop(drop_cols, axis=1)
            if mode in ["train_val_test", "cv"]:
                z_target_encoded_val = z_target_encoded_val.drop(drop_cols, axis=1)
            z_target_encoded_test = z_target_encoded_test.drop(z_target_encoded_test.columns[z_target_encoded_test.nunique() == 1], axis=1)

        return_dict["z_ohe_encoded_train"+str_num] = z_ohe_encoded_train
        if mode in ["train_val_test", "cv"]:
            return_dict["z_ohe_encoded_val"+str_num] = z_ohe_encoded_val
        return_dict["z_ohe_encoded_test"+str_num] = z_ohe_encoded_test
        return_dict["z_target_encoded_train"+str_num] = z_target_encoded_train
        if mode in ["train_val_test", "cv"]:
            return_dict["z_target_encoded_val"+str_num] = z_target_encoded_val
        return_dict["z_target_encoded_test"+str_num] = z_target_encoded_test


        ### Final one-hot-ecoding
        # Encode low-card cat features
        if len(cat_cols) > 0:
            enc = OneHotEncoder(handle_unknown='ignore')

            encoded_train = pd.DataFrame(enc.fit_transform(X_train[cat_cols]).toarray(),
                                         columns=enc.get_feature_names_out(cat_cols), index=X_train.index)
            X_train.drop(columns=cat_cols, inplace=True)
            X_train = X_train.join(encoded_train)

            if mode in ["train_val_test", "cv"]:
                encoded_val = pd.DataFrame(enc.transform(X_val[cat_cols]).toarray(),
                                           columns=enc.get_feature_names_out(cat_cols), index=X_val.index)
                X_val.drop(columns=cat_cols, inplace=True)
                X_val = X_val.join(encoded_val)

            encoded_test = pd.DataFrame(enc.transform(X_test[cat_cols]).toarray(),
                                        columns=enc.get_feature_names_out(cat_cols), index=X_test.index)
            X_test.drop(columns=cat_cols, inplace=True)
            X_test = X_test.join(encoded_test)

        ### Define Z
        Z_train = X_train[z_cols]
        if mode in ["train_val_test", "cv"]:
            Z_val = X_val[z_cols]
        Z_test = X_test[z_cols]


        # GLMM Encoding
        qs = [np.max([Z_train[col].max(),Z_val[col].max(),Z_test[col].max()]) + 1 for col in Z_train.columns]

        # start = time.time()
        # if target == "categorical":
        #     n_classes = np.unique(y_train).shape[0]
        #     z_glmm_encoded_train = np.array([], dtype=np.int64).reshape(X_train.shape[0], 0)
        #     z_glmm_encoded_val = np.array([], dtype=np.int64).reshape(X_val.shape[0], 0)
        #     z_glmm_encoded_test = np.array([], dtype=np.int64).reshape(X_test.shape[0], 0)
        #
        #     np.concatenate([z_glmm_encoded_train, z_glmm_encoded_train], axis=1)
        #     for c in range(n_classes):
        #         z_glmm_encoded_train_class, z_glmm_encoded_val_class, z_glmm_encoded_test_class = glmm5CV_encode_multiple_features_gpboost(
        #             Z_train.values, Z_val.values, Z_test.values, X_train, X_val, X_test, (y_train == c).astype(int), qs, RS)
        #
        #         z_glmm_encoded_train = np.concatenate([z_glmm_encoded_train, z_glmm_encoded_train_class], axis=1)
        #         z_glmm_encoded_val = np.concatenate([z_glmm_encoded_val, z_glmm_encoded_val_class], axis=1)
        #         z_glmm_encoded_test = np.concatenate([z_glmm_encoded_test, z_glmm_encoded_test_class], axis=1)
        #     glmm_enc_cols = list(np.array([[f"{col}_c{c}" for c in range(n_classes)] for col in z_cols]).ravel())
        # else:
        #     z_glmm_encoded_train, z_glmm_encoded_val, z_glmm_encoded_test = glmm5CV_encode_multiple_features_gpboost(Z_train.values,
        #                                                                                         Z_val.values,
        #                                                                                                      Z_test.values,
        #                                                                                                      X_train,
        #                                                                                                      X_val,
        #                                                                                                      X_test,
        #                                                                                                      y_train,
        #                                                                                                      qs, RS)
        #     glmm_enc_cols = z_cols
        # end = time.time()
        # glmm_encoding_time = end - start
        #
        # return_dict["z_glmm_encoded_train"+str_num] = pd.DataFrame(z_glmm_encoded_train, index=X_train.index, columns=glmm_enc_cols)
        # return_dict["z_glmm_encoded_val"+str_num] = pd.DataFrame(z_glmm_encoded_val, index=X_val.index, columns=glmm_enc_cols)
        # return_dict["z_glmm_encoded_test"+str_num] = pd.DataFrame(z_glmm_encoded_test, index=X_test.index, columns=glmm_enc_cols)
        #
        # return_dict["glmm_encoding_time"+str_num] = glmm_encoding_time
        return_dict["target_encoding_time"+str_num] = target_encoding_time
        return_dict["ohe_encoding_time"+str_num] = ohe_encoding_time

        X_train = X_train.drop(z_cols, axis=1)
        if mode in ["train_val_test", "cv"]:
            X_val = X_val.drop(z_cols, axis=1)
        X_test = X_test.drop(z_cols, axis=1)

        # Set datatytes
        return_dict["Z_train"+str_num] = Z_train.values.astype(np.int32)
        return_dict["X_train"+str_num] = X_train.astype(np.float32)
        if target=="categorical":
            return_dict["y_train" + str_num] = y_train.astype(np.int32).values.ravel()
        else:
            return_dict["y_train"+str_num] = y_train.astype(np.float32).values.ravel()

        if mode in ["train_val_test", "cv"]:
            return_dict["Z_val"+str_num] = Z_val.values.astype(np.int32)
            return_dict["X_val"+str_num] = X_val.astype(np.float32)
            if target == "categorical":
                return_dict["y_val" + str_num] = y_val.astype(np.int32).values.ravel()
            else:
                return_dict["y_val"+str_num] = y_val.astype(np.float32).values.ravel()

        return_dict["Z_test"+str_num] = Z_test.values.astype(np.int32)
        return_dict["X_test"+str_num] = X_test.astype(np.float32)
        if target=="categorical":
            return_dict["y_test" + str_num] = y_test.astype(np.int32).values.ravel()
        else:
            return_dict["y_test"+str_num] = y_test.astype(np.float32).values.ravel()


    with open(f"../data/prepared/{dataset_name}/{new_path}/data_dict.pickle", 'wb') as handle:
        pickle.dump(return_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




