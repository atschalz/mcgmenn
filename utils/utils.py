import category_encoders as ce
import numpy as np
import pandas as pd
import gpboost as gpb
import random
import os
import tensorflow as tf

from sklearn.model_selection import KFold

def set_seed(seed: int = 42) -> None:
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  # tf.set_random_seed(seed)
  # When running on the CuDNN backend, two further options must be set
  # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  # os.environ['TF_DETERMINISTIC_OPS'] = '1'
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed)
  print(f"Random seed set as {seed}")


def glmm5CV_encode_multiple_features_gpboost(Z_train, Z_val, Z_test, X_train, X_val, X_test, y_train, qs, RS):
    if np.unique(y_train).shape[0]==2:
        likelihood = "binary"
    else:
        likelihood = "gaussian"

    z_glmm_encoded_train = pd.DataFrame(Z_train, index=X_train.index)

    #     encoders = {fold: {q_num: ce.GLMMEncoder() for q_num in range(len(qs))} for fold in range(5)}
    kf_enc = KFold(n_splits=5, shuffle=True, random_state=RS)
    split_enc = kf_enc.split(Z_train, y_train)

    for num, (fit_indices, transform_indices) in enumerate(split_enc):
        Z_fit = Z_train[fit_indices]
        y_fit = y_train.iloc[fit_indices]
        Z_transform = Z_train[transform_indices]
        # y_transform = y_train.iloc[transform_indices]
        for q_num in range(len(qs)):
            print(f"Fit GLMM for fold {num} and feature {q_num}")
            gp_model = gpb.GPModel(group_data=Z_fit[:,[q_num]], likelihood=likelihood)
            gp_model.fit(y=y_fit, X=pd.DataFrame(np.ones([Z_fit.shape[0],1]),columns=["Intercept"]))

            n = Z_fit.shape[0]
            group = np.arange(n)
            m = qs[q_num]
            for i in range(m):
                group[int(i * n / m):int((i + 1) * n / m)] = i
            all_training_data_random_effects = gp_model.predict_training_data_random_effects()
            temp_mapping = dict(pd.concat([pd.DataFrame(Z_fit[:,[q_num]]),all_training_data_random_effects],axis=1).groupby(0).mean()["Group_1"])
            final_mapping = {i: float(gp_model.get_coef().values) if i not in list(temp_mapping.keys()) else temp_mapping[i] for i in range(qs[q_num])}

            z_glmm_encoded_train.iloc[transform_indices, q_num] = z_glmm_encoded_train.iloc[transform_indices, q_num].apply(lambda x: final_mapping[x])
    z_glmm_encoded_train = z_glmm_encoded_train.values

    encoded_val = []
    encoded_test = []
    for q_num in range(len(qs)):
        print(f"Fit GLMM on whole train data for feature {q_num}")
        gp_model = gpb.GPModel(group_data=Z_train[:,[q_num]], likelihood=likelihood)
        gp_model.fit(y=y_train, X=pd.DataFrame(np.ones([Z_train.shape[0],1]),columns=["Intercept"]))

        n = X_train.shape[0]
        group = np.arange(n)
        m = qs[q_num]
        for i in range(m):
            group[int(i * n / m):int((i + 1) * n / m)] = i
        all_training_data_random_effects = gp_model.predict_training_data_random_effects()
        temp_mapping = dict(pd.concat([pd.DataFrame(Z_train[:,[q_num]]),all_training_data_random_effects],axis=1).groupby(0).mean()["Group_1"])
        final_mapping = {i: float(gp_model.get_coef().values) if i not in list(temp_mapping.keys()) else temp_mapping[i] for i in range(qs[q_num])}

        encoded_val.append(pd.Series(Z_val[:,q_num]).apply(lambda x: final_mapping[x]))
        encoded_test.append(pd.Series(Z_test[:,q_num]).apply(lambda x: final_mapping[x]))

        z_glmm_encoded_val = pd.concat(encoded_val, axis=1).values
        z_glmm_encoded_test = pd.concat(encoded_test, axis=1).values

    return z_glmm_encoded_train, z_glmm_encoded_val, z_glmm_encoded_test


def glmm5CV_encode_multiple_features_statsmodels(Z_train, Z_val, Z_test, X_train, X_val, X_test, y_train, qs, RS):
    Z_train_df = pd.DataFrame(Z_train, index=X_train.index)
    Z_val_df = pd.DataFrame(Z_val, index=X_val.index)
    Z_test_df = pd.DataFrame(Z_test, index=X_test.index)

    z_glmm_encoded_train = pd.DataFrame(np.zeros(Z_train_df.shape), index=X_train.index)

    encoders = {fold: {q_num: ce.GLMMEncoder() for q_num in range(len(qs))} for fold in range(5)}
    kf_enc = KFold(n_splits=5, shuffle=True, random_state=RS)
    split_enc = kf_enc.split(Z_train_df, y_train)

    for num, (fit_indices, transform_indices) in enumerate(split_enc):
        Z_fit = Z_train_df.iloc[fit_indices]
        y_fit = y_train.iloc[fit_indices]
        Z_transform = Z_train_df.iloc[transform_indices]
        # y_transform = y_train.iloc[transform_indices]
        for q_num in range(len(qs)):
            print(f"Fit GLMM for fold {num} and feature {q_num}")
            encoders[num][q_num].fit(Z_fit.astype(object)[[q_num]], y_fit)
            #         encoded.append(encoder.transform(pd.DataFrame(Z_train,index=X_train.index).astype(object)[[q_num]]))
            z_glmm_encoded_train.iloc[transform_indices, q_num] = encoders[num][q_num].transform(Z_transform[[q_num]])[
                q_num]
    z_glmm_encoded_train = z_glmm_encoded_train.values

    encoded_val = []
    encoded_test = []
    for q_num in range(len(qs)):
        print(f"Fit GLMM on whole train data for feature {q_num}")
        encoder = ce.GLMMEncoder()
        encoder.fit(Z_train_df.astype(object)[[q_num]], y_train)

        encoded_val.append(encoder.transform(Z_val_df.astype(object)[[q_num]]))
        encoded_test.append(encoder.transform(Z_test_df.astype(object)[[q_num]]))

        z_glmm_encoded_val = pd.concat(encoded_val, axis=1).values
        z_glmm_encoded_test = pd.concat(encoded_test, axis=1).values

    return z_glmm_encoded_train, z_glmm_encoded_val, z_glmm_encoded_test


class TargetEncoderMultiClass():
    def __init__(self, num_classes):
        self.ohe_encoder = ce.OneHotEncoder()
        self.te_encoders = [ce.TargetEncoder()]*(num_classes)
        self.fitted = False


    def fit(self, Z, y):
        y_onehot = self.ohe_encoder.fit_transform(y.astype(object))
        self.class_names = y_onehot.columns  # names of onehot encoded columns

        for num, class_ in enumerate(self.class_names):
            self.te_encoders[num].fit(Z.astype(object), y_onehot[class_])
        self.fitted = True

    def fit_transform(self, Z, y):
        y_onehot = self.ohe_encoder.fit_transform(y.astype(object))
        self.class_names = y_onehot.columns  # names of onehot encoded columns

        Z_te = pd.DataFrame(index=y.index)
        for num, class_ in enumerate(self.class_names):
            self.te_encoders[num].fit(Z.astype(object), y_onehot[class_])
            Z_te_c = self.te_encoders[num].transform(Z.astype(object), y_onehot[class_])
            Z_te_c.columns = [str(x) + '_' + str(class_) for x in Z_te_c.columns]
            Z_te = pd.concat([Z_te, Z_te_c], axis=1)

        self.fitted = True

        return Z_te

    def transform(self, Z, y):
        assert self.fitted == True, "Encoder not fitted!"
        y_onehot = self.ohe_encoder.transform(y.astype(object))
        self.class_names = y_onehot.columns  # names of onehot encoded columns

        Z_te = pd.DataFrame(index=y.index)
        for num, class_ in enumerate(self.class_names):
            Z_te_c = self.te_encoders[num].transform(Z.astype(object), y_onehot[class_])
            Z_te_c.columns = [str(x) + '_' + str(class_) for x in Z_te_c.columns]
            Z_te = pd.concat([Z_te, Z_te_c], axis=1)

        return Z_te

