from KDEpy import FFTKDE
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import accuracy_score as acc
from tensorflow.keras.metrics import CategoricalAccuracy as cat_acc
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import roc_auc_score as auroc
from sklearn.metrics import f1_score as f1
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2


def logit(x):
    if x == 0:
        return np.log((1 / (x+1e-2) - 1))
    elif x==1:
        return np.log((1 / (x-1e-2) - 1))
    else:
        return np.log((1 / (x) - 1))


def get_metrics(y_true,y_pred,target):
    if target == "binary":
        # Inputs: True=vector with binary labels, Pred=vector with class 1 probabilitues
        res = {
            "Accuracy": acc(y_true, np.round(y_pred)),
            "AUROC": auroc(y_true, y_pred),
            "F1": f1(y_true, np.round(y_pred))
        }
    elif target == "continuous":
        # Inputs: True=numerical vector, Pred=numerical vector
        res = {
           "MSE": mse(y_true, y_pred),
           "R2": r2(y_true, y_pred)
              }

    elif target == "categorical":
        # Inputs: True=N*C OHE matrix, Pred=N*C Softmax matrix
        res = {
            "Accuracy": cat_acc()(y_true, y_pred).numpy(),
            "F1": F1Score(num_classes=y_true.shape[1], average="macro")(y_true, y_pred).numpy(),
            "AUROC": auroc(y_true, y_pred, multi_class="ovo", average="macro")
                       }
    return res


def get_sigma_df(model_menn_info, Z_train, y_train, Z_test, y_test, target, sig2bs_true=None, sig2bs_lmmnn=None):
    # Todo: Extend to multiclass
    # Todo: Add sig2e in case of regression
    res = {}
    if target in ["binary", "continuous"]:
        for re_num in range(len(model_menn_info["random_effects"])):
            res[re_num] = {}

            train_encoder = TargetEncoder(handle_missing="value",handle_unknown="value").fit(Z_train[:,re_num].astype(object), y_train)
            test_encoder = TargetEncoder(handle_missing="value",handle_unknown="value").fit(Z_test[:,re_num].astype(object), y_test)
            mean_y_group_train = train_encoder.transform(np.unique(Z_train[:,re_num])).values.ravel()
            mean_y_group_test = test_encoder.transform(np.unique(Z_train[:,re_num])).values.ravel()
            if target == "binary":
                mean_y_group_train = np.array(list(map(logit,mean_y_group_train)))
                mean_y_group_test = np.array(list(map(logit,mean_y_group_test)))

            res[re_num]["MENN"] = round(model_menn_info["_stddev_z"][re_num]**2,4)
            res[re_num]["Group Mean Train"] = round(mean_y_group_train.std()**2,4)
            res[re_num]["Group Mean Test"] = round(mean_y_group_test.std()**2,4)
            if sig2bs_true is not None:
                res[re_num]["True"] = round(sig2bs_true[re_num],4)
            if sig2bs_lmmnn is not None:
                res[re_num]["LMMNN"] = round(sig2bs_lmmnn[re_num]**2,4)

        return pd.DataFrame(res)

    elif target == "categorical":
        num_classes = np.unique(y_train).shape[0]
        y_train = tf.one_hot(y_train,num_classes).numpy()
        y_test = tf.one_hot(y_test,num_classes).numpy()
        for c in range(num_classes):
            res_c = {}
            for re_num in range(len(model_menn_info["random_effects"])):
                res_c[re_num] = {}

                train_encoder = TargetEncoder(handle_missing="value",handle_unknown="value").fit(Z_train[:, re_num].astype(object), y_train[:,c])
                test_encoder = TargetEncoder(handle_missing="value",handle_unknown="value").fit(Z_test[:, re_num].astype(object), y_test[:,c])
                mean_y_group_train = train_encoder.transform(np.unique(Z_train[:, re_num])).values.ravel()
                mean_y_group_test = test_encoder.transform(np.unique(Z_train[:, re_num])).values.ravel()
                mean_y_group_train = np.array(list(map(logit,mean_y_group_train)))
                mean_y_group_test = np.array(list(map(logit,mean_y_group_test)))

                res_c[re_num]["MENN"] = round(model_menn_info["_stddev_z"][re_num][c] ** 2, 4)
                res_c[re_num]["Group Mean Train"] = round(mean_y_group_train.std() ** 2, 4)
                res_c[re_num]["Group Mean Test"] = round(mean_y_group_test.std() ** 2, 4)
                if sig2bs_true is not None:
                    res_c[re_num]["True"] = round(sig2bs_true[re_num][c], 4)
            res[f"classs_{c}"] = pd.DataFrame(res_c)
        return pd.concat(res)


def compute_corr(re, Z_train, y_train, Z_test, y_test, target, b_hats_true=None, b_hats_lmmnn=None):
    # Todo: Add true and lmmnn
    # Todo:Extend to regression & multiclass

    res = {}
    if target in ["continuous", "binary"]:
        for re_num in range(len(re)):
            res[re_num] = {}

            train_encoder = TargetEncoder().fit(Z_train[:, re_num].astype(object), y_train)
            test_encoder = TargetEncoder().fit(Z_test[:, re_num].astype(object), y_test)
            mean_y_group_train = train_encoder.transform(np.unique(Z_train[:,re_num])).values.ravel()
            mean_y_group_test = test_encoder.transform(np.unique(Z_train[:,re_num])).values.ravel()
            if target == "binary":
                mean_y_group_train = np.array(list(map(logit,mean_y_group_train)))
                mean_y_group_test = np.array(list(map(logit,mean_y_group_test)))

            res[re_num]["corr_to_rand_std"] = np.std(
                [np.corrcoef([np.random.randn(re[re_num].shape[0]), re[re_num]])[0, 1] for i in range(1000)])
            res[re_num]["corr_to_te_train_menn"] = np.corrcoef([mean_y_group_train, re[re_num]])[0, 1]
            res[re_num]["corr_to_te_test_menn"] = np.corrcoef([mean_y_group_test, re[re_num]])[0, 1]
            if b_hats_lmmnn is not None:
                res[re_num]["corr_to_te_train_lmmnn"] = np.corrcoef([mean_y_group_train, b_hats_lmmnn[re_num][np.unique(Z_train[:,re_num])]])[0, 1]
                res[re_num]["corr_to_te_test_lmmnn"] = np.corrcoef([mean_y_group_test, b_hats_lmmnn[re_num][np.unique(Z_train[:,re_num])]])[0, 1]
            if b_hats_true is not None:
                res[re_num]["corr_to_te_train_true"] = np.corrcoef([mean_y_group_train, b_hats_true[re_num][np.unique(Z_train[:,re_num])]])[0, 1]
                res[re_num]["corr_to_te_test_true"] = np.corrcoef([mean_y_group_test, b_hats_true[re_num][np.unique(Z_train[:,re_num])]])[0, 1]


        return pd.DataFrame(res)

    if target =="categorical":
        num_classes = np.unique(y_train).shape[0]
        y_train = tf.one_hot(y_train, num_classes).numpy()
        y_test = tf.one_hot(y_test, num_classes).numpy()
        for c in range(num_classes):
            res_c = {}
            for re_num in range(len(re)):
                res_c[re_num] = {}

                train_encoder = TargetEncoder(handle_missing="value",handle_unknown="value").fit(Z_train[:, re_num].astype(object), y_train[:,c])
                test_encoder = TargetEncoder(handle_missing="value",handle_unknown="value").fit(Z_test[:, re_num].astype(object), y_test[:,c])
                mean_y_group_train = train_encoder.transform(np.unique(Z_train[:, re_num])).values.ravel()
                mean_y_group_test = test_encoder.transform(np.unique(Z_train[:, re_num])).values.ravel()
                mean_y_group_train = np.array(list(map(logit,mean_y_group_train)))
                mean_y_group_test = np.array(list(map(logit,mean_y_group_test)))

                res_c[re_num]["corr_to_rand_std"] = np.std(
                    [np.corrcoef([np.random.randn(re[re_num][c].shape[0]), re[re_num][c]])[0, 1] for i in range(1000)])
                res_c[re_num]["corr_to_te_train_menn"] = np.corrcoef([mean_y_group_train, re[re_num][c]])[0, 1]
                res_c[re_num]["corr_to_te_test_menn"] = np.corrcoef([mean_y_group_test, re[re_num][c]])[0, 1]
                if b_hats_true is not None:
                    if np.std(b_hats_true[re_num][:,c]) == 0:
                        print(f"Skip RE {re_num} for class {c} as it has zero variance")
                        continue

                    res_c[re_num]["corr_to_te_train_true"] = np.corrcoef([mean_y_group_train, b_hats_true[re_num][[np.unique(Z_train[:,re_num])],c]])[0, 1]
                    res_c[re_num]["corr_to_te_test_true"] = np.corrcoef([mean_y_group_test, b_hats_true[re_num][[np.unique(Z_train[:,re_num])],c]])[0, 1]

            res[f"classs_{c}"] = pd.DataFrame(res_c)
        return pd.concat(res)



def plot_b_hats(model_menn_info, Z_train, y_train, target, test_data=None, mode="raw", normalize=False, b_hats_true=None,
                b_hats_lmmnn=None):
    '''modes: raw, kde
    If a feature is high cardinality and many samples per cluster are available, KDE is well suited to visualized the distribution.
    Otherwise plotting the raw b_hat is preferred

    '''
    normalize_func = lambda x: (x - x.mean()) / (x.std()+1e-10)

    if test_data is not None:
        Z_test, y_test = test_data

    res = {}


    if target in ["continuous", "binary"]:
        for re_num in range(len(model_menn_info["random_effects"])):
            res[re_num] = {}

            re = model_menn_info["random_effects"][re_num]
            if b_hats_true is not None:
                re_true = b_hats_true[re_num][np.unique(Z_train[:,re_num])]
                if np.std(re_true) == 0:
                    print(f"Skip RE {re_num} for class {c} as it has zero variance")
                    continue
            if b_hats_lmmnn is not None:
                re_lmmnn = b_hats_lmmnn[re_num][np.unique(Z_train[:,re_num])]


            train_encoder = TargetEncoder().fit(Z_train[:, re_num].astype(object), y_train)
            mean_y_group_train = train_encoder.transform(np.unique(Z_train[:,re_num])).values.ravel()
            if test_data is not None:
                test_encoder = TargetEncoder().fit(Z_test[:, re_num].astype(object), y_test)
                mean_y_group_test = test_encoder.transform(np.unique(Z_train[:, re_num])).values.ravel()
            if target == "binary":
                mean_y_group_train = np.array(list(map(logit, mean_y_group_train)))
                mean_y_group_test = np.array(list(map(logit, mean_y_group_test)))

            if normalize:
                re = normalize_func(re)
                mean_y_group_train = normalize_func(mean_y_group_train)
                if test_data is not None:
                    mean_y_group_test = normalize_func(mean_y_group_test)
                if b_hats_lmmnn is not None:
                    re_lmmnn = normalize_func(re_lmmnn)
                if b_hats_true is not None:
                    re_true = normalize_func(re_true)

            legend = ["MENN", "Cluster Mean Train", "Cluster Mean Test"]

            if mode == "kde":
                edges_menn = np.linspace(re.min() - 1e-10, re.max() + 1e-10, np.min([100, len(np.unique(re))]))
                kde_menn = FFTKDE(kernel='gaussian', bw="ISJ").fit(re)
                probabilities_menn = kde_menn.evaluate(edges_menn)
                plt.plot(edges_menn, probabilities_menn)
            else:
                plt.plot(re)

            if mode == "kde":
                edges_groupmean_train = np.linspace(mean_y_group_train.min() - 1e-10, mean_y_group_train.max() + 1e-10,
                                                    np.min([100, len(np.unique(mean_y_group_train))]))
                kde_groupmean_train = FFTKDE(kernel='gaussian', bw="ISJ").fit(mean_y_group_train)
                probabilities_groupmean_train = kde_groupmean_train.evaluate(edges_groupmean_train)
                plt.plot(edges_groupmean_train, probabilities_groupmean_train)
            else:
                plt.plot(mean_y_group_train)
            if test_data is not None:
                if mode == "kde":
                    edges_groupmean_test = np.linspace(mean_y_group_test.min() - 1e-10, mean_y_group_test.max() + 1e-10,
                                                       np.min([100, len(np.unique(mean_y_group_test))]))
                    kde_groupmean_test = FFTKDE(kernel='gaussian', bw="ISJ").fit(mean_y_group_test)
                    probabilities_groupmean_test = kde_groupmean_test.evaluate(edges_groupmean_test)
                    plt.plot(edges_groupmean_test, probabilities_groupmean_test)
                else:
                    plt.plot(mean_y_group_test)

            if b_hats_true is not None:
                legend += ["b_hat_true"]
                if mode == "kde":
                    edges_b_true = np.linspace(re_true.min() - 1e-10, re_true.max() + 1e-10,
                                               np.min([100, len(np.unique(re_true))]))
                    kde_b_true = FFTKDE(kernel='gaussian', bw="ISJ").fit(re_true)
                    probabilities_b_true = kde_b_true.evaluate(edges_b_true)
                    plt.plot(edges_b_true, probabilities_b_true)
                else:
                    plt.plot(re_true)

            if b_hats_lmmnn is not None:
                legend += ["LMMNN"]
                if mode == "kde":
                    edges_lmmnn = np.linspace(re_lmmnn.min() - 1e-10, re_lmmnn.max() + 1e-10,
                                              np.min([100, len(np.unique(re_lmmnn))]))
                    kde_lmmnn = FFTKDE(kernel='gaussian', bw="ISJ").fit(re_lmmnn)
                    probabilities_lmmnn = kde_lmmnn.evaluate(edges_lmmnn)
                    plt.plot(edges_lmmnn, probabilities_lmmnn)
                else:
                    plt.plot(re_lmmnn)
            plt.legend(legend)

            plt.show()

    elif target == "categorical":

        num_classes = np.unique(y_train).shape[0]
        y_train = tf.one_hot(y_train, num_classes).numpy()
        y_test = tf.one_hot(y_test, num_classes).numpy()
        for c in range(num_classes):
            res_c = {}
            for re_num in range(len(model_menn_info["random_effects"])):
                res_c[re_num] = {}

                re = model_menn_info["random_effects"][re_num][c]
                if b_hats_true is not None:
                    re_true = b_hats_true[re_num][:,c]
                    if np.std(re_true)==0:
                        print(f"Skip RE {re_num} for class {c} as it has zero variance")
                        continue

                train_encoder = TargetEncoder(handle_missing="value",handle_unknown="value").fit(Z_train[:, re_num].astype(object), y_train[:,c])
                mean_y_group_train = train_encoder.transform(np.unique(Z_train[:, re_num])).values.ravel()
                mean_y_group_train = np.array(list(map(logit,mean_y_group_train)))
                if test_data is not None:
                    test_encoder = TargetEncoder(handle_missing="value", handle_unknown="value").fit(
                        Z_test[:, re_num].astype(object), y_test[:, c])
                    mean_y_group_test = test_encoder.transform(np.unique(Z_train[:, re_num])).values.ravel()
                    mean_y_group_test = np.array(list(map(logit,mean_y_group_test)))

                if normalize:
                    re = normalize_func(re)
                    mean_y_group_train = normalize_func(mean_y_group_train)
                    if test_data is not None:
                        mean_y_group_test = normalize_func(mean_y_group_test)
                    if b_hats_true is not None:
                        re_true = normalize_func(re_true)

                legend = ["MENN", "Cluster Mean Train", "Cluster Mean Test"]

                if mode == "kde":
                    edges_menn = np.linspace(re.min() - 1e-10, re.max() + 1e-10, np.min([100, len(np.unique(re))]))
                    kde_menn = FFTKDE(kernel='gaussian', bw="ISJ").fit(re)
                    probabilities_menn = kde_menn.evaluate(edges_menn)
                    plt.plot(edges_menn, probabilities_menn)
                else:
                    plt.plot(re)

                if mode == "kde":
                    edges_groupmean_train = np.linspace(mean_y_group_train.min() - 1e-10, mean_y_group_train.max() + 1e-10,
                                                        np.min([100, len(np.unique(mean_y_group_train))]))
                    kde_groupmean_train = FFTKDE(kernel='gaussian', bw="ISJ").fit(mean_y_group_train)
                    probabilities_groupmean_train = kde_groupmean_train.evaluate(edges_groupmean_train)
                    plt.plot(edges_groupmean_train, probabilities_groupmean_train)
                else:
                    plt.plot(mean_y_group_train)
                if test_data is not None:
                    if mode == "kde":
                        edges_groupmean_test = np.linspace(mean_y_group_test.min() - 1e-10, mean_y_group_test.max() + 1e-10,
                                                           np.min([100, len(np.unique(mean_y_group_test))]))
                        kde_groupmean_test = FFTKDE(kernel='gaussian', bw="ISJ").fit(mean_y_group_test)
                        probabilities_groupmean_test = kde_groupmean_test.evaluate(edges_groupmean_test)
                        plt.plot(edges_groupmean_test, probabilities_groupmean_test)
                    else:
                        plt.plot(mean_y_group_test)

                if b_hats_true is not None:
                    legend += ["b_hat_true"]
                    if mode == "kde":
                        edges_b_true = np.linspace(re_true.min() - 1e-10, re_true.max() + 1e-10,
                                                   np.min([100, len(np.unique(re_true))]))
                        kde_b_true = FFTKDE(kernel='gaussian', bw="ISJ").fit(re_true)
                        probabilities_b_true = kde_b_true.evaluate(edges_b_true)
                        plt.plot(edges_b_true, probabilities_b_true)
                    else:
                        plt.plot(re_true)

                plt.legend(legend)
                plt.title(f"Random effect {re_num} for class {c}")
                plt.show()


def plot_modeling_history(model_menn_info, target, metrics = None, num_samples=0, perc_last=0.9):
    loss_history = model_menn_info["loss_history"]
    samples = model_menn_info["effect_z_samples"]
    num_re = len(model_menn_info["_stddev_z"])
    if target == "categorical":
        num_classes = model_menn_info["_stddev_z"][0].shape[0]

    print("Plot losses")
    # Convergence check for loss
    plt.plot(loss_history["train_loss"])
    plt.plot(loss_history["val_loss"])
    plt.ylabel(r'Loss $-\log$ $p(y\mid\mathbf{x})$')
    plt.xlabel('Iteration')
    plt.title("Training and validation loss")
    plt.show()

    # Convergence check for loss
    plt.plot(loss_history["train_loss"])
    plt.ylabel(r'Loss $-\log$ $p(y\mid\mathbf{x})$')
    plt.xlabel('Iteration')
    plt.title("Training loss")
    plt.show()

    # Convergence check for loss
    plt.plot(loss_history["val_loss"])
    plt.ylabel(r'Loss $-\log$ $p(y\mid\mathbf{x})$')
    plt.xlabel('Iteration')
    plt.title("Validation loss")
    plt.show()

    # Convergence check for FE val_loss
    plt.plot(loss_history["fe_loss"])
    plt.ylabel(r'Loss $-\log$ $p(y\mid\mathbf{x})$')
    plt.xlabel('Iteration')
    plt.title("Training loss (only FE)")
    plt.show()

    # Convergence check for FE val_loss
    plt.plot(loss_history["fe_loss_val"])
    plt.ylabel(r'Loss $-\log$ $p(y\mid\mathbf{x})$')
    plt.xlabel('Iteration')
    plt.title("Validation loss (only FE)")
    plt.show()


    # Convergence check for variance loss
    for num ,i in enumerate(np.array(loss_history["re_loss"]).transpose()):
        plt.plot(i)
        plt.ylabel(f"RE Loss {num}")
        plt.xlabel('Iteration')
        plt.title(f"Loss for RE {num}")
        plt.show()

    print("---------------------------------------------")
    print("Plot Parameter convergence")


    # Convergence check for variance parameter
    if target in ["binary", "continuous"]:
        plt.plot(loss_history["z"])
        plt.ylabel("sigma")
        plt.xlabel('Iteration')
        plt.legend([f"z{num}" for num in range(num_re)])
        plt.title("Convergence of all variance parameters")
        plt.show()
    elif target == "categorical":
        for c in range(num_classes):
            for re in range(num_re):
                plt.plot(np.array(loss_history["z"])[:, re, c])
                plt.ylabel("sigma")
                plt.xlabel('Iteration')
                plt.title(f"Convergence of all variance parameters for class {c}")
            plt.show()


    #### Plot auf iteration Ebene
    loss_history = model_menn_info["loss_history"]

    samples = model_menn_info["effect_z_samples"]
    if target in ["binary", "continuous"]:
        for q_num in range(len(model_menn_info["_stddev_z"])):
            plt.plot(np.array(loss_history["z"])[: ,q_num ]**2) # sample_stds
            plt.plot \
                ([model_menn_info["_stddev_z"][q_num ]**2 ] *model_menn_info["num_iters_actual"]) # re_std of final b_hat
            plt.plot([np.array(samples[q_num]).std(axis=1).mean( )**2 ] *model_menn_info
                ["num_iters_actual"]) # re_std of final b_hat
            plt.plot \
                ([np.array(samples[q_num])[round(len(samples[0] ) *perc_last):].std(axis=1).mean( )**2 ] *model_menn_info
                    ["num_iters_actual"]) # re_std of final b_hat
            plt.legend(["Var/Iter", "Var last iter", "Var (mean/iter) all samples", "Var (mean/iter) last 90%", f"Mean_RE_last{round(perc_last *100)}%"])
            plt.title(f"Samples and variances for RE={q_num}")
            plt.show()
    elif target == "categorical":
        for c in range(num_classes):
            for q_num in range(len(model_menn_info["_stddev_z"])):
                plt.plot(np.array(loss_history["z"])[: ,q_num ][:,c]**2) # sample_stds
                plt.plot \
                    ([model_menn_info["_stddev_z"][q_num][c]**2 ] * model_menn_info["num_iters_actual"]) # re_std of final b_hat
                plt.plot([np.array(samples[q_num])[:,c].std(axis=1).mean( )**2 ] * model_menn_info
                    ["num_iters_actual"]) # re_std of final b_hat
                plt.plot \
                    ([np.array(samples[q_num])[round(len(samples[0] ) *perc_last):,c].std(axis=1).mean( )**2 ] *model_menn_info["num_iters_actual"]) # re_std of final b_hat
                plt.legend(["Var/Iter", "Var last iter", "Var (mean/iter) all samples", "Var (mean/iter) last 90%", f"Mean_RE_last{round(perc_last *100)}%"])
                plt.title(f"Samples and variances for RE={q_num} and calss {c}")
                plt.show()



    print("---------------------------------------------")
    print("Plot Evaluation metrics")

    # Evaluation metrics over iterations
    if metrics is not None:
        for metric in metrics:

            plt.plot(loss_history[f"train_{metric}"])
            plt.plot(loss_history[f"fe_train_{metric}"])
            plt.plot(loss_history[f"val_{metric}"])
            plt.plot(loss_history[f"fe_val_{metric}"])

            plt.ylabel(metric)
            plt.xlabel('Iteration')
            plt.legend([f"train_{metric}" ,f"fe_train_{metric}",
                        f"val_{metric}" ,f"fe_val_{metric}"])
            plt.title(f"Performance over iterations: {metric}")
            plt.show()

    print("---------------------------------------------")
    print("Plot sampling trajectory")
    if num_samples >0:
        if target in ["binary", "continuous"]:
            for re in range(num_re):
                num_samples = np.min([num_samples, len(samples[re][0])])

                # Effect transitions for first k effects
                for i in range(num_samples):
                    plt.plot(np.array(samples[re][:])[: ,i])

                plt.legend([i for i in range(num_samples)], loc='lower center', fontsize="xx-small", handleheight=0.2)
                plt.ylabel(f'Z{re} Effects')
                plt.xlabel('Iteration')
                plt.title(f'Effect transitions for first {num_samples} Z{re} effects')
                plt.show()


                # Random Effects for first k clusters
                for i in range(num_samples):
                    means = np.array([np.mean(np.array(samples[re][:])[round(num *( 1 -perc_last)):num ,i]) for num in range(1 ,len(samples[re][:]))])
                    plt.plot(means)

                plt.legend([i for i in range(num_samples)], loc='lower center', fontsize="xx-small", handleheight=0.2)
                plt.ylabel(f'Z{re} Effects')
                plt.xlabel('Iteration')
                plt.title \
                    (f'Sample mean of previous {round(perc_last *100)}% samples for first {num_samples} Z{re} effects')
                plt.show()
        elif target == "categorical":
            for c in range(num_classes):
                num_samples = np.min([num_samples, len(np.array(samples[re])[:,0])])

                # Effect transitions for first k effects
                for i in range(num_samples):
                    plt.plot(np.array(samples[re][:])[:,c, i])

                plt.legend([i for i in range(num_samples)], loc='lower center', fontsize="xx-small", handleheight=0.2)
                plt.ylabel(f'Z{re} Effects')
                plt.xlabel('Iteration')
                plt.title(f'Effect transitions for first {num_samples} Z{re} effects and class {c}')
                plt.show()

                # Random Effects for first k clusters
                for i in range(num_samples):
                    means = np.array([np.mean(np.array(samples[re][:])[round(num * (1 - perc_last)):num, c, i]) for num in
                                      range(1, len(samples[re][:]))])
                    plt.plot(means)

                plt.legend([i for i in range(num_samples)], loc='lower center', fontsize="xx-small", handleheight=0.2)
                plt.ylabel(f'Z{re} Effects')
                plt.xlabel('Iteration')
                plt.title \
                    (f'Sample mean of previous {round(perc_last * 100)}% samples for first {num_samples} Z{re} effects and class {c}')
                plt.show()









