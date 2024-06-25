import numpy as np
# import pandas as pd

import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()

from tensorflow.keras.layers import Dense, Embedding#, Dropout, Input
# # from tensorflow.keras import Model
# from tensorflow.keras.layers import ReLU
from tensorflow.keras.activations import sigmoid, softmax, linear
# from tensorflow.keras.optimizers import Adam
#
# from sklearn.metrics import f1_score as f1
# from tensorflow.keras.metrics import Accuracy,  MeanSquaredError, AUC
# from tensorflow_addons.metrics import F1Score, RSquare
# from sklearn.metrics import roc_auc_score
# from tensorflow.keras.metrics import CategoricalAccuracy as cat_acc
import time
from utils.evaluation import get_metrics

import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class RandomInterceptLayer(tf.Module):
    '''
    Single-Category, Random-Intercept Neural Networks.

    To choose the target distribution, specify target="continuous" for regression, target="binary" for binary
    classification and target="categorical" for multi-class classification.

    To fix the stochasticity for the experiments, fix the global random seed with tf.random_seed() as well as the random
    seeds for all non-deterministic functions. The main stochasticity is from the NN initialization and the e-step

    Parameters:
        qs: List with numbers of categories per RE
        # fe_model: Any keras model producing an output that fits the target distribution
        target: Target distribution, one of ["continuous", "binary", "categorical"]
        num_outputs: Output size of the fe_model. Should be 1 for continuous and binary targets and the number of
            classes for categorical.
        initial_stds: List of value(s) to initialise the std parameter(s)
        positive_constraint_method: Method used to ensure that std/variance stays positive. One of ["exp", "clip"]. Default: "clip"
        RS: Random seed

    '''

    def __init__(self, qs, target="continuous", num_outputs=1, initial_stds=[1.], initial_error_std=1.,
                 positive_constraint_method="clip", RS=42):
        self.qs = qs
        self.RS = RS
        self.target = target
        self.num_outputs = num_outputs
        self.initial_std = initial_stds

        if target == "continuous":
            self.link_function = linear
            self.distribution = tfd.Normal
            if positive_constraint_method == "exp":
                self._stddev_e = tfp.util.TransformedVariable(initial_error_std, bijector=tfb.Exp(), name="stddev_e")
            elif positive_constraint_method == "clip":
                self._stddev_e = tf.Variable(initial_error_std, name='stddev_e',
                                             constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty))

        elif target == "binary":
            self.link_function = sigmoid
            self.distribution = tfd.Bernoulli
        elif target == "categorical":
            self.link_function = softmax
            self.distribution = tfd.OneHotCategorical
        else:
            raise AttributeError("Invalid target type specified. Please specify the target parameter as one of \
                                 ['continuous', 'binary', 'categorical']")

        # Initialize sig2b
        if positive_constraint_method == "exp":
            self._stddev_z = [tfp.util.TransformedVariable(std, bijector=tfb.Exp(), name=f"stddev_z{num}") for num, std
                              in enumerate(self.initial_std)]
        elif positive_constraint_method == "clip":
            self._stddev_z = [
                tf.Variable(std, name=f'stddev_z{num}', constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty)) for
                num, std in enumerate(self.initial_std)]

    def __call__(self, fX, X, Z, training=False):
        # Let W, b be the parameters of the NN f, b the random effects and sigma the variance of the random effect.
        # Let Rho be the set of all parameters {W, b, sigma}

        # For regression, we assume y ~ Normal(f(X) + b, 1) where b ~ Normal(0, sigma)
        # Hence the probability density function is pdf(y; f(X)+b, 1) = exp(-0.5 (y-(f(x)+b))**2) / (2 pi)**0.5
        # Alternatively, we could formulate the problem like Simchoni as y ~ Normal(f(X), V(sigma))
        if self.target == "continuous":
            priors = [
                # Set up prior distribution Normal(b; 0, sigma)
                tfd.MultivariateNormalDiag(
                    loc=tf.zeros(self.qs[num]),
                    scale_diag=self._stddev_z[num])
                for num in range(len(self._stddev_z))]

            likelihood = lambda *effect_z: tfd.Independent(
                self.distribution(
                    loc=tf.experimental.numpy.ravel(fX) +
                        tf.reduce_sum(
                            [tf.gather(effect_z[len(self.qs) - 1 - num], Z[:, num]) for num in range(len(self.qs))],
                            axis=0),
                    scale=self._stddev_e
                ),
                reinterpreted_batch_ndims=1)

        # For binary classification, we assume y ~ Bernoulli(eta) where eta = f(X) + b and b ~ Normal(0, sigma)
        # For multi-class classification, we assume y ~ Categorical(eta) where eta = f(X) + b and b ~ Normal(0, sigma) and b is a matrix q*c
        elif self.target in ["binary", "categorical"]:
            priors = [
                # Set up prior distribution Normal(b; 0, sigma)
                tfd.MultivariateNormalDiag(
                    loc=tf.zeros([self.qs[num],self.num_outputs]),
                    scale_diag=self._stddev_z[num])
                for num in range(len(self._stddev_z))]

            likelihood = lambda *effect_z: tfd.Independent(
                self.distribution(
                    logits=fX +
                           tf.reduce_sum([tf.gather(effect_z[len(self.qs) - 1 - num], Z[:, num]) for num in range(len(self.qs))], axis=0)
                ),
                reinterpreted_batch_ndims=1)

        # Set the joint distribution as the prior and likelihood for the observed.
        joint = priors + [likelihood]

        # The MLE is solved as min_{W,b,sigma} -log p(y | X, Z; W, b, sigma) = min_{W,b,sigma} -log integral[ p(b; sigma) p(y | X, Z, b; W, b) db
        model = tfd.JointDistributionSequentialAutoBatched(joint)

        model._to_track = self

        return model

class RandomSlopeLayer(tf.Module):
    '''
    Single-Category, Random-Intercept Neural Networks.

    To choose the target distribution, specify target="continuous" for regression, target="binary" for binary
    classification and target="categorical" for multi-class classification.

    To fix the stochasticity for the experiments, fix the global random seed with tf.random_seed() as well as the random
    seeds for all non-deterministic functions. The main stochasticity is from the NN initialization and the e-step

    Parameters:
        qs: List with numbers of categories per RE
        # fe_model: Any keras model producing an output that fits the target distribution
        target: Target distribution, one of ["continuous", "binary", "categorical"]
        num_outputs: Output size of the fe_model. Should be 1 for continuous and binary targets and the number of
            classes for categorical.
        initial_stds: List of value(s) to initialise the std parameter(s)
        positive_constraint_method: Method used to ensure that std/variance stays positive. One of ["exp", "clip"]. Default: "clip"
        RS: Random seed

    '''

    def __init__(self, qs, target="continuous",
                 num_outputs=1, initial_stds=[1.], initial_error_std=1.,
                 positive_constraint_method="clip",
                 embed_z = False,
                 embed_x=False,
                 RS=42):

        self.embed_x = embed_x
        self.embed_z = embed_z
        if self.embed_z:
            self.qs = [round(np.sqrt(q)) for q in qs]
        else:
            self.qs = qs

        self.RS = RS
        self.target = target
        self.num_outputs = num_outputs
        self.initial_std = initial_stds

        if target == "continuous":
            self.link_function = linear
            self.distribution = tfd.Normal
            if positive_constraint_method == "exp":
                self._stddev_e = tfp.util.TransformedVariable(initial_error_std, bijector=tfb.Exp(), name="stddev_e")
            elif positive_constraint_method == "clip":
                self._stddev_e = tf.Variable(initial_error_std, name='stddev_e',
                                             constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty))

        elif target == "binary":
            self.link_function = sigmoid
            self.distribution = tfd.Bernoulli
        elif target == "categorical":
            self.link_function = softmax
            self.distribution = tfd.OneHotCategorical
        else:
            raise AttributeError("Invalid target type specified. Please specify the target parameter as one of \
                                 ['continuous', 'binary', 'categorical']")

        # Initialize sig2b
        if positive_constraint_method == "exp":
            self._stddev_z = [tfp.util.TransformedVariable(std, bijector=tfb.Exp(), name=f"stddev_z{num}") for num, std
                              in enumerate(self.initial_std)]
        elif positive_constraint_method == "clip":
            self._stddev_z = [
                tf.Variable(std, name=f'stddev_z{num}', constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty)) for
                num, std in enumerate(self.initial_std)]

    def __call__(self, fX, X, Z, training=False):
        # Let W, b be the parameters of the NN f, b the random effects and sigma the variance of the random effect.
        # Let Rho be the set of all parameters {W, b, sigma}

        # For regression, we assume y ~ Normal(f(X) + b, 1) where b ~ Normal(0, sigma)
        # Hence the probability density function is pdf(y; f(X)+b, 1) = exp(-0.5 (y-(f(x)+b))**2) / (2 pi)**0.5
        # Alternatively, we could formulate the problem like Simchoni as y ~ Normal(f(X), V(sigma))
        if self.target == "continuous":
            print("Currently not implemented")
            # priors = [
            #     # Set up prior distribution Normal(b; 0, sigma)
            #     tfd.MultivariateNormalDiag(
            #         loc=tf.zeros(X.shape[1],self.qs[num]),
            #         scale_diag=self._stddev_z[num])
            #     for num in range(len(self._stddev_z))]
            #
            # likelihood = lambda *effect_z: tfd.Independent(
            #     self.distribution(
            #         loc=tf.experimental.numpy.ravel(fX) +
            #             tf.reduce_sum(
            #                 [tf.gather(effect_z[len(self.qs) - 1 - num], Z[:, num]) for num in range(len(self.qs))],
            #                 axis=0),
            #         scale=self._stddev_e
            #     ),
            #     reinterpreted_batch_ndims=1)

        # For binary classification, we assume y ~ Bernoulli(eta) where eta = f(X) + b and b ~ Normal(0, sigma)
        # For multi-class classification, we assume y ~ Categorical(eta) where eta = f(X) + b and b ~ Normal(0, sigma) and b is a matrix q*c
        elif self.target in ["binary", "categorical"]:
            priors = [
                # Set up prior distribution Normal(b; 0, sigma)
                tfd.MultivariateNormalDiag(
                    loc=tf.zeros([self.qs[num], X.shape[1], self.num_outputs]),
                    scale_diag=self._stddev_z[num])
                for num in range(len(self._stddev_z))]

            if self.embed_z:
                likelihood = lambda *effect_z: tfd.Independent(
                    self.distribution(
                        logits=fX +
                               tf.reduce_sum(
                                   [tf.reduce_sum(
                                       tf.multiply(
                                           tf.reshape(tf.repeat(X, self.num_outputs), X.shape + [self.num_outputs]),
                                           tf.tensordot(Z[num], effect_z[len(self.qs) - 1 - num],[1,0])
                                       ), axis=1)
                                       for num in range(len(self.qs))],
                                   axis=0)

                    ),
                    reinterpreted_batch_ndims=1)

            else:
                likelihood = lambda *effect_z: tfd.Independent(
                    self.distribution(
                        logits=fX +
                               tf.reduce_sum(
                                   [tf.reduce_sum(
                                       tf.multiply(
                                           tf.reshape(tf.repeat(X,self.num_outputs),X.shape+[self.num_outputs]),
                                           tf.gather(effect_z[len(self.qs) - 1 - num], Z[:, num])
                                       ), axis=1)
                                       for num in range(len(self.qs))],
                                   axis=0)

                    ),
                    reinterpreted_batch_ndims=1)

        # Set the joint distribution as the prior and likelihood for the observed.
        joint = priors + [likelihood]

        # The MLE is solved as min_{W,b,sigma} -log p(y | X, Z; W, b, sigma) = min_{W,b,sigma} -log integral[ p(b; sigma) p(y | X, Z, b; W, b) db
        model = tfd.JointDistributionSequentialAutoBatched(joint)

        model._to_track = self

        return model


class MCMCSamplingCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 num_mcmc_samples=1,
                 step_size=0.01,
                 perc_burnin=0.1,
                 num_burnin_steps=0,
                 warm_restart=None):
        super().__init__()

        self.num_mcmc_samples = tf.constant(num_mcmc_samples)
        self.perc_burnin = perc_burnin
        self.num_burnin_steps = num_burnin_steps
        self.warm_restart = warm_restart
        self.step_size = tf.Variable(step_size,trainable=False)
        self.step_sizes = []


    def on_train_begin(self, logs=None):
        self.mcmc_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=self.model.target_log_prob_fn,
            step_size=self.step_size)

        self.get_mcmc_kernel = lambda step_size: tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=self.model.target_log_prob_fn,
            step_size=step_size)

        # self.mcmc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        #     inner_kernel=tfp.mcmc.NoUTurnSampler(
        #         target_log_prob_fn=self.model.target_log_prob_fn,
        #         step_size=self.step_size),
        #     num_adaptation_steps=500,
        #     target_accept_prob=0.651)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch==0:
            self.model.all_samples.extend(([[state[num] for state in self.model.current_state] for num in range(1)]))

            self.model.mean_samples = [tf.reduce_mean([sample[q] for sample in self.model.all_samples[round(epoch*(self.perc_burnin)):]], axis=0) for q in
                                       range(len(self.model.qs))]
        if self.model.fe_pretraining:
            if self.model.fe_converged:
                self.run_sampling(epoch)
            else:
                self.model.acceptance_rates.append(-1)
        else:
            self.run_sampling(epoch)

    def run_sampling(self,epoch):
        self.model.fX.assign(self.model.fe_model(self.model.X, training=False))

        if self.model.embed_x:
            self.model.X_embedded.assign(self.model.X_embed_model(self.model.X, training=False))

        if self.model.embed_z:
            for q_num in range(len(self.model.qs)):
                self.model.Z_embedded[q_num].assign(self.model.Z_embed_models[q_num](self.model.Z[:,q_num], training=False))

                ## Find initial step size
        # if self.model.previous_kernel_results.log_accept_ratio == -np.inf:
        # if len(self.model.acceptance_rates)>0 and self.model.acceptance_rates[-1]<0.5:
        if len(self.model.acceptance_rates)>0 and self.model.acceptance_rates[-1]<0.0001:
            # self.mcmc_kernel.parameters["step_size"] = self.mcmc_kernel.parameters["step_size"]/2
            # self.model.previous_kernel_results["new_step_size"] = self.model.previous_kernel_results.step_size/2
            # setattr(self.model.previous_kernel_results, "new_step_size", self.model.previous_kernel_results.step_size/2)
            self.step_size.assign(self.step_size/2)
            print(f"Adapt step size to {float(self.step_size)}")


        if self.warm_restart!=None and epoch>0:
            ## Warm restart
            # if self.model.previous_kernel_results.log_accept_ratio == -np.inf:
                # restart = True
            # else:
                # restart = False
            # else:
            restart = ((epoch + 1) % self.warm_restart) == 0 and epoch != 0

            if restart:
                print("\n Warm restart to unstuck the chain")
                if self.model.embed_z and self.model.embed_x:
                    self.model.current_state = self.model.data_model(self.model.fX, self.model.X_embedded, self.model.Z_embedded).sample(1, seed=self.model.RS)[:-1]
                elif self.model.embed_z and not self.model.embed_x:
                    self.model.current_state = self.model.data_model(self.model.fX, self.model.X, self.model.Z_embedded).sample(1, seed=self.model.RS)[:-1]
                elif not self.model.embed_z and self.model.embed_x:
                    self.model.current_state = self.model.data_model(self.model.fX, self.model.X_embedded, self.model.Z).sample(1, seed=self.model.RS)[:-1]
                else:
                    self.model.current_state = self.model.data_model(self.model.fX, self.model.X, self.model.Z).sample(1, seed=self.model.RS)[:-1]

        print("\n Start sampling for epoch {} of training".format(epoch + 1))
        start = time.time()
        new_state, self.model.previous_kernel_results = self.get_mcmc_samples(self.model.current_state,
                                                                              tf.constant(self.num_mcmc_samples),
                                                                              None
                                                                                               )
        # self.model.divide_constants.assign(
        #     list(1/np.mean(self.model.data_model._stddev_z,axis=1))+[1.])
        # self.model.divide_constants.assign(
        #     list((lambda x: 1+(x-x.mean()))(np.array(1+tf.math.softmax(1/len(self.model.qs)+0.5*tf.math.softmax(np.abs([np.mean(i) for i in self.model.previous_kernel_results.grads_target_log_prob]))))))+[1.])
        # print(np.round(self.model.divide_constants,2))
        try:
            log_accept_ratio = self.model.previous_kernel_results.log_accept_ratio
        except:
            log_accept_ratio = self.model.previous_kernel_results.inner_results.log_accept_ratio
        acceptance_rate = tf.math.exp(tf.minimum(log_accept_ratio, 0.))

        self.step_sizes.append(float(self.step_size))

        end = time.time()


        self.model.current_state = [tf.identity(i) for i in new_state]
        # Todo: Append all current states
        self.model.acceptance_rates.append(acceptance_rate)
        # self.model.all_samples.append(
        #     [tf.math.reduce_mean(self.model.current_state[q_num], axis=0) for q_num in range(len(self.model.qs))])
        self.model.all_samples.extend(([[state[num] for state in self.model.current_state] for num in range(self.num_mcmc_samples)]))

        self.model.mean_samples = [tf.reduce_mean([sample[q] for sample in self.model.all_samples[round(epoch*(self.perc_burnin)):]], axis=0) for q in
                                   range(len(self.model.qs))]

        self.model.e_step_times.append(round(end - start, 2))

    # def on_epoch_end(self, epoch, logs=None):
        for q_num in range(len(self.model.qs)):
                self.model.data_model.trainable_variables[q_num].assign(
                    tf.math.reduce_std(self.model.current_state[q_num][-1],axis=0))

        self.model.stds.append([tf.identity(i) for i in self.model.data_model._stddev_z])

    @tf.function(reduce_retracing=True)  # autograph=False, jit_compile=True, reduce_retracing=True)
    def get_mcmc_samples(self, current_state, num_mcmc_samples=tf.constant(1), previous_kernel_results=None):
        samples, _, previous_kernel_results = tfp.mcmc.sample_chain(
            kernel=self.get_mcmc_kernel(self.step_size), num_results=num_mcmc_samples,
            current_state=[state[-1] for state in current_state],
            num_burnin_steps=self.num_burnin_steps,
            trace_fn=None, previous_kernel_results=previous_kernel_results,
            return_final_kernel_results=True, seed=self.model.RS)
        #     current_state=[sample[-1] for sample in samples]

        return samples, previous_kernel_results


class PrintMetrics(tf.keras.callbacks.Callback):
    def __init__(self, X, Z, y, X_val, Z_val, y_val):
        super().__init__()
        self.X = X
        self.Z = Z
        self.y = y
        self.X_val = X_val
        self.Z_val = Z_val
        self.y_val = y_val

        self.early_stopping_metric_fe = "fe_auc_val"
        self.current_best_fe = -np.inf

    def on_train_begin(self, logs=None):
        # self.model.losses_history = {}
        self.model.performance_history = {}
        #
        # self.model.losses_history["me_loss"] = []
        # self.model.losses_history["me_loss_val"] = []
        # self.model.losses_history["fe_loss"] = []
        # self.model.losses_history["fe_loss_val"] = []
        #
        # self.model.performance_history["me_auc"] = []
        self.model.performance_history["me_auc_val"] = []
        # self.model.performance_history["fe_auc"] = []
        self.model.performance_history["fe_auc_val"] = []
        #
        # self.model.performance_history["own_eval_fe"] = []
        # self.model.performance_history["own_eval"] = []

        self.me_auc = 0.
        self.val_me_auc = 0.
        self.fe_auc = 0.
        self.val_fe_auc = 0.

        self.me_loss = 0.
        self.val_me_loss = 0.
        self.fe_loss = 0.
        self.val_fe_loss = 0.
        self.current_stds = None
        self.acceptance_rate = None

    def on_epoch_end(self, epoch, logs={}):
        # Get predictions
        y_pred, y_pred_fe = self.model((self.X, self.Z), training=False)
        y_pred_val, y_pred_fe_val = self.model((self.X_val, self.Z_val), training=False)

        if self.model.target=="binary":
            # Get loss
            self.me_loss  = tf.keras.losses.BinaryCrossentropy()(self.y, y_pred)
            self.fe_loss = tf.keras.losses.BinaryCrossentropy()(self.y, y_pred_fe)
            self.me_loss_val = tf.keras.losses.BinaryCrossentropy()(self.y_val, y_pred_val)
            self.fe_loss_val = tf.keras.losses.BinaryCrossentropy()(self.y_val, y_pred_fe_val)

            # Update metrics
            self.me_auc = tf.keras.metrics.AUC()(self.y, y_pred)
            self.fe_auc = tf.keras.metrics.AUC()(self.y, y_pred_fe)
            self.me_auc_val = tf.keras.metrics.AUC()(self.y_val, y_pred_val)
            self.fe_auc_val = tf.keras.metrics.AUC()(self.y_val, y_pred_fe_val)

            # self.model.losses_history["me_loss"].append(self.me_loss)
            # self.model.losses_history["fe_loss"].append(self.fe_loss)
            # self.model.losses_history["me_loss_val"].append(self.me_loss_val)
            # self.model.losses_history["fe_loss_val"].append(self.fe_loss_val)
            #
            # self.model.performance_history["me_auc"].append(self.me_auc)
            self.model.performance_history["me_auc_val"].append(self.me_auc_val)
            # self.model.performance_history["fe_auc"].append(self.fe_auc)
            self.model.performance_history["fe_auc_val"].append(self.fe_auc_val)

            # eval_me = get_metrics(self.y_val, y_pred_val, target=self.model.target)
            # self.model.performance_history["own_eval_fe"].append(
            #     get_metrics(self.y_val, y_pred_fe_val, target=self.model.target))
            # self.model.performance_history["own_eval"].append(eval_me)

        elif self.model.target == "categorical":
            # Get loss
            self.me_loss = tf.keras.losses.CategoricalCrossentropy()(self.y, y_pred).numpy()
            self.fe_loss = tf.keras.losses.CategoricalCrossentropy()(self.y, y_pred_fe).numpy()
            self.me_loss_val = tf.keras.losses.CategoricalCrossentropy()(self.y_val, y_pred_val).numpy()
            self.fe_loss_val = tf.keras.losses.CategoricalCrossentropy()(self.y_val, y_pred_fe_val).numpy()

            # Update metrics
            self.me_auc = tf.keras.metrics.AUC(multi_label=True)(self.y, y_pred).numpy()
            self.fe_auc = tf.keras.metrics.AUC(multi_label=True)(self.y, y_pred_fe).numpy()
            self.me_auc_val = tf.keras.metrics.AUC(multi_label=True)(self.y_val, y_pred_val).numpy()
            # self.me_auc_val = tf.keras.metrics.SparseCategoricalAccuracy(name="acc_me")(tf.argmax(self.y_val,axis=1), y_pred_val)
            self.fe_auc_val = tf.keras.metrics.AUC(multi_label=True)(self.y_val, y_pred_fe_val).numpy()

            # self.model.losses_history["me_loss"].append(self.me_loss)
            # self.model.losses_history["fe_loss"].append(self.fe_loss)
            # self.model.losses_history["me_loss_val"].append(self.me_loss_val)
            # self.model.losses_history["fe_loss_val"].append(self.fe_loss_val)
            #
            # self.model.performance_history["me_auc"].append(self.me_auc)
            self.model.performance_history["me_auc_val"].append(self.me_auc_val )
            # self.model.performance_history["fe_auc"].append(self.fe_auc)
            self.model.performance_history["fe_auc_val"].append(self.fe_auc_val)

            # eval_me = get_metrics(self.y_val, y_pred_val, target=self.model.target)
            # self.model.performance_history["own_eval_fe"].append(get_metrics(self.y_val, y_pred_fe_val, target=self.model.target))
            # self.model.performance_history["own_eval"].append(eval_me)


        self.current_stds = np.mean(self.model.data_model._stddev_z,axis=1)
        self.acceptance_rate = self.model.acceptance_rates[-1]

        logs["me_loss"] = self.me_loss
        logs["me_loss_val"] = self.me_loss_val
        logs["fe_loss"] = self.fe_loss
        logs["fe_loss_val"] = self.fe_loss_val

        logs["me_auc"] = self.me_auc
        logs["me_auc_val"] = self.me_auc_val
        logs["fe_auc"] = self.fe_auc
        logs["fe_auc_val"] = self.fe_auc_val
        logs["stds"] = self.current_stds
        logs["acceptance_rate"] = self.acceptance_rate

        if self.model.early_stopping_fe is not None:
            if not self.model.fe_converged:
                if self.early_stopping_metric_fe in ["fe_auc_val"]:
                    self.early_stop_condition_fe = self.model.performance_history[self.early_stopping_metric_fe][-1] <= self.current_best_fe
                # else:
                    # early_stop_condition_fe = self.model.performance_history[self.early_stopping_metric_fe][-1] >= self.current_best_fe

                if self.early_stop_condition_fe:
                    self.early_stop_fe += 1
                else:
                    self.early_stop_fe = 0
                    self.current_best_fe = self.model.performance_history[self.early_stopping_metric_fe][-1]
                if self.early_stop_fe == self.model.early_stopping_fe:
                    print(f"\n Early stopping of FE by {self.early_stopping_metric_fe} at {epoch+1} epochs")
                    self.model.fe_converged.assign(True)
                    self.model.fe_model.trainable = False
                    self.model.fe_model.compile()


class MixedEffectsNetwork(tf.keras.Model):

    def __init__(self, X, Z, y, fe_model,
                 qs = None, target=None, initial_stds = None,
                 mode="intercepts",
                 embed_z=False,
                 embed_x=False,
                 fe_loss_weight=0.,
                 early_stopping_fe=None,
                 fe_pretraining=False,
                 RS=42):
        super(MixedEffectsNetwork, self).__init__()

        '''
        Notes:
         - Z and y need to be tensors, Z needs to be integer tensor and y needs to be float

        Todos:
            - For efficieny make fX be computed as few times as possible

        '''
        self.early_stopping_fe = early_stopping_fe
        self.fe_converged = tf.Variable(False,trainable=False)
        self.fe_pretraining = tf.Variable(fe_pretraining,trainable=False)

        self.fe_loss_weight = fe_loss_weight
        self.mode = mode
        self.embed_z = embed_z
        self.embed_x = embed_x

        # Set random state
        self.RS = RS

        self.fe_model = fe_model
        self.Z = Z
        self.X = X
        self.fX = tf.Variable(self.fe_model(X, training=False),trainable=False)
        if self.embed_x:
            self.X_embed_model = Dense(round(np.sqrt(X.shape[1])),activation="relu")
            self.X_embedded = tf.Variable(self.X_embed_model(X, training=False),trainable=False)


        # Retrieve target type
        self.target = target

        # Retrieve output dimension
        if self.target == "categorical":
            self.num_outputs = y.shape[1]
        else:
            self.num_outputs = 1

        # Get cardinatlities and initialize standard deviations
        if qs is None:
            self.qs = tf.reduce_max(Z, axis=0) + 1
        else:
            self.qs = qs

        if self.embed_z:
            self.Z_embed_models = [Embedding(self.qs[q_num], round(np.sqrt(self.qs[q_num])), input_length=1) for q_num in range(len(self.qs))]
            self.Z_embedded = [tf.Variable(self.Z_embed_models[q_num](Z[:,q_num], training=False),trainable=False) for q_num in range(len(self.qs))]

        if self.mode == "intercepts":
            self.initial_stds =  initial_stds

            self.data_model = RandomInterceptLayer(
                qs=self.qs,
                initial_stds=self.initial_stds,
                target=self.target,
                num_outputs=self.num_outputs,
                RS=self.RS
            )
        elif self.mode == "slopes":
            if self.embed_x:
                self.initial_stds = [[[0.1] * self.num_outputs] * self.X_embedded.shape[1]] * len(self.qs)
            else:
                self.initial_stds = [[[0.1] * self.num_outputs] * X.shape[1]] * len(self.qs)

            self.data_model = RandomSlopeLayer(
                qs=self.qs,
                initial_stds=self.initial_stds,
                target=self.target,
                num_outputs=self.num_outputs,
                RS=self.RS,
                embed_x=self.embed_x,
                embed_z=self.embed_z
            )


        if self.embed_z and self.embed_x:
            self.target_log_prob_fn = lambda *x: self.data_model(self.fX, self.X_embedded, self.Z_embedded).log_prob(x + (y,))
            # The initial state for HMC is randomly sampled from the prior distribution
            self.current_state = self.data_model(self.fX, self.X_embedded, self.Z_embedded).sample(1, seed=self.RS)[:-1]
        elif self.embed_z and not self.embed_x:
            self.target_log_prob_fn = lambda *x: self.data_model(self.fX, X, self.Z_embedded).log_prob(x + (y,))
            # The initial state for HMC is randomly sampled from the prior distribution
            self.current_state = self.data_model(self.fX, X, self.Z_embedded).sample(1, seed=self.RS)[:-1]
        elif not self.embed_z and self.embed_x:
            self.target_log_prob_fn = lambda *x: self.data_model(self.fX, self.X_embedded, Z).log_prob(x + (y,))
            # The initial state for HMC is randomly sampled from the prior distribution
            self.current_state = self.data_model(self.fX, self.X_embedded, Z).sample(1, seed=self.RS)[:-1]
        else:
            self.current_state = self.data_model(self.fX, X, Z).sample(1, seed=self.RS)[:-1]
            # self.divide_constants = tf.Variable([0.1]*len(self.qs)+[1.],trainable=False,dtype=tf.float32)
            # self.divide_constants = tf.Variable([100,10,100]+[1.],trainable=False,dtype=tf.float32)
            #
            # self.target_log_prob_fn = lambda *x: tf.reduce_sum(tf.vectorized_map(self.log_prob_divide, [tf.convert_to_tensor(
            #     self.data_model(fe_model(X), X, Z).log_prob_parts(
            #         x + (y,))), self.divide_constants]))
            # self.target_log_prob_fn = tf.reduce_sum(tf.convert_to_tensor(
            #     self.data_model(self.fX, X, Z).log_prob_parts(
            #         [i[0] for i in self.current_state] + [y])) / tf.convert_to_tensor(
            #     self.qs + [y.shape[0]], dtype=tf.float32))
            self.target_log_prob_fn = lambda *x: self.data_model(self.fX, X, Z).log_prob(x + (y,))
            # The initial state for HMC is randomly sampled from the prior distribution

        # Add zero state for FE component
        self.zero_state = [tf.zeros(state[0].shape, dtype=tf.float32) for state in self.current_state]
        self.previous_kernel_results = None

        self.all_samples = []
        self.e_step_times = []
        self.stds = []
        self.acceptance_rates = []

    def call(self, inputs, training=None):
        X, Z = inputs

        fX = self.fe_model(X, training=training)
        if self.embed_z:
            Z = [self.Z_embed_models[q_num](Z[:,q_num], training=training) for q_num in range(len(self.qs))]
        if self.embed_x:
            X = self.X_embed_model(X, training=training)

        if training:
            ### m_mode = "mean"

            if self.mode == "intercepts":
                out = tf.reduce_mean(fX + tf.reduce_sum(
                    [tf.gather(self.current_state[num], Z[:, num],axis=1) for num in range(len(self.qs))], axis=0),
                                     axis=0)
            elif self.mode == "slopes" and self.embed_z:
                out = tf.reduce_mean(fX + tf.reduce_sum(
                    [tf.reduce_sum(
                        tf.multiply(
                            tf.reshape(tf.repeat(X, self.num_outputs), tf.concat([tf.shape(X), [self.num_outputs]],axis=0)),
                            tf.reshape(tf.tensordot(Z[num], self.current_state[num],[1,1]),[tf.shape(self.current_state[num])[0],tf.shape(Z[num])[0],tf.shape(self.current_state[num])[2],self.num_outputs])
                        ), axis=2)
                        for num in range(len(self.qs))],
                    axis=0),
                                     axis=0)
            elif self.mode == "slopes" and not self.embed_z:
                out = tf.reduce_mean(fX + tf.reduce_sum(
                    [tf.reduce_sum(
                        tf.multiply(
                            tf.reshape(tf.repeat(X, self.num_outputs), tf.concat([tf.shape(X), [self.num_outputs]],axis=0)),
                            tf.gather(self.current_state[num], Z[:, num],axis=1)
                        ), axis=2)
                        for num in range(len(self.qs))],
                    axis=0),
                                     axis=0)

            ### m_mode = "last"
            # out = fX + tf.reduce_sum(
            #     [tf.gather(self.current_state[num][-1], Z[:, num]) for num in range(len(self.qs))], axis=0)
            ### m_mode = "mean_condition"
            # out = fX + tf.reduce_sum(
            #     [tf.gather(self.mean_samples[num], Z[:, num]) for num in range(len(self.qs))], axis=0)


        else:
            ### mean cond.
            if self.mode == "intercepts":
                out = fX + tf.reduce_sum(
                    [tf.gather(self.mean_samples[num], Z[:, num]) for num in range(len(self.qs))], axis=0)

            elif self.mode == "slopes" and self.embed_z:
                out = fX + tf.reduce_sum(
                    [tf.reduce_sum(
                        tf.multiply(
                            tf.reshape(tf.repeat(X, self.num_outputs),
                                       tf.concat([tf.shape(X), [self.num_outputs]], axis=0)),
                            tf.tensordot(Z[num], self.mean_samples[num],[1,0])
                        ), axis=1)
                        for num in range(len(self.qs))],
                    axis=0)

            elif self.mode == "slopes" and not self.embed_z:
                out = fX + tf.reduce_sum(
                               [tf.reduce_sum(
                                   tf.multiply(
                                       tf.reshape(tf.repeat(X,self.num_outputs),tf.concat([tf.shape(X), [self.num_outputs]],axis=0)),
                                       tf.gather(self.mean_samples[num], Z[:, num])
                                   ), axis=1)
                                   for num in range(len(self.qs))],
                               axis=0)



            ### last
            # out = fX + tf.reduce_sum(
            #     [tf.gather(self.current_state[num][-1], Z[:, num]) for num in range(len(self.qs))], axis=0)

        return self.data_model.link_function(out), self.data_model.link_function(fX)

    def compile(self,
                loss_class_me,
                loss_class_fe,
                # metric_class_me,
                # metric_class_fe,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
                ):
        super().compile()

        self.loss_class_me = loss_class_me
        self.loss_class_fe = loss_class_fe

        self.optimizer = optimizer

        # Loss trackers
        # self.loss_class_me_tracker = tf.keras.metrics.Mean(name='me_loss')
        # self.loss_class_fe_tracker = tf.keras.metrics.Mean(name='fe_loss')

        # self.metric_class_me = metric_class_me
        # self.metric_class_fe = metric_class_fe

    @property
    def metrics(self):
        return [#self.loss_class_me_tracker,
                # self.loss_class_fe_tracker,
                # self.metric_class_me,
                # self.metric_class_fe
                ]
    @tf.function()
    def update_step(self, X, Z, y):
        with tf.GradientTape() as tape:
            # Get predictions
            y_pred, fX = self((X, Z), training=True)
            loss_class_me = self.loss_class_me(y, y_pred)
            loss_class_fe = self.loss_class_fe(y, fX)

            # fX = self.fe_model(X, training=True)
            # loss_class_me = -self.data_model(fX, X, Z).log_prob(self.current_state + [y])
            # loss_class_fe = -self.data_model(fX, X, Z).log_prob(self.zero_state + [y])
            # #
            total_loss = loss_class_me + self.fe_loss_weight*loss_class_fe

        # y_pred, fX = self((X, Z), training=False)

        # Update gradients
        grads_class = tape.gradient(total_loss, self.fe_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads_class, self.fe_model.trainable_variables))

        return loss_class_me, loss_class_fe


    def train_step(self, data):
        (X, Z), y = data
        # if not self.fe_converged:
            # Unpack data
        # loss_class_me,loss_class_fe = self.update_step(X, Z, y)
        loss_class_me,loss_class_fe = tf.cond(self.fe_converged, true_fn=lambda: (tf.ones(1),tf.ones(1)), false_fn=lambda: self.update_step(X, Z, y))

        # Update metrics
        # self.metric_class_me.update_state(y, y_pred)
        # self.metric_class_fe.update_state(y, fX)

        # self.loss_class_me_tracker.update_state(loss_class_me)
        # self.loss_class_fe_tracker.update_state(loss_class_fe)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def log_prob_divide(self, inputs):
        log_prob, divide_constant = inputs
        return tf.divide(log_prob, divide_constant)

    # def test_step(self, data):
    #     # Unpack data
    #     (X, Z), y = data
    #
    #     y_pred, fX = self((X, Z), training=False)
    #
    #     loss_class_me = self.loss_class_me(y, y_pred)
    #     loss_class_fe = self.loss_class_fe(y, fX)
    #     # loss_class_me = -self.data_model(fX, X, Z).log_prob(self.current_state + [y])
    #     # loss_class_fe = -self.data_model(fX, X, Z).log_prob(self.zero_state + [y])
    #
    #     # Update metrics
    #     # self.metric_class_me.update_state(y, y_pred)
    #     # self.metric_class_fe.update_state(y, fX)
    #
    #     self.loss_class_me_tracker.update_state(loss_class_me)
    #     self.loss_class_fe_tracker.update_state(loss_class_fe)
    #
    #     return {m.name: m.result() for m in self.metrics}

    # @tf.function
    # def updt(self):
    #     fe_model = self.fe_model#tf.keras.models.clone_model(self.fe_model)
    #     fe_model.trainable = False
    #
    #     # for num, layer in enumerate(fe_model.trainable_variables):
    #     #     layer.assign(tf.identity(self.fe_model.trainable_variables[num]))
    #     #
    #     # tf.print(fe_model.layers[1].weights[0][:3,:3])
    #     # tf.print(self.fe_model.layers[1].weights[0][:3,:3])
    #
    #     return fe_model
