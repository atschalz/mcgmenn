'''
Simple 2D CNN classifiers, including domain adversarial and mixed effects extensions.
'''
import tensorflow as tf
import tensorflow.keras.layers as tkl
from .random_effects import RandomEffects

class ImageClassifier(tf.keras.Model):
    '''
    Simple 2D image binary classifier with 7 convolution blocks and 2 final dense layers.
    '''
    
    def __init__(self, name='classifier', **kwargs):
        """Simple 2D image binary classifier with 7 convolution blocks and 2 final dense layers.

        Args:
            name (str, optional): Model name. Defaults to 'classifier'.
        """        
        
        super(ImageClassifier, self).__init__(name=name, **kwargs)
        
        # D x D
        self.conv0 = tkl.Conv2D(64, 3, padding='same', name='conv0')
        self.bn0 = tkl.BatchNormalization(name='bn0')
        self.prelu0 = tkl.PReLU(name='prelu0')
        self.maxpool0 = tkl.MaxPool2D(name='maxpool0')
        # D/2 x D/2        
        self.conv1 = tkl.Conv2D(128, 3, padding='same', name='conv1')
        self.dropout1 = tkl.Dropout(0.5, name='dropout1')
        self.bn1 = tkl.BatchNormalization(name='bn1')
        self.prelu1 = tkl.PReLU(name='prelu1')
        self.maxpool1 = tkl.MaxPool2D(name='maxpool1')
        # D/4 x D/4
        self.conv2 = tkl.Conv2D(128, 3, padding='same', name='conv2')
        self.bn2 = tkl.BatchNormalization(name='bn2')
        self.prelu2 = tkl.PReLU(name='prelu2')
        self.maxpool2 = tkl.MaxPool2D(name='maxpool2')
        # D/8 x D/8
        self.conv3 = tkl.Conv2D(256, 3, padding='same', name='conv3')
        self.dropout3 = tkl.Dropout(0.5, name='dropout3')
        self.bn3 = tkl.BatchNormalization(name='bn3')
        self.prelu3 = tkl.PReLU(name='prelu3')
        self.maxpool3 = tkl.MaxPool2D(name='maxpool3')
        # D/16 x D/16
        self.conv4 = tkl.Conv2D(256, 3, padding='same', name='conv4')
        self.bn4 = tkl.BatchNormalization(name='bn4')
        self.prelu4 = tkl.PReLU(name='prelu4')
        self.maxpool4 = tkl.MaxPool2D(name='maxpool4')
        # D/32 x D/32
        self.conv5 = tkl.Conv2D(512, 3, padding='same', name='conv5')
        self.dropout5 = tkl.Dropout(0.5, name='dropout5')
        self.bn5 = tkl.BatchNormalization(name='bn5')
        self.prelu5 = tkl.PReLU(name='prelu5')
        self.maxpool5 = tkl.MaxPool2D(name='maxpool5')
        # # D/64 x D/64
        self.conv6 = tkl.Conv2D(512, 3, padding='valid', name='conv6')
        self.prelu6 = tkl.PReLU(name='prelu6')
        self.flatten = tkl.Flatten(name='flatten')
        self.dense = tkl.Dense(512, name='dense', activation='relu',
                               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01))
        self.out = tkl.Dense(1, name='output')
        self.sigmoid = tkl.Activation('sigmoid', name='sigmoid')
                
        
    def call(self, inputs, return_layer_activations=False):
        c0 = self.conv0(inputs)
        c0 = self.bn0(c0)
        c0 = self.prelu0(c0)
        
        c1 = self.maxpool0(c0)
        c1 = self.conv1(c1)
        c1 = self.dropout1(c1)
        c1 = self.bn1(c1)
        c1 = self.prelu1(c1)
        
        c2 = self.maxpool1(c1)
        c2 = self.conv2(c2)
        c2 = self.bn2(c2)
        c2 = self.prelu2(c2)
        
        c3 = self.maxpool2(c2)
        c3 = self.conv3(c3)
        c3 = self.dropout3(c3)
        c3 = self.bn3(c3)
        c3 = self.prelu3(c3)
        
        c4 = self.maxpool3(c3)
        c4 = self.conv4(c4)
        c4 = self.bn4(c4)
        c4 = self.prelu4(c4)
        
        c5 = self.maxpool4(c4)
        c5 = self.conv5(c5)
        c5 = self.dropout5(c5)
        c5 = self.bn5(c5)
        c5 = self.prelu5(c5)
        
        c6 = self.maxpool5(c5)
        c6 = self.conv6(c6)
        c6 = self.prelu6(c6)
        h = self.flatten(c6)
        h = self.dense(h)
        y = self.out(h)
        y = self.sigmoid(y)
        if return_layer_activations:
            return c0, c1, c2, c3, c4, c5, c6, h, y
        else:
            return y

class ClusterInputImageClassifier(ImageClassifier):
    '''
    Simple 2D image classifier that takes an additional input containing the
    one-hot encoded cluster membership of each sample. This input is
    concatenated to the flattened output of the last convolution layer, before
    the first dense layer.
    '''
    def call(self, inputs, return_layer_activations=False):
        x, z = inputs
        c0 = self.conv0(x)
        c0 = self.bn0(c0)
        c0 = self.prelu0(c0)
        
        c1 = self.maxpool0(c0)
        c1 = self.conv1(c1)
        c1 = self.dropout1(c1)
        c1 = self.bn1(c1)
        c1 = self.prelu1(c1)
        
        c2 = self.maxpool1(c1)
        c2 = self.conv2(c2)
        c2 = self.bn2(c2)
        c2 = self.prelu2(c2)
        
        c3 = self.maxpool2(c2)
        c3 = self.conv3(c3)
        c3 = self.dropout3(c3)
        c3 = self.bn3(c3)
        c3 = self.prelu3(c3)
        
        c4 = self.maxpool3(c3)
        c4 = self.conv4(c4)
        c4 = self.bn4(c4)
        c4 = self.prelu4(c4)
        
        c5 = self.maxpool4(c4)
        c5 = self.conv5(c5)
        c5 = self.dropout5(c5)
        c5 = self.bn5(c5)
        c5 = self.prelu5(c5)
        
        c6 = self.maxpool5(c5)
        c6 = self.conv6(c6)
        c6 = self.prelu6(c6)
        h = self.flatten(c6)
        h = tf.concat([h, z], axis=1)
        h = self.dense(h)
        y = self.out(h)
        y = self.sigmoid(y)
        if return_layer_activations:
            return c0, c1, c2, c3, c4, c5, c6, h, y
        else:
            return y
        
class AdversarialClassifier(tf.keras.Model):
    '''
    Domain adversarial classifier for the ImageClassifier. Receives the
    layer activations from the ImageClassifier as inputs and predicts
    cluster membership.
    '''
    def __init__(self, n_clusters, name='adversary', **kwargs):
        """Domain adversarial classifier for the ImageClassifier. Receives the
        layer activations from the ImageClassifier as inputs and predicts
        cluster membership.

        Args: 
            n_clusters (int): number of clusters 
            name (str, optional): Model name. Defaults to 'adversary'.
        """        
        super(AdversarialClassifier, self).__init__(name=name, **kwargs)
        self.n_clusters = n_clusters
        
        # D x D
        self.conv0 = tkl.Conv2D(64, 3, padding='same', name='conv0')
        self.bn0 = tkl.BatchNormalization(name='bn0')
        self.prelu0 = tkl.PReLU(name='prelu0')
        self.maxpool0 = tkl.MaxPool2D(name='maxpool0')
        # D/2 x D/2        
        self.conv1 = tkl.Conv2D(64, 3, padding='same', name='conv1')
        self.dropout1 = tkl.Dropout(0.5, name='dropout1')
        self.bn1 = tkl.BatchNormalization(name='bn1')
        self.prelu1 = tkl.PReLU(name='prelu1')
        self.maxpool1 = tkl.MaxPool2D(name='maxpool1')
        # D/4 x D/4
        self.conv2 = tkl.Conv2D(128, 3, padding='same', name='conv2')
        self.bn2 = tkl.BatchNormalization(name='bn2')
        self.prelu2 = tkl.PReLU(name='prelu2')
        self.maxpool2 = tkl.MaxPool2D(name='maxpool2')
        # D/8 x D/8
        self.conv3 = tkl.Conv2D(128, 3, padding='same', name='conv3')
        self.dropout3 = tkl.Dropout(0.5, name='dropout3')
        self.bn3 = tkl.BatchNormalization(name='bn3')
        self.prelu3 = tkl.PReLU(name='prelu3')
        self.maxpool3 = tkl.MaxPool2D(name='maxpool3')
        # D/16 x D/16
        self.conv4 = tkl.Conv2D(256, 3, padding='same', name='conv4')
        self.bn4 = tkl.BatchNormalization(name='bn4')
        self.prelu4 = tkl.PReLU(name='prelu4')
        self.maxpool4 = tkl.MaxPool2D(name='maxpool4')
        # D/32 x D/32
        self.conv5 = tkl.Conv2D(256, 3, padding='same', name='conv5')
        self.dropout5 = tkl.Dropout(0.5, name='dropout5')
        self.bn5 = tkl.BatchNormalization(name='bn5')
        self.prelu5 = tkl.PReLU(name='prelu5')
        self.maxpool5 = tkl.MaxPool2D(name='maxpool5')
        # # D/64 x D/64
        self.conv6 = tkl.Conv2D(512, 3, padding='valid', name='conv6')
        self.prelu6 = tkl.PReLU(name='prelu6')
        self.flatten = tkl.Flatten(name='flatten')
        self.dense = tkl.Dense(512, name='dense', activation='relu')
        self.out = tkl.Dense(n_clusters, name='output', activation='softmax')
        
    def call(self, inputs):
        c0, c1, c2, c3, c4, c5, c6, h = inputs
        
        x = self.conv0(c0)
        x = self.bn0(x)
        x = self.prelu0(x)
        
        x = self.maxpool0(x)
        x = tf.concat([x, c1], axis=-1)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        
        x = self.maxpool1(x)
        x = tf.concat([x, c2], axis=-1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        
        x = self.maxpool2(x)
        x = tf.concat([x, c3], axis=-1)
        x = self.conv3(x)
        x = self.dropout3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        
        x = self.maxpool3(x)
        x = tf.concat([x, c4], axis=-1)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.prelu4(x)
        
        x = self.maxpool4(x)
        x = tf.concat([x, c5], axis=-1)
        x = self.conv5(x)
        x = self.dropout5(x)
        x = self.bn5(x)
        x = self.prelu5(x)
        
        x = self.maxpool5(x)
        # Don't concatenate c6 because the tensor shapes don't line up
        x = self.conv6(x)
        x = self.prelu6(x)
        x = self.flatten(x)
        x = tf.concat([x, h], axis=-1)
        x = self.dense(x)
        x = self.out(x)
        return x
        
        
class DomainAdversarialImageClassifier(tf.keras.Model):
    '''
    Domain adversarial 2D image classifier which learns the classification 
    task while competing with an adversary, which learns to predict cluster 
    membership from the classifier's layer activations.
    '''
    
    def __init__(self, n_clusters, name='da_classifier', **kwargs):
        """Domain adversarial 2D image classifier which learns the classification 
        task while competing with an adversary, which learns to predict cluster 
        membership from the classifier's layer activations.

        Args:
            n_clusters (int): number of clusters
            name (str, optional): Model name. Defaults to 'da_classifier'.
        """        
        super(DomainAdversarialImageClassifier, self).__init__(name=name, **kwargs)
        self.classifier = ImageClassifier()
        self.adversary = AdversarialClassifier(n_clusters)
        self.n_clusters = n_clusters
        
    def call(self, inputs):
        x, z, = inputs
        y_pred = self.classifier(x)
        
        return y_pred
    
    def compile(self,
                loss_classifier=tf.keras.losses.BinaryCrossentropy(),
                loss_adversary=tf.keras.losses.CategoricalCrossentropy(),
                loss_classifier_weight=1.0,
                loss_gen_weight=0.1,
                metric_classifier=tf.keras.metrics.AUC(),
                opt_classifier=tf.keras.optimizers.Nadam(lr=0.0001),
                opt_adversary=tf.keras.optimizers.Nadam(lr=0.0001)
                ):
        """Compile model. Must be called before training or loading weights.

        Args:
            loss_classifier (loss, optional): Main classification loss. 
                Defaults to tf.keras.losses.BinaryCrossentropy().
            loss_adversary (loss, optional): Adversary classification loss. 
                Defaults to tf.keras.losses.CategoricalCrossentropy().
            loss_classifier_weight (float, optional): Main classification loss weight. 
                Defaults to 1.0.
            loss_gen_weight (float, optional): Generalization loss weight. 
                Defaults to 0.01.
            metric_classifier (metric, optional): Main classification metric. 
                Defaults to tf.keras.metrics.AUC().
            opt_classifier (optimizer, optional): Optimizer for main classifier. 
                Defaults to tf.keras.optimizers.Nadam(lr=0.0001).
            opt_adversary (optimizer, optional): Optimizer for adversary. 
                Defaults to tf.keras.optimizers.Nadam(lr=0.0001).
        """        
        super().compile()
        
        self.loss_classifier = loss_classifier
        self.loss_adversary = loss_adversary
        self.loss_classifier_weight = loss_classifier_weight
        self.loss_gen_weight = loss_gen_weight
        self.metric_classifier = metric_classifier
        self.opt_classifier = opt_classifier
        self.opt_adversary = opt_adversary
        
        self.loss_classifier_tracker = tf.keras.metrics.Mean(name='class_loss')
        self.loss_gen_tracker = tf.keras.metrics.Mean(name='gen_loss')
        self.loss_adversary_tracker = tf.keras.metrics.Mean(name='adv_loss')
        self.loss_total_tracker = tf.keras.metrics.Mean(name='total_loss')
        
    @property
    def metrics(self):
        return [self.loss_classifier_tracker,
                self.loss_gen_tracker,
                self.loss_adversary_tracker,
                self.loss_total_tracker,
                self.metric_classifier]
        
    def _compute_update_loss(self, loss_class, loss_gen):
        '''Compute total loss and update loss running means'''
        self.loss_classifier_tracker.update_state(loss_class)
        self.loss_gen_tracker.update_state(loss_gen)
        
        loss_total = (self.loss_classifier_weight * loss_class) + (self.loss_gen_weight * loss_gen)
        self.loss_total_tracker.update_state(loss_total)
        
        return loss_total
    
    def train_step(self, data):
        if len(data) == 3:
            (images, clusters), labels, sample_weights = data
        else:
            (images, clusters), labels = data
            sample_weights = None
            
        # train adversary
        with tf.GradientTape() as gt:
            clf_outs = self.classifier(images, return_layer_activations=True)
            layer_activations = clf_outs[:-1]
            clusters_pred = self.adversary(layer_activations)
            loss_adv = self.loss_adversary(clusters, clusters_pred, sample_weight=sample_weights)

        grads_adv = gt.gradient(loss_adv, self.adversary.trainable_weights)
        self.opt_adversary.apply_gradients(zip(grads_adv, self.adversary.trainable_weights))
        self.loss_adversary_tracker.update_state(loss_adv)
        
        # train main classifier
        with tf.GradientTape() as gt2:
            clf_outs = self.classifier(images, return_layer_activations=True)
            y_pred = clf_outs[-1]
            loss_class = self.loss_classifier(tf.reshape(labels, (-1, 1)), 
                                              tf.reshape(y_pred, (-1, 1)), 
                                              sample_weight=sample_weights)
                        
            layer_activations = clf_outs[:-1]
            clusters_pred = self.adversary(layer_activations)
            loss_gen = self.loss_adversary(clusters, clusters_pred, sample_weight=sample_weights)
            
            loss_total = self._compute_update_loss(loss_class, loss_gen)
            
        grads = gt2.gradient(loss_total, self.classifier.trainable_weights)
        self.opt_classifier.apply_gradients(zip(grads, self.classifier.trainable_weights))
        
        self.metric_classifier.update_state(labels, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        (images, clusters), labels = data
        clf_outs = self.classifier(images, return_layer_activations=True)
        y_pred = clf_outs[-1]
        loss_class = self.loss_classifier(labels, y_pred)
        
        layer_activations = clf_outs[:-1]
        clusters_pred = self.adversary(layer_activations)
        loss_gen = self.loss_adversary(clusters, clusters_pred)
            
        _ = self._compute_update_loss(loss_class, loss_gen)
        
        self.metric_classifier.update_state(labels, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
class RandomEffectsClassifier(ImageClassifier):
    
    def __init__(self, 
                 slope_post_init_scale=0.1,
                 intercept_post_init_scale=0.1,
                 slope_prior_scale=0.25,
                 intercept_prior_scale=0.25,
                 kl_weight=1e-3,
                 name='re_classifier',
                 **kwargs):
        """2D CNN binary classifier with random effect slopes and intercept.

        Args:
            slope_post_init_scale (float, optional): S.d. for random normal initialization 
                of random slopes. Defaults to 0.1.
            intercept_post_init_scale (float, optional): S.d. for random normal 
                initialization of random intercepts. Defaults to 0.1.
            slope_prior_scale (float, optional): S.d. of prior distribution for random 
                slopes. Defaults to 0.25.
            intercept_prior_scale (float, optional): S.d. of prior distribution for random 
                intercepts. Defaults to 0.25.
            kl_weight ([type], optional): KL divergence weight. Defaults to 1e-3.
            name (str, optional): Model name. Defaults to 're_classifier'.
        """        
        super(RandomEffectsClassifier, self).__init__(name=name, **kwargs)
        
        # Single slope and intercept RE layer
        self.re_slopes = RandomEffects(units=512,
                                       post_loc_init_scale=slope_post_init_scale,
                                       prior_scale=slope_prior_scale,
                                       kl_weight=kl_weight,
                                       name='re_slopes')
        
        self.re_intercept = RandomEffects(units=1,
                                       post_loc_init_scale=intercept_post_init_scale,
                                       prior_scale=intercept_prior_scale,
                                       kl_weight=kl_weight,
                                       name='re_intercept')
        
    def call(self, inputs, return_layer_activations=False):
        x, z, = inputs
        c0 = self.conv0(x)
        c0 = self.bn0(c0)
        c0 = self.prelu0(c0)
        
        c1 = self.maxpool0(c0)
        c1 = self.conv1(c1)
        c1 = self.dropout1(c1)
        c1 = self.bn1(c1)
        c1 = self.prelu1(c1)
        
        c2 = self.maxpool1(c1)
        c2 = self.conv2(c2)
        c2 = self.bn2(c2)
        c2 = self.prelu2(c2)
        
        c3 = self.maxpool2(c2)
        c3 = self.conv3(c3)
        c3 = self.dropout3(c3)
        c3 = self.bn3(c3)
        c3 = self.prelu3(c3)
        
        c4 = self.maxpool3(c3)
        c4 = self.conv4(c4)
        c4 = self.bn4(c4)
        c4 = self.prelu4(c4)
        
        c5 = self.maxpool4(c4)
        c5 = self.conv5(c5)
        c5 = self.dropout5(c5)
        c5 = self.bn5(c5)
        c5 = self.prelu5(c5)
        
        c6 = self.maxpool5(c5)
        c6 = self.conv6(c6)
        c6 = self.prelu6(c6)
        h = self.flatten(c6)
        h = self.dense(h)
        slopes = self.re_slopes(z)
        intercepts = self.re_intercept(z)
        y = self.out(h * slopes)
        y = self.sigmoid(y + intercepts)
        
        if return_layer_activations:
            return c0, c1, c2, c3, c4, c5, c6, h, y
        else:
            return y
        
    
class MixedEffectsClassifier(DomainAdversarialImageClassifier):
    ''' Random linear slope and intercept introduced before last dense layer
    '''
    def __init__(self, 
                 n_clusters,
                 slope_post_init_scale=0.1,
                 intercept_post_init_scale=0.1,
                 slope_prior_scale=0.25,
                 intercept_prior_scale=0.25,
                 kl_weight=1e-3,
                 name='me_classifier', **kwargs):
        """2D CNN binary classifier using domain adversarial training to enforce the learning 
        of cluster-agnostic convolutional features. Additionally, cluster-specific random 
        slopes and intercepts are learned in the dense layers of the model. 

        Args:
            slope_post_init_scale (float, optional): S.d. for random normal initialization 
                of random slopes. Defaults to 0.1.
            intercept_post_init_scale (float, optional): S.d. for random normal 
                initialization of random intercepts. Defaults to 0.1.
            slope_prior_scale (float, optional): S.d. of prior distribution for random 
                slopes. Defaults to 0.25.
            intercept_prior_scale (float, optional): S.d. of prior distribution for random 
                intercepts. Defaults to 0.25.
            kl_weight ([type], optional): KL divergence weight. Defaults to 1e-3.
            name (str, optional): Model name. Defaults to 'me_classifier'.
        """      
      
        super(MixedEffectsClassifier, self).__init__(n_clusters, name=name, **kwargs)
                
        # Single slope and intercept RE layer
        self.re_slopes = RandomEffects(units=512,
                                       post_loc_init_scale=slope_post_init_scale,
                                       prior_scale=slope_prior_scale,
                                       kl_weight=kl_weight,
                                       name='re_slopes')
        
        self.re_intercept = RandomEffects(units=1,
                                       post_loc_init_scale=intercept_post_init_scale,
                                       prior_scale=intercept_prior_scale,
                                       kl_weight=kl_weight,
                                       name='re_intercept')
        
    def call(self, inputs, return_layer_activations=False):
        x, z = inputs
        c0 = self.classifier.conv0(x)
        c0 = self.classifier.bn0(c0)
        c0 = self.classifier.prelu0(c0)
        
        c1 = self.classifier.maxpool0(c0)
        c1 = self.classifier.conv1(c1)
        c1 = self.classifier.dropout1(c1)
        c1 = self.classifier.bn1(c1)
        c1 = self.classifier.prelu1(c1)
        
        c2 = self.classifier.maxpool1(c1)
        c2 = self.classifier.conv2(c2)
        c2 = self.classifier.bn2(c2)
        c2 = self.classifier.prelu2(c2)
        
        c3 = self.classifier.maxpool2(c2)
        c3 = self.classifier.conv3(c3)
        c3 = self.classifier.dropout3(c3)
        c3 = self.classifier.bn3(c3)
        c3 = self.classifier.prelu3(c3)
        
        c4 = self.classifier.maxpool3(c3)
        c4 = self.classifier.conv4(c4)
        c4 = self.classifier.bn4(c4)
        c4 = self.classifier.prelu4(c4)
        
        c5 = self.classifier.maxpool4(c4)
        c5 = self.classifier.conv5(c5)
        c5 = self.classifier.dropout5(c5)
        c5 = self.classifier.bn5(c5)
        c5 = self.classifier.prelu5(c5)
        
        c6 = self.classifier.maxpool5(c5)
        c6 = self.classifier.conv6(c6)
        c6 = self.classifier.prelu6(c6)
        h = self.classifier.flatten(c6)
        h = self.classifier.dense(h)
        slopes = self.re_slopes(z)
        intercepts = self.re_intercept(z)
        y = self.classifier.out(h * (1 + slopes))
        y = self.classifier.sigmoid(y + intercepts)
        
        if return_layer_activations:
            return c0, c1, c2, c3, c4, c5, c6, h, y
        else:
            return y
    
    def compile(self,
            loss_classifier=tf.keras.losses.BinaryCrossentropy(),
            loss_adversary=tf.keras.losses.CategoricalCrossentropy(),
            loss_classifier_weight=1.0,
            loss_gen_weight=0.1,
            metric_classifier=tf.keras.metrics.AUC(),
            opt_classifier=tf.keras.optimizers.Nadam(lr=0.0001),
            opt_adversary=tf.keras.optimizers.Nadam(lr=0.0001)
            ):
        """Compile model. Must be called before training or loading weights.

        Args:
            loss_classifier (loss, optional): Main classification loss. 
                Defaults to tf.keras.losses.BinaryCrossentropy().
            loss_adversary (loss, optional): Adversary classification loss. 
                Defaults to tf.keras.losses.CategoricalCrossentropy().
            loss_classifier_weight (float, optional): Main classification loss weight. 
                Defaults to 1.0.
            loss_gen_weight (float, optional): Generalization loss weight. 
                Defaults to 0.01.
            metric_classifier (metric, optional): Main classification metric. 
                Defaults to tf.keras.metrics.AUC().
            opt_classifier (optimizer, optional): Optimizer for main classifier. 
                Defaults to tf.keras.optimizers.Nadam(lr=0.0001).
            opt_adversary (optimizer, optional): Optimizer for adversary. 
                Defaults to tf.keras.optimizers.Nadam(lr=0.0001).
        """  
        
        super(MixedEffectsClassifier, self).compile(loss_classifier=loss_classifier,
                                                    loss_adversary=loss_adversary,
                                                    loss_classifier_weight=loss_classifier_weight,
                                                    loss_gen_weight=loss_gen_weight,
                                                    metric_classifier=metric_classifier,
                                                    opt_classifier=opt_classifier,
                                                    opt_adversary=opt_adversary)
        self.loss_kld_tracker = tf.keras.metrics.Mean(name='kld')
        
    @property
    def metrics(self):
        return [self.loss_classifier_tracker,
                self.loss_gen_tracker,
                self.loss_adversary_tracker,
                self.loss_kld_tracker,
                self.loss_total_tracker,
                self.metric_classifier]
        
    def _compute_update_loss(self, loss_class, loss_gen, training=True):
        '''Compute total loss and update loss running means'''
        self.loss_classifier_tracker.update_state(loss_class)
        self.loss_gen_tracker.update_state(loss_gen)
        if training:
            kld = tf.reduce_sum(self.re_slopes.losses) + tf.reduce_sum(self.re_intercept.losses)
            self.loss_kld_tracker.update_state(kld)
        else:
            # KLD can't be computed at inference time because posteriors are simplified to 
            # point estimates
            kld = 0
        
        loss_total = (self.loss_classifier_weight * loss_class) \
            + (self.loss_gen_weight * loss_gen) \
            + kld
        self.loss_total_tracker.update_state(loss_total)
        
        return loss_total
        
    
    def train_step(self, data):
        if len(data) == 3:
            (images, clusters), labels, sample_weights = data
        else:
            (images, clusters), labels = data
            sample_weights = None
            
        # train adversary
        with tf.GradientTape() as gt:
            clf_outs = self((images, clusters), return_layer_activations=True)
            layer_activations = clf_outs[:-1]
            clusters_pred = self.adversary(layer_activations)
            loss_adv = self.loss_adversary(clusters, clusters_pred, sample_weight=sample_weights)

        grads_adv = gt.gradient(loss_adv, self.adversary.trainable_weights)
        self.opt_adversary.apply_gradients(zip(grads_adv, self.adversary.trainable_weights))
        self.loss_adversary_tracker.update_state(loss_adv)
        
        # train main classifier
        with tf.GradientTape() as gt2:
            clf_outs = self((images, clusters), return_layer_activations=True)
            y_pred = clf_outs[-1]
            loss_class = self.loss_classifier(tf.reshape(labels, (-1, 1)), 
                                    tf.reshape(y_pred, (-1, 1)), 
                                    sample_weight=sample_weights)
                        
            layer_activations = clf_outs[:-1]
            clusters_pred = self.adversary(layer_activations)
            loss_gen = self.loss_adversary(clusters, clusters_pred, sample_weight=sample_weights)
            
            loss_total = self._compute_update_loss(loss_class, loss_gen)
            
        grads = gt2.gradient(loss_total, self.classifier.trainable_weights)
        self.opt_classifier.apply_gradients(zip(grads, self.classifier.trainable_weights))
        
        self.metric_classifier.update_state(labels, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        (images, clusters), labels = data
        clf_outs = self((images, clusters), return_layer_activations=True)
        y_pred = clf_outs[-1]
        loss_class = self.loss_classifier(labels, y_pred)
        
        layer_activations = clf_outs[:-1]
        clusters_pred = self.adversary(layer_activations)
        loss_gen = self.loss_adversary(clusters, clusters_pred)
            
        _ = self._compute_update_loss(loss_class, loss_gen)
        
        self.metric_classifier.update_state(labels, y_pred)
        return {m.name: m.result() for m in self.metrics}

