'''
Autoencoder-classifiers, including domain adversarial and mixed effects variations.
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tkl

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec

from .random_effects import ClusterScaleBiasBlock, RandomEffects

class TiedConv2DTranspose(tkl.Conv2DTranspose):
    def __init__(self, 
                 source_layer: tkl.Conv2D,
                 filters, 
                 kernel_size, 
                 strides=(1, 1), 
                 padding='valid', 
                 output_padding=None,
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """Conv2DTranspose layer that shares weights with a given Conv2D layer.
        
        (The bias tensor is not shared as the dimensionality of the output is
        inherently different.)

        Args: 
            source_layer (Conv2D): Conv2D layer with which to share weights
            all other arguments same as original Conv2DTranspose
            
        """        
        self.source_layer = source_layer
                
        super().__init__(filters,
                         kernel_size,
                         strides=strides, 
                         padding=padding, 
                         output_padding=output_padding, 
                         data_format=data_format, 
                         dilation_rate=dilation_rate, 
                         activation=activation, 
                         use_bias=use_bias, 
                         kernel_initializer=kernel_initializer, 
                         bias_initializer=bias_initializer, 
                         kernel_regularizer=kernel_regularizer, 
                         bias_regularizer=bias_regularizer, 
                         activity_regularizer=activity_regularizer, 
                         kernel_constraint=kernel_constraint, 
                         bias_constraint=bias_constraint, 
                         **kwargs)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank 4. Received input '
                            'shape: ' + str(input_shape))
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                            'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        # kernel_shape = self.kernel_size + (self.filters, input_dim)

        # Link to weights from the source conv layer
        self.kernel = self.source_layer.weights[0]
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        else:
            self.bias = None
        self.built = True


class Encoder(tkl.Layer):
        
    def __init__(self, 
                 n_latent_dims: int=56, 
                 layer_filters: list=[64, 64, 64, 128, 256, 512],
                 return_layer_activations: bool=False,
                 name='encoder', 
                 **kwargs):
        """Transforms 2D image into a compressed vector representation. Contains
        6x 2D strided convolutional layers.

        Args: 
            n_latent_dims (int, optional): Size of compressed representation
                output. Defaults to 56. 
            layer_filters (list, optional): Filters per convolutional layer. 
                Defaults to [64, 128, 256, 512, 1024, 1024].
            return_layer_activations (bool, optional): Whether to return every 
                layer's output. Defaults to False.
            name (str, optional): Name. Defaults to 'encoder'.
        """        
        super(Encoder, self).__init__(name=name, **kwargs)
        
        self.n_latent_dims = n_latent_dims
        self.layer_filters = layer_filters
        self.return_layer_activations = return_layer_activations
        
        self.conv0 = tkl.Conv2D(layer_filters[0], 4, padding='same', name=name + '_conv0')
        self.bn0 = tkl.BatchNormalization(name=name+ '_bn0')
        self.prelu0 = tkl.PReLU(name=name + '_prelu0')
        
        self.conv1 = tkl.Conv2D(layer_filters[1], 4, padding='same', name=name + '_conv1')
        self.bn1 = tkl.BatchNormalization(name=name+ '_bn1')
        self.prelu1 = tkl.PReLU(name=name + '_prelu1')
        
        self.conv2 = tkl.Conv2D(layer_filters[2], 4, padding='same', name=name + '_conv2')
        self.bn2 = tkl.BatchNormalization(name=name+ '_bn2')
        self.prelu2 = tkl.PReLU(name=name + '_prelu2')
        
        self.conv3 = tkl.Conv2D(layer_filters[3], 4, strides=(2, 2), padding='same', name=name + '_conv3')
        self.bn3 = tkl.BatchNormalization(name=name+ '_bn3')
        self.prelu3 = tkl.PReLU(name=name + '_prelu3')
        
        self.conv4 = tkl.Conv2D(layer_filters[4], 4, strides=(2, 2), padding='same', name=name + '_conv4')
        self.bn4 = tkl.BatchNormalization(name=name+ '_bn4')
        self.prelu4 = tkl.PReLU(name=name + '_prelu4')
        
        self.conv5 = tkl.Conv2D(layer_filters[5], 4, strides=(2, 2), padding='same', name=name + '_conv5')
        self.bn5 = tkl.BatchNormalization(name=name+ '_bn5')
        self.prelu5 = tkl.PReLU(name=name + '_prelu5')
               
        self.flatten = tkl.Flatten(name=name + '_flatten')
        self.dense = tkl.Dense(n_latent_dims, name=name + '_latent')
        self.bn_out = tkl.BatchNormalization(name=name + '_output')
        
    def call(self, inputs, training=None):
        x0 = self.conv0(inputs)
        x0 = self.bn0(x0, training=training)
        x0 = self.prelu0(x0)
        
        x1 = self.conv1(x0)
        x1 = self.bn1(x1, training=training)
        x1 = self.prelu1(x1)
        
        x2 = self.conv2(x1)
        x2 = self.bn2(x2, training=training)
        x2 = self.prelu2(x2)
        
        x3 = self.conv3(x2)
        x3 = self.bn3(x3, training=training)
        x3 = self.prelu3(x3)
        
        x4 = self.conv4(x3)
        x4 = self.bn4(x4, training=training)
        x4 = self.prelu4(x4)
        
        x5 = self.conv5(x4)
        x5 = self.bn5(x5, training=training)
        x5 = self.prelu5(x5)

        latent = self.flatten(x5)
        latent = self.dense(latent)
        latent = self.bn_out(latent)
        
        if self.return_layer_activations:
            return x0, x1, x2, x3, x4, x5, latent
        else:
            return latent
        
    def get_config(self):
        return {'n_latent_dims': self.n_latent_dims,
                'layer_filters': self.layer_filters,
                'return_layer_activations': self.return_layer_activations}

class Decoder(tkl.Layer):
    
    def __init__(self, 
                 image_shape: tuple=(256, 256, 1),
                 layer_filters: list=[512, 256, 128, 64, 64, 64],
                 name='decoder', 
                 **kwargs):
        """Transforms compressed vector representation back into a 2D image.
        Contains 6x 2D transposed convolutional layers.

        Args: 
            image_shape (tuple, optional): Output image shape. Defaults to 
                (256, 256, 1). 
            layer_filters (list, optional): Number of filters in each 
                convolutional layer. This should be the reverse of the 
                layer_filters argument given to the encoder. 
                Defaults to [1024, 1024, 512, 256, 128, 64]. 
            name (str, optional): Name. Defaults to 'decoder'.
        """        
        super(Decoder, self).__init__(name=name, **kwargs)
        
        self.image_shape = image_shape
        self.layer_filters = layer_filters
        
        tupReshape = (image_shape[0] // 8, image_shape[1] // 8, layer_filters[0])
        
        self.dense = tkl.Dense(np.product(tupReshape), name=name + '_dense')
        self.reshape = tkl.Reshape(tupReshape, name=name + '_reshape')
        self.prelu_dense = tkl.PReLU(name=name + '_prelu_dense')
                
        self.tconv0 = tkl.Conv2DTranspose(layer_filters[1], 4, 
                                          strides=(2, 2), padding='same', name=name + '_tconv0')
        self.bn0 = tkl.BatchNormalization(name=name+ '_bn0')
        self.prelu0 = tkl.PReLU(name=name + '_prelu0')
        
        self.tconv1 = tkl.Conv2DTranspose(layer_filters[2], 4, 
                                          strides=(2, 2), padding='same', name=name + '_tconv1')
        self.bn1 = tkl.BatchNormalization(name=name+ '_bn1')
        self.prelu1 = tkl.PReLU(name=name + '_prelu1')
        
        self.tconv2 = tkl.Conv2DTranspose(layer_filters[3], 4, 
                                          strides=(2, 2), padding='same', name=name + '_tconv2')
        self.bn2 = tkl.BatchNormalization(name=name+ '_bn2')
        self.prelu2 = tkl.PReLU(name=name + '_prelu2')
        
        self.tconv3 = tkl.Conv2DTranspose(layer_filters[4], 4, 
                                          padding='same', name=name + '_tconv3')
        self.bn3 = tkl.BatchNormalization(name=name+ '_bn3')
        self.prelu3 = tkl.PReLU(name=name + '_prelu3')
        
        self.tconv4 = tkl.Conv2DTranspose(layer_filters[5], 4, 
                                          padding='same', name=name + '_tconv4')
        self.bn4 = tkl.BatchNormalization(name=name+ '_bn4')
        self.prelu4 = tkl.PReLU(name=name + '_prelu4')
        
        self.tconv5 = tkl.Conv2DTranspose(1, 4, padding='same', name=name + '_tconv5')
        self.bn5 = tkl.BatchNormalization(name=name+ '_bn5')
        self.sigmoid_out = tkl.Activation('sigmoid', name=name + '_sigmoid')
                
    def call(self, inputs, training=None):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.prelu_dense(x)
        
        x = self.tconv0(x)
        x = self.bn0(x, training=training)
        x = self.prelu0(x)
        
        x = self.tconv1(x)
        x = self.bn1(x, training=training)
        x = self.prelu1(x)
        
        x = self.tconv2(x)
        x = self.bn2(x, training=training)
        x = self.prelu2(x)
        
        x = self.tconv3(x)
        x = self.bn3(x, training=training)
        x = self.prelu3(x)
        
        x = self.tconv4(x)
        x = self.bn4(x, training=training)
        x = self.prelu4(x)
        
        x = self.tconv5(x)
        x = self.bn5(x, training=training)
        x = self.sigmoid_out(x)
        
        return x
        
    def get_config(self):
        return {'image_shape': self.image_shape,
                'layer_filters': self.layer_filters}
        
class TiedDecoder(Decoder):
    
    def __init__(self, 
                 encoder_layers: list,
                 image_shape: tuple=(256, 256, 1),
                 layer_filters: list=[512, 256, 128, 64, 64, 64],
                 name='decoder', 
                 **kwargs):
        """Transforms compressed vector representation back into a 2D image.
        Contains 6x 2D transposed convolutional layers, and filter weights are
        tied to a given encoder. 

        Args: 
            encoder_layers (list): List of encoder layers whose weights will
                be shared with this decoder.
            image_shape (tuple, optional): Output image shape. Defaults to 
                (256, 256, 1). 
            layer_filters (list, optional): Number of filters in each 
                convolutional layer. This should be the reverse of the 
                layer_filters argument given to the encoder. 
                Defaults to [1024, 1024, 512, 256, 128, 64]. 
            name (str, optional): Name. Defaults to 'decoder'.
        """        
        super(TiedDecoder, self).__init__(image_shape=image_shape, layer_filters=layer_filters,
                                          name=name, **kwargs)
                        
        # Replace conventional Conv2DTranspose layers with ones that share weights                        
        self.tconv0 = TiedConv2DTranspose(encoder_layers[-1], layer_filters[1], 4, 
                                          strides=(2, 2), padding='same', name=name + '_tconv0')
        
        self.tconv1 = TiedConv2DTranspose(encoder_layers[-2], layer_filters[2], 4, 
                                          strides=(2, 2), padding='same', name=name + '_tconv1')
        
        self.tconv2 = TiedConv2DTranspose(encoder_layers[-3], layer_filters[3], 4, 
                                          strides=(2, 2), padding='same', name=name + '_tconv2')

        self.tconv3 = TiedConv2DTranspose(encoder_layers[-4], layer_filters[4], 4, 
                                          padding='same', name=name + '_tconv3')

        self.tconv4 = TiedConv2DTranspose(encoder_layers[-5], layer_filters[5], 4, 
                                          padding='same', name=name + '_tconv4')
        
        self.tconv5 = TiedConv2DTranspose(encoder_layers[-6], 1, 4, padding='same', name=name + '_tconv5')
                
class AuxClassifier(tkl.Layer):
           
    def __init__(self, 
                 units: int=32,
                 name='auxclassifier', 
                 **kwargs):
        """Simple dense binary classifier with one hidden layer and 
        sigmoid output. Intended to be attached to the autoencoder to 
        perform classification.

        Args:
            units (int, optional): Number of hidden layer neurons. 
                Defaults to 32.
            name (str, optional): Name. Defaults to 'auxclassifier'.
        """        
        super(AuxClassifier, self).__init__(name=name, **kwargs)
        
        self.units = units
        
        self.hidden = tkl.Dense(units, name=name + '_dense')
        self.activation = tkl.LeakyReLU(name=name + '_leakyrelu')
        self.dense_out = tkl.Dense(1, activation='sigmoid', name=name + '_output')
        
    def call(self, inputs):
        x = self.hidden(inputs)
        x = self.activation(x)
        x = self.dense_out(x)
        return x
    
    def get_config(self):
        return {'units': self.units}
        
        
class BaseAutoencoderClassifier(tf.keras.Model):
        
    def __init__(self,
                 image_shape: tuple=(256, 256, 1),
                 n_latent_dims: int=56, 
                 encoder_layer_filters: list=[64, 64, 64, 128, 256, 512],
                 classifier_hidden_units: int=32,
                 name='autoencoder',
                 **kwargs):
        """Basic autoencoder with auxiliary classifier to predict a binary 
        label from the latent representation.

        Args:
            image_shape (tuple, optional): Input image shape. Defaults to 
                (256, 256, 1).
            n_latent_dims (int, optional): Size of latent representation. 
                Defaults to 56.
            encoder_layer_filters (list, optional): Number of filters per 
                encoder layer. Defaults to [64, 128, 256, 512, 1024, 1024].
            classifier_hidden_units (int, optional): Number of hidden layer 
                neurons in the auxiliary classifier. Defaults to 32.
            name (str, optional): Name. Defaults to 'autoencoder'.
        """        
        
        super(BaseAutoencoderClassifier, self).__init__(name=name, **kwargs)
        
        self.image_shape = image_shape
        self.n_latent_dims = n_latent_dims
        self.encoder_layer_filters = encoder_layer_filters
        self.decoder_layer_filters = encoder_layer_filters[-1::-1]
        self.classifier_hidden_units = classifier_hidden_units
                
        self.encoder = Encoder(n_latent_dims=n_latent_dims,
                               layer_filters=encoder_layer_filters,
                               return_layer_activations=False)
        
        lsEncoderLayers = [self.encoder.conv0,
                    self.encoder.conv1,
                    self.encoder.conv2,
                    self.encoder.conv3, 
                    self.encoder.conv4,
                    self.encoder.conv5]
        
        self.decoder = TiedDecoder(lsEncoderLayers, 
                                   image_shape=image_shape, 
                                   layer_filters=self.decoder_layer_filters)
        
        self.classifier = AuxClassifier(units=classifier_hidden_units)
        
    def call(self, inputs, training=None):
        
        latent = self.encoder(inputs, training=training)
        recon = self.decoder(latent, training=training)
        classification = self.classifier(latent)
        
        return recon, classification
    
class AdversarialClassifier(tkl.Layer):
    
    def __init__(self, 
                 image_shape: tuple, 
                 n_clusters: int, 
                 layer_filters: list=[16, 32, 32, 64, 64, 128, 128, 256],
                 dense_units: int=512,
                 name='adversary', 
                 **kwargs):
        """Domain adversarial classifier for predicting a sample's cluster 
        from the layer outputs of the Encoder.

        Args:
            image_shape (tuple): Original image shape.
            n_clusters (int): Number of possible clusters (domains), i.e. 
                the size of the softmax output.
            layer_filters (list, optional): Number of filters in 
                each adversary layer. Defaults to [16, 32, 64, 128, 256, 512].
            dense_units (int, optional): Number of neurons in 
                adversary dense layer. Defaults to 512.
            name (str, optional): Name. Defaults to 'adversary'.
        """        
        
        super(AdversarialClassifier, self).__init__(name=name, **kwargs)
        
        if image_shape[:2] != (256, 256):
            raise ValueError('Only 256x256 images are supported at this time.')
        
        self.image_shape = image_shape
        self.n_clusters = n_clusters
        self.layer_filters = layer_filters
        self.dense_units = dense_units
                
        self.conv0 = tkl.Conv2D(layer_filters[0], 4, strides=(2, 2), padding='same', name=name + '_conv0')
        self.bn0 = tkl.BatchNormalization(name=name + '_bn0')
        self.prelu0 = tkl.PReLU(name=name + '_prelu0')
        
        self.concat1 = tkl.Concatenate(axis=-1, name=name + '_concat1')
        self.conv1 = tkl.Conv2D(layer_filters[1], 4, strides=(2, 2), padding='same', name=name + '_conv1')
        self.bn1 = tkl.BatchNormalization(name=name + '_bn1')
        self.prelu1 = tkl.PReLU(name=name + '_prelu1')
        
        self.concat2 = tkl.Concatenate(axis=-1, name=name + '_concat2')
        self.conv2 = tkl.Conv2D(layer_filters[2], 4, strides=(2, 2), padding='same', name=name + '_conv2')
        self.bn2 = tkl.BatchNormalization(name=name + '_bn2')
        self.prelu2 = tkl.PReLU(name=name + '_prelu2')
        
        self.concat3 = tkl.Concatenate(axis=-1, name=name + '_concat3')
        self.conv3 = tkl.Conv2D(layer_filters[3], 4, strides=(2, 2), padding='same', name=name + '_conv3')
        self.bn3 = tkl.BatchNormalization(name=name + '_bn3')
        self.prelu3 = tkl.PReLU(name=name + '_prelu3')
        
        self.conv4 = tkl.Conv2D(layer_filters[4], 4, strides=(2, 2), padding='same', name=name + '_conv4')
        self.bn4 = tkl.BatchNormalization(name=name + '_bn4')
        self.prelu4 = tkl.PReLU(name=name + '_prelu4')
        
        self.conv5 = tkl.Conv2D(layer_filters[5], 4, strides=(2, 2), padding='same', name=name + '_conv5')
        self.bn5 = tkl.BatchNormalization(name=name + '_bn5')
        self.prelu5 = tkl.PReLU(name=name + '_prelu5')
                
        self.flatten = tkl.Flatten(name=name + '_flatten')
        self.dense = tkl.Dense(units=self.dense_units, name=name + '_dense')
        self.softmax = tkl.Dense(units=n_clusters, activation='softmax', name=name + '_softmax')
        
    def call(self, inputs):
        act0, act1, act2, act3, act4, act5, latents = inputs
        
        # Layer outputs from the encoder are fed into this model at the point
        # where their shape matches the data shape.
        
        # First 3 encoder layers don't use strided conv and have the same output
        # shape as the original image. Concatenate them and feed them into the
        # first layer at the same time.
        x = tf.concat([act0, act1, act2], axis=-1) # -> D x D
        
        x = self.conv0(x) # -> D/2 x D/2
        x = self.bn0(x)
        x = self.prelu0(x)
        
        x = self.concat1([x, act3])
        x = self.conv1(x) # -> D/4 x D/4
        x = self.bn1(x)
        x = self.prelu1(x)
        
        x = self.concat2([x, act4])
        x = self.conv2(x) # -> D/8 x D/8
        x = self.bn2(x)
        x = self.prelu2(x)
        
        x = self.concat3([x, act5])
        x = self.conv3(x) # -> D/16 x D/16
        x = self.bn3(x)
        x = self.prelu3(x)
        
        x = self.conv4(x) # -> D/32 x D/32
        x = self.bn4(x)
        x = self.prelu4(x)
        
        x = self.conv5(x) # -> D/64 x D/64
        x = self.bn5(x)
        x = self.prelu5(x)
        
        x = self.flatten(x)
        x = tf.concat([x, latents], axis=-1)
        x = self.dense(x)
        x = self.softmax(x)
        return x        
        
    def get_config(self):
        return {'image_shape': self.image_shape,
                'n_clusters': self.n_clusters,
                'layer_filters': self.layer_filters,
                'dense_units': self.dense_units}
        
        
class DomainAdversarialAEC(BaseAutoencoderClassifier):
    
    def __init__(self,
                 image_shape: tuple=(256, 256, 1),
                 n_clusters: int=10,
                 n_latent_dims: int=56, 
                 encoder_layer_filters: list=[64, 64, 64, 128, 256, 512],
                 classifier_hidden_units: int=32,
                 adversary_layer_filters: list=[16, 32, 32, 64, 64, 128, 128, 256],
                 name='autoencoder',
                 **kwargs
                 ):
        """Domain adversarial autoencoder-classifier. Adds an adversarial 
        classifier to predict the cluster/domain membership of each sample 
        based on the encoder's intermediate outputs. This compels the 
        encoder to learn features unassociated with cluster characteristics.

        Args:
            image_shape (tuple, optional): Image shape. Defaults to 
                (256, 256, 1).
            n_clusters (int, optional): Number of clusters. Defaults to 10.
            n_latent_dims (int, optional): Size of latent representation. 
                Defaults to 56.
            encoder_layer_filters (list, optional): Number of filters in 
                each encoder layer. Defaults to [64, 64, 64, 128, 256, 512].
            classifier_hidden_units (int, optional): Number of neurons in 
                auxiliary classifier hidden layer. Defaults to 32.
            adversary_layer_filters (list, optional): Number of filters in 
                each adversary layer. Defaults to [16, 32, 32, 64, 64, 128, 128, 256].
            name (str, optional): Model name. Defaults to 'autoencoder'.
        """        
        super(BaseAutoencoderClassifier, self).__init__(
            name=name, 
            **kwargs)

        self.image_shape = image_shape
        self.n_latent_dims = n_latent_dims
        self.encoder_layer_filters = encoder_layer_filters
        self.decoder_layer_filters = encoder_layer_filters[-1::-1]
        self.classifier_hidden_units = classifier_hidden_units
                
        self.encoder = Encoder(n_latent_dims=n_latent_dims,
                               layer_filters=encoder_layer_filters,
                               return_layer_activations=True)
        
        lsEncoderLayers = [self.encoder.conv0,
                    self.encoder.conv1,
                    self.encoder.conv2,
                    self.encoder.conv3, 
                    self.encoder.conv4,
                    self.encoder.conv5]
        
        self.decoder = TiedDecoder(lsEncoderLayers, 
                                   image_shape=image_shape, 
                                   layer_filters=self.decoder_layer_filters)
        
        self.classifier = AuxClassifier(units=classifier_hidden_units)
                
        self.adversary = AdversarialClassifier(image_shape, n_clusters,
                                               layer_filters=adversary_layer_filters)
        
    def call(self, inputs, training=None):
        images, clusters = inputs
        # Call encoder and get layer outputs
        encoder_outs = self.encoder(images, training=training)
        latent = encoder_outs[-1]
        # Reconstruct image from latents
        recon = self.decoder(latent, training=training)
        # Classify image from latents
        classification = self.classifier(latent)
        # Predict cluster from encoder layer outputs
        pred_cluster = self.adversary(encoder_outs)
        
        return (recon, classification, pred_cluster)
    
    def compile(self,
                loss_recon=tf.keras.losses.MeanSquaredError(),
                loss_class=tf.keras.losses.BinaryCrossentropy(),
                loss_adv=tf.keras.losses.CategoricalCrossentropy(),
                metric_class=tf.keras.metrics.AUC(name='auroc'),
                metric_adv=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                opt_autoencoder=tf.keras.optimizers.Adam(lr=0.0001),
                opt_adversary=tf.keras.optimizers.Adam(lr=0.0001),
                loss_recon_weight=1.0,
                loss_class_weight=0.01,
                loss_gen_weight=0.05,
                ):
        """Compile model with given losses, metrics, and optimizers.
        
        The main autoencoder-classifier is trained with this loss function:
        loss_recon_weight * loss_recon      (image reconstruction loss)
        + loss_class_weight * loss_class    (phenotype classification loss)
        - loss_gen_weight * loss_adv        (generalization (adversarial) loss)

        While the adversarial classifier is trained with loss_adv.

        Args:
            loss_recon (loss, optional): Image reconstruction loss. Defaults to
                tf.keras.losses.MeanSquaredError().
            loss_class (loss, optional): Auxiliary classification loss. Defaults
                to tf.keras.losses.BinaryCrossentropy().
            loss_adv (loss, optional): Adversarial classification loss. Defaults 
                to tf.keras.losses.CategoricalCrossentropy().
            metric_class (metric, optional): Auxiliary classification metric. 
                Defaults to tf.keras.metrics.AUC(name='auroc').
            metric_adv (metric, optional): Adversarial classification metric. 
                Defaults to tf.keras.metrics.CategoricalAccuracy(name='acc').
            opt_autoencoder (optimizer, optional): Optimizer for the main model. 
                Defaults to tf.keras.optimizers.Adam(lr=0.0001).
            opt_adversary (optimizer, optional): Optimizer for the adversarial 
                classifier. Defaults to tf.keras.optimizers.Adam(lr=0.0001).
            loss_recon_weight (float, optional): Weight for reconstruction loss. 
                Defaults to 1.0.
            loss_class_weight (float, optional): Weight for auxiliary classification 
                loss. Defaults to 0.01.
            loss_gen_weight (float, optional): Weight for generalization loss. 
                Defaults to 0.05.
        """        
        super().compile()

        self.loss_recon = loss_recon
        self.loss_class = loss_class
        self.loss_adv = loss_adv

        self.opt_autoencoder = opt_autoencoder
        self.opt_adversary = opt_adversary
        
        # Loss trackers to maintain a running mean of each loss
        self.loss_recon_tracker = tf.keras.metrics.Mean(name='recon_loss')
        self.loss_class_tracker = tf.keras.metrics.Mean(name='class_loss')
        self.loss_adv_tracker = tf.keras.metrics.Mean(name='adv_loss')
        self.loss_total_tracker = tf.keras.metrics.Mean(name='total_loss')

        self.metric_class = metric_class
        self.metric_adv = metric_adv
        
        self.loss_recon_weight = loss_recon_weight
        self.loss_class_weight = loss_class_weight
        self.loss_gen_weight = loss_gen_weight      
        
    @property
    def metrics(self):
        return [self.loss_recon_tracker,
                self.loss_class_tracker,
                self.loss_adv_tracker,
                self.loss_total_tracker,
                self.metric_class,
                self.metric_adv]
        
    def train_step(self, data):
        if len(data) == 3:
            (images, clusters), (_, labels), sample_weights = data
        else:
            (images, clusters), (_, labels) = data
            sample_weights = None
            
        # Train adversary
        encoder_outs = self.encoder(images, training=True)
        with tf.GradientTape() as gt:
            pred_cluster = self.adversary(encoder_outs)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            
        grads_adv = gt.gradient(loss_adv, self.adversary.trainable_variables)
        self.opt_adversary.apply_gradients(zip(grads_adv, self.adversary.trainable_variables))
        
        # Update adversarial loss tracker
        self.metric_adv.update_state(clusters, pred_cluster)
        self.loss_adv_tracker.update_state(loss_adv)
        
        # Train autoencoder
        with tf.GradientTape(persistent=True) as gt2:
            pred_recon, pred_class, pred_cluster = self((images, clusters), training=True)
            loss_class = self.loss_class(labels, pred_class, sample_weight=sample_weights)
            loss_recon = self.loss_recon(images, pred_recon, sample_weight=sample_weights)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            
            total_loss = (self.loss_recon_weight * loss_recon) \
                + (self.loss_class_weight * loss_class) \
                - (self.loss_gen_weight * loss_adv)
                
        lsWeights = self.encoder.trainable_variables + self.decoder.trainable_variables \
                + self.classifier.trainable_variables
        grads_aec = gt2.gradient(total_loss, lsWeights)
        self.opt_autoencoder.apply_gradients(zip(grads_aec, lsWeights))
        
        # Update loss trackers
        self.metric_class.update_state(labels, pred_class)
        self.loss_class_tracker.update_state(loss_class)
        self.loss_recon_tracker.update_state(loss_recon)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        (images, clusters), (_, labels) = data
                        
        pred_recon, pred_class, pred_cluster = self((images, clusters), training=False)
        loss_class = self.loss_class(labels, pred_class)
        loss_recon = self.loss_recon(images, pred_recon)
        loss_adv = self.loss_adv(clusters, pred_cluster)
            
        total_loss = (self.loss_recon_weight * loss_recon) \
            + (self.loss_class_weight * loss_class) \
            - (self.loss_gen_weight * loss_adv)
                    
        self.metric_class.update_state(labels, pred_class)
        self.metric_adv.update_state(clusters, pred_cluster)
        
        self.loss_class_tracker.update_state(loss_class)
        self.loss_recon_tracker.update_state(loss_recon)
        self.loss_adv_tracker.update_state(loss_adv)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}
    
          
def load_weights_base_aec(model, weights_path: str):
    """Loads weights into BaseAutoencoderClassifier. When using the native Keras
    model.load_weights(), it fails to match up the saved weights with the model
    layers. This workaround manually matches them up by name.

    Args: 
        model (tf.keras.Model): model
        weights_path (str): path to weights

    """    
    import h5py
    h5Weights = h5py.File(weights_path, 'r')

    for strSubModel in h5Weights.keys():
        submodel = model.get_layer(strSubModel)
        lsWeightsToSet = []
        for weights in submodel.weights:
            strWeightName = weights.name.split('autoencoder/')[1]
            def match_weight(name):
                if strWeightName in name:
                    return name
            # Recursively search for the weight name in the H5 object
            w = h5Weights[strSubModel].visit(match_weight)
            lsWeightsToSet += [h5Weights[strSubModel][w]]
        submodel.set_weights(lsWeightsToSet)
        

class LatentClassifier(tkl.Layer):
    def __init__(self, 
                 n_clusters: int,
                 name='latent_classifier', 
                 **kwargs):
        """Dense classifier to predict cluster from a vector 
            latent representation.

        Args:
            n_clusters (int): Number of clusters.
            name (str, optional): Name. Defaults to 
                'latent_classifier'.
        """        
                
        super(LatentClassifier, self).__init__(name=name, **kwargs)
        
        self.n_clusters = n_clusters        
        self.dense_target = tkl.Dense(32, activation='relu', name=name + '_dense_target')
        self.dense_cluster0 = tkl.Dense(32, activation='relu', name=name + '_dense_cluster0')
        self.dense_cluster1 = tkl.Dense(16, activation='relu', name=name + '_dense_cluster1')
        self.out_cluster = tkl.Dense(n_clusters, activation='softmax', name=name + '_cluster')
        self.out_target = tkl.Dense(1, activation='sigmoid', name=name + '_target')
        
    def call(self, inputs, training=None):
        y = self.dense_target(inputs)
        y = self.out_target(y)
        
        c = self.dense_cluster0(inputs)
        c = self.dense_cluster1(c)
        c = self.out_cluster(c)
        return y, c
        
    def get_config(self):
        return {'n_clusters': self.n_clusters}

class ImageClassifier(tkl.Layer):
    
    def __init__(self, 
                 n_clusters: int, 
                 layer_filters=[16, 32, 32, 64, 64, 128, 256],
                 name='discriminator', 
                 **kwargs):
        """Classifier to predict cluster from an image (reconstruction).

        Args:
            n_clusters (int): Number of clusters
            layer_filters (list, optional): Convolutional filters in 
                each of 7 layers. Defaults to [16, 32, 32, 64, 64, 128, 256].
            name (str, optional): Name. Defaults to 'discriminator'.
        """        
        
        super(ImageClassifier, self).__init__(name=name, **kwargs)
        
        self.n_clusters = n_clusters
        self.layer_filters = layer_filters
        self.blocks = []

        for iLayer, nFilters in enumerate(layer_filters):
            conv = tkl.Conv2D(nFilters, 4, strides=(2, 2), padding='same', name=name + f'_conv{iLayer}')
            bn = tkl.BatchNormalization(name=name + f'_bn{iLayer}')
            prelu = tkl.PReLU(name=name + f'_prelu{iLayer}')
            self.blocks += [(conv, bn, prelu)]    
        
        # At this point, the data dimension should be 2 x 2
        self.cluster_head = [
            tkl.Conv2D(n_clusters, 2, padding='valid', name=name + '_convout'),
            tkl.Flatten(name=name + '_flatten'),
            tkl.Softmax(name=name + '_softmax')
        ]
                
    def call(self, inputs, training=None):
        x = inputs
        
        for conv, bn, prelu in self.blocks:
            x = conv(x)
            x = bn(x, training=training)
            x = prelu(x)   
        
        conv, flat, act = self.cluster_head
        c = conv(x)
        c = flat(c)
        pred_cluster = act(c)
        
        return pred_cluster
        
    def get_config(self):
        return {'n_clusters': self.n_clusters,
                'layer_filters': self.layer_filters}


class RandomEffectEncoder(Encoder):
    def __init__(self, 
                 n_latent_dims: int=56, 
                 layer_filters: list=[64, 64, 64, 128, 256, 512], 
                 post_loc_init_scale: float=0.1,
                 prior_scale: float=0.25,
                 kl_weight: float=1e-5,
                 name='encoder', 
                 **kwargs):
        """Encoder with random effect cluster-specific scales and biases 
        for each convolutional filter.

        Args:
            n_latent_dims (int, optional): Dimensionality of latent
                representation output. Defaults to 56.
            layer_filters (list, optional): Convolutional filters for each of
                the 6 layers. Defaults to [64, 64, 64, 128, 256, 512].
            post_loc_init_scale (float, optional): S.d. for random normal
                initialization of posteriors. Defaults to 0.1.
            prior_scale (float, optional): S.d. of normal prior distributions. 
                Defaults to 0.25.
            kl_weight (float, optional): KL Divergence loss weight. Defaults
                to 1e-5.
            name (str, optional): Model name. Defaults to 'encoder'.
        """        
        super(RandomEffectEncoder, self).__init__(n_latent_dims=n_latent_dims, 
                                                  layer_filters=layer_filters, 
                                                  name=name, **kwargs)
        
        # Replace batch norm layers with cluster-specific scale/bias layers
        self.re0 = ClusterScaleBiasBlock(self.layer_filters[0],
                                         post_loc_init_scale=post_loc_init_scale,
                                         prior_scale=prior_scale,
                                         kl_weight=kl_weight,
                                         name=name + '_re0')
        self.re1 = ClusterScaleBiasBlock(self.layer_filters[1],
                                         post_loc_init_scale=post_loc_init_scale,
                                         prior_scale=prior_scale,
                                         kl_weight=kl_weight,
                                         name=name + '_re1')
        self.re2 = ClusterScaleBiasBlock(self.layer_filters[2],
                                         post_loc_init_scale=post_loc_init_scale,
                                         prior_scale=prior_scale,
                                         kl_weight=kl_weight,
                                         name=name + '_re2')
        self.re3 = ClusterScaleBiasBlock(self.layer_filters[3],
                                         post_loc_init_scale=post_loc_init_scale,
                                         prior_scale=prior_scale,
                                         kl_weight=kl_weight,
                                         name=name + '_re3')
        self.re4 = ClusterScaleBiasBlock(self.layer_filters[4],
                                         post_loc_init_scale=post_loc_init_scale,
                                         prior_scale=prior_scale,
                                         kl_weight=kl_weight,
                                         name=name + '_re4')
        self.re5 = ClusterScaleBiasBlock(self.layer_filters[5],
                                         post_loc_init_scale=post_loc_init_scale,
                                         prior_scale=prior_scale,
                                         kl_weight=kl_weight,
                                         name=name + '_re5')
        
        del self.bn0, self.bn1, self.bn2, self.bn3, self.bn4, self.bn5, self.bn_out
        
    def call(self, inputs, training=None):
        x, z = inputs
        x = self.conv0(x)
        x = self.re0((x, z), training=training)
        x = self.prelu0(x)
        
        x = self.conv1(x)
        x = self.re1((x, z), training=training)
        x = self.prelu1(x)
        
        x = self.conv2(x)
        x = self.re2((x, z), training=training)
        x = self.prelu2(x)
        
        x = self.conv3(x)
        x = self.re3((x, z), training=training)
        x = self.prelu3(x)
        
        x = self.conv4(x)
        x = self.re4((x, z), training=training)
        x = self.prelu4(x)
        
        x = self.conv5(x)
        x = self.re5((x, z), training=training)
        x = self.prelu5(x)
        
        x = self.flatten(x)
        x = self.dense(x)
        
        return x

    
class RandomEffectDecoder(Decoder):
    def __init__(self, 
                 image_shape: tuple=(256, 256, 1),
                 layer_filters: list=[512, 256, 128, 64, 64, 64],
                 post_loc_init_scale: float=0.1,
                 prior_scale: float=0.25,
                 kl_weight: float=1e-5,
                 name='decoder', 
                 **kwargs): 
        """Decoder with random effect cluster-specific scales and biases 
        for each convolutional filter.

        Args:
            image_shape (tuple, optional): Shape of reconstructed image.
                Defaults to (256, 256, 1).
            layer_filters (list, optional): Convolutional filters for each of
                the 6 layers. Defaults to [64, 64, 64, 128, 256, 512].
            post_loc_init_scale (float, optional): S.d. for random normal
                initialization of posteriors. Defaults to 0.1.
            prior_scale (float, optional): S.d. of normal prior distributions. 
                Defaults to 0.25.
            kl_weight (float, optional): KL Divergence loss weight. Defaults
                to 1e-5.
            name (str, optional): Model name. Defaults to 'encoder'.
        """  
        super(RandomEffectDecoder, self).__init__(image_shape=image_shape,                      layer_filters=layer_filters, 
                                                  name=name, 
                                                  **kwargs)
        
        # Replace batch norm layers with cluster-specific scale/bias layers
        self.re0 = ClusterScaleBiasBlock(self.layer_filters[1],
                                         post_loc_init_scale=post_loc_init_scale,
                                         prior_scale=prior_scale,
                                         kl_weight=kl_weight,
                                         name=name + '_re0')
        self.re1 = ClusterScaleBiasBlock(self.layer_filters[2],
                                         post_loc_init_scale=post_loc_init_scale,
                                         prior_scale=prior_scale,
                                         kl_weight=kl_weight,
                                         name=name + '_re1')
        self.re2 = ClusterScaleBiasBlock(self.layer_filters[3],
                                         post_loc_init_scale=post_loc_init_scale,
                                         prior_scale=prior_scale,
                                         kl_weight=kl_weight,
                                         name=name + '_re2')
        self.re3 = ClusterScaleBiasBlock(self.layer_filters[4],
                                         post_loc_init_scale=post_loc_init_scale,
                                         prior_scale=prior_scale,
                                         kl_weight=kl_weight,
                                         name=name + '_re3')
        self.re4 = ClusterScaleBiasBlock(self.layer_filters[5],
                                         post_loc_init_scale=post_loc_init_scale,
                                         prior_scale=prior_scale,
                                         kl_weight=kl_weight,
                                         name=name + '_re4')
        self.re5 = ClusterScaleBiasBlock(1,
                                         post_loc_init_scale=post_loc_init_scale,
                                         prior_scale=prior_scale,
                                         kl_weight=kl_weight,
                                         name=name + '_re5')
        del self.bn0, self.bn1, self.bn2, self.bn3, self.bn4, self.bn5
        
    def call(self, inputs, training=None):
        x, z = inputs
        x = self.dense(x)
        x = self.reshape(x)
        x = self.prelu_dense(x)
        
        x = self.tconv0(x)
        x = self.re0((x, z), training=training)
        x = self.prelu0(x)
        
        x = self.tconv1(x)
        x = self.re1((x, z), training=training)
        x = self.prelu1(x)
        
        x = self.tconv2(x)
        x = self.re2((x, z), training=training)
        x = self.prelu2(x)
        
        x = self.tconv3(x)
        x = self.re3((x, z), training=training)
        x = self.prelu3(x)
        
        x = self.tconv4(x)
        x = self.re4((x, z), training=training)
        x = self.prelu4(x)
        
        x = self.tconv5(x)
        x = self.re5((x, z), training=training)
        x = self.sigmoid_out(x)
        
        return x
    
class DomainEnhancingAutoencoderClassifier(tf.keras.Model):
    
    def __init__(self,
                 image_shape: tuple=(256, 256, 1),
                 n_clusters: int=10,
                 n_latent_dims: int=56, 
                 encoder_layer_filters: list=[64, 64, 64, 128, 256, 512],
                 post_loc_init_scale: float=0.1,
                 prior_scale: float=0.25,
                 kl_weight: float=1e-5,
                 name='autoencoder',
                 **kwargs):
        """
        Autoencoder that emphasizes cluster differences in both the compressed
        latent representation and the reconstructed image. This is done by
        adding random effects layers to the encoder and decoder, as well as
        using additional classifiers to maximize the cluster-predictive
        information present in the latents and reconstructions.

        Args:
            image_shape (tuple, optional): Input image size. Defaults to 
                (256, 256, 1). 
            n_clusters (int, optional): Number of clusters. Defaults to 10.
            n_latent_dims (int, optional): Dimensionality of latent 
                representations. Defaults to 56.
            encoder_layer_filters (list, optional): Convolutional filters 
                in each of the 6 encoder layers. Defaults to 
                [64, 64, 64, 128, 256, 512]. 
            post_loc_init_scale (float, optional): S.d. for random normal
                initialization of posteriors. Defaults to 0.1.
            prior_scale (float, optional): S.d. of normal prior distributions. 
                Defaults to 0.25.
            kl_weight (float, optional): KL Divergence loss weight. Defaults
                to 1e-5.
            name (str, optional): Model name. Defaults to 'autoencoder'.
        """        
        
        super(DomainEnhancingAutoencoderClassifier, self).__init__(name=name, **kwargs)
        
        self.image_shape = image_shape
        self.n_latent_dims = n_latent_dims
        self.encoder_layer_filters = encoder_layer_filters
        # Decoder layers should mirror the encoder layers
        self.decoder_layer_filters = encoder_layer_filters[-1::-1]
        
        self.encoder = RandomEffectEncoder(n_latent_dims=n_latent_dims,
                                           layer_filters=encoder_layer_filters,
                                           post_loc_init_scale=post_loc_init_scale,
                                           prior_scale=prior_scale,
                                           kl_weight=kl_weight)
        
        self.decoder = RandomEffectDecoder(layer_filters=self.decoder_layer_filters,
                                           post_loc_init_scale=post_loc_init_scale,
                                           prior_scale=prior_scale,
                                           kl_weight=kl_weight)
        
        # Classifiers to guide the latents and reconstructions to producing outputs
        # that are laden with cluster-predictive information
        self.latent_classifier = LatentClassifier(n_clusters=n_clusters)
        self.image_classifier = ImageClassifier(n_clusters=n_clusters)
        
    def call(self, inputs, training=None):
        if len(inputs) != 2:
            raise ValueError('Model inputs need to be a tuple of (images, clusters)')
        
        x, z = inputs
        # Call encoder and get latents        
        latent = self.encoder((x, z), training=training)
        # Predict cluster from latents
        pred_y, pred_c_latent = self.latent_classifier(latent)
        # Reconstruct image from latents
        recon = self.decoder((latent, z), training=training)
        # Predict cluster from reconstruction
        pred_c_recon = self.image_classifier(recon)
        
        return recon, pred_y, pred_c_latent, pred_c_recon
    
    def compile(self,
                loss_recon=tf.keras.losses.MeanSquaredError(),
                loss_class=tf.keras.losses.BinaryCrossentropy(),
                loss_cluster=tf.keras.losses.CategoricalCrossentropy(),
                metric_class=tf.keras.metrics.AUC(name='auroc'),
                optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                loss_class_weight=0.01,
                loss_latent_cluster_weight=0.001,
                loss_image_cluster_weight=0.001
                ):
        """Compile model with given losses, metrics, and optimizer.
        
        The autoencoder-classifier is trained with this loss function:
        loss_recon                          (image reconstruction loss)
        + loss_class_weight * loss_class    (phenotype classification loss)
        + loss_latent_cluster_weight * loss_cluster(latent classifier)  (cluster predictiveness of latents)
        + loss_image_cluster_weight * loss_cluster(recon classifier)  (cluster predictiveness of reconstructions)

        Args:
            loss_recon (loss, optional): Image reconstruction loss. Defaults to
                tf.keras.losses.MeanSquaredError().
            loss_class (loss, optional): Auxiliary classification loss. Defaults
                to tf.keras.losses.BinaryCrossentropy().
            loss_cluster (loss, optional): Cluster classification loss. Defaults 
                to tf.keras.losses.CategoricalCrossentropy().
            metric_class (metric, optional): Auxiliary classification metric. 
                Defaults to tf.keras.metrics.AUC(name='auroc').
            optimizer (optimizer, optional): Optimizer. Defaults to 
                tf.keras.optimizers.Adam(lr=0.0001).
            loss_class_weight (float, optional): Weight for auxiliary classification 
                loss. Defaults to 0.01.
            loss_latent_cluster_weight (float, optional): Weight for cluster prediction 
                loss for latents. Defaults to 0.001.
            loss_image_cluster_weight (float, optional): Weight for cluster prediction 
                loss for recons. Defaults to 0.001.
        """   
        
        super().compile()

        self.loss_recon = loss_recon
        self.loss_class = loss_class
        self.loss_cluster = loss_cluster
        self.optimizer = optimizer
        self.metric_class = metric_class
        self.loss_class_weight = loss_class_weight
        self.loss_latent_cluster_weight = loss_latent_cluster_weight
        self.loss_image_cluster_weight = loss_image_cluster_weight
        
        # Loss trackers (running means)
        self.loss_recon_tracker = tf.keras.metrics.Mean(name='recon_loss')
        self.loss_class_tracker = tf.keras.metrics.Mean(name='class_loss')
        self.loss_latent_cluster_tracker = tf.keras.metrics.Mean(name='la_clus_loss')
        self.loss_image_cluster_tracker = tf.keras.metrics.Mean(name='im_clus_loss')
        self.loss_kl_tracker = tf.keras.metrics.Mean(name='kld')
        self.loss_total_tracker = tf.keras.metrics.Mean(name='total_loss')
        
        
    @property
    def metrics(self):
        return [self.loss_recon_tracker,
                self.loss_class_tracker,
                self.metric_class,
                self.loss_latent_cluster_tracker,
                self.loss_image_cluster_tracker,
                self.loss_kl_tracker,
                self.loss_total_tracker]
        
    def _compute_update_loss(self, loss_recon, loss_class, loss_latent_cluster, loss_image_cluster,
                             training=True):
        '''Compute total loss and update loss running means'''
        self.loss_recon_tracker.update_state(loss_recon)
        self.loss_class_tracker.update_state(loss_class)
        self.loss_latent_cluster_tracker.update_state(loss_latent_cluster)
        self.loss_image_cluster_tracker.update_state(loss_image_cluster)
        
        if training:
            kld = tf.reduce_mean(self.encoder.losses) + tf.reduce_mean(self.decoder.losses)
            self.loss_kl_tracker.update_state(kld)
        else:
            # KLD can't be computed at inference time because posteriors are simplified to 
            # point estimates
            kld = 0
        
        loss_total = loss_recon \
            + (self.loss_class_weight * loss_class) \
            + (self.loss_latent_cluster_weight * loss_latent_cluster) \
            + (self.loss_image_cluster_weight * loss_image_cluster) \
            + kld
        self.loss_total_tracker.update_state(loss_total)
        
        return loss_total
                
    def train_step(self, data):
        if len(data) == 3:
            (images, clusters), (_, labels), sample_weights = data
        else:
            (images, clusters), (_, labels) = data
            sample_weights = None
        
        # Train the recon image-cluster classifier on real images
        with tf.GradientTape() as gt:
            # Predict cluster from real images
            pred_c_image = self.image_classifier(images)
            loss_image_cluster = self.loss_cluster(clusters, pred_c_image)
            
        grads = gt.gradient(loss_image_cluster, self.image_classifier.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.image_classifier.trainable_variables))
        
        # Train the rest of the model
        with tf.GradientTape() as gt:
            recon, pred_y, pred_c_latent, pred_c_recon = self((images, clusters), training=True)
            loss_recon = self.loss_recon(images, recon)
            loss_class = self.loss_class(labels, pred_y)
            loss_latent_cluster = self.loss_cluster(clusters, pred_c_latent)
            loss_image_cluster = self.loss_cluster(clusters, pred_c_recon)
            
            loss_total = self._compute_update_loss(loss_recon, loss_class, loss_latent_cluster, 
                                                   loss_image_cluster)
            
        lsWeights = self.encoder.trainable_variables + self.decoder.trainable_variables \
            + self.latent_classifier.trainable_variables
            
        grads = gt.gradient(loss_total, lsWeights)
        self.optimizer.apply_gradients(zip(grads, lsWeights))
        
        # Update metrics
        self.metric_class.update_state(labels, pred_y)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        (images, clusters), (_, labels) = data
                        
        recon, pred_y, pred_c_latent, pred_c_recon = self((images, clusters), training=False)
        loss_recon = self.loss_recon(images, recon)
        loss_class = self.loss_class(labels, pred_y)
        loss_latent_cluster = self.loss_cluster(clusters, pred_c_latent)
        loss_image_cluster = self.loss_cluster(clusters, pred_c_recon)
        
        _ = self._compute_update_loss(loss_recon, loss_class, loss_latent_cluster, 
                                      loss_image_cluster, training=False)
        
        self.metric_class.update_state(labels, pred_y)
        return {m.name: m.result() for m in self.metrics}

        
class MixedEffectAuxClassifier(tkl.Layer):
           
    def __init__(self, 
                 units: int=32,
                 post_loc_init_scale: float=0.1,
                 prior_scale: float=0.25,
                 kl_weight: float=1e-5,
                 name='auxclassifier', **kwargs):
        """Mixed effects dense classifier with one hidden layer 
        and sigmoid output.

        Args:
            units (int, optional): Number of hidden layer neurons. Defaults 
                to 32.
            post_loc_init_scale (float, optional): S.d. for random normal
                initialization of posteriors. Defaults to 0.1.
            prior_scale (float, optional): S.d. of normal prior distributions. 
                Defaults to 0.25.
            kl_weight (float, optional): KL Divergence loss weight. Defaults
                to 1e-5.
            name (str, optional): Name. Defaults to 'auxclassifier'.
        """        
        super().__init__(name=name, **kwargs)
        
        self.units = units
        
        self.hidden = tkl.Dense(units, name=name + '_dense')
        self.activation = tkl.LeakyReLU(name=name + '_leakyrelu')
        self.re_slopes = RandomEffects(units, 
                                       post_loc_init_scale=post_loc_init_scale,
                                       prior_scale=prior_scale,
                                       kl_weight=kl_weight,
                                       name='re_slopes')        
        self.dense_out = tkl.Dense(1, name=name + '_output')
        self.re_intercept = RandomEffects(1,
                                          post_loc_init_scale=post_loc_init_scale,
                                          prior_scale=prior_scale,
                                          kl_weight=kl_weight,
                                          name='re_intercept')
        self.sigmoid = tkl.Activation('sigmoid', name='sigmoid')
        
    def call(self, inputs, training=None):
        x, z = inputs
        x = self.hidden(x)
        x = self.activation(x)
        g = self.re_slopes(z, training=training)
        x = self.dense_out((1 + g) * x)
        b = self.re_intercept(z, training=training)
        y = self.sigmoid(b + x)
        return y
    
    def get_config(self):
        return {'units': self.units}

class MixedEffectsAEC(DomainAdversarialAEC):
    ''' Still under development!'''
    
    def __init__(self,
                 image_shape: tuple=(256, 256, 1),
                 n_clusters: int=10,
                 n_latent_dims: int=56, 
                 encoder_layer_filters: list=[64, 64, 64, 128, 256, 512],
                 classifier_hidden_units: int=32,
                 adversary_layer_filters: list=[16, 32, 32, 64, 64, 128, 128, 256],
                 post_loc_init_scale: float=0.1,
                 prior_scale: float=0.25,
                 kl_weight: float=1e-5,
                 name='autoencoder',
                 **kwargs
                 ):
        """Mixed effects autoencoder-classifier. Adds an adversarial 
        classifier to predict the cluster/domain membership of each sample 
        based on the encoder's intermediate outputs. This compels the 
        encoder to learn features unassociated with cluster characteristics.
        Then, random effects are learned in the decoder through cluster-specific
        feature scales and biases and in the auxiliary classifier through cluster
        -specific slopes and intercepts.

        Args:
            image_shape (tuple, optional): Image shape. Defaults to 
                (256, 256, 1).
            n_clusters (int, optional): Number of clusters. Defaults to 10.
            n_latent_dims (int, optional): Size of latent representation. 
                Defaults to 56.
            encoder_layer_filters (list, optional): Number of filters in 
                each encoder layer. Defaults to [64, 64, 64, 128, 256, 512].
            classifier_hidden_units (int, optional): Number of neurons in 
                auxiliary classifier hidden layer. Defaults to 32.
            adversary_layer_filters (list, optional): Number of filters in 
                each adversary layer. Defaults to [16, 32, 32, 64, 64, 128, 128, 256].
            post_loc_init_scale (float, optional): S.d. for random normal
                initialization of posteriors. Defaults to 0.1.
            prior_scale (float, optional): S.d. of normal prior distributions. 
                Defaults to 0.25.
            kl_weight (float, optional): KL Divergence loss weight. Defaults
                to 1e-5.
            name (str, optional): Model name. Defaults to 'autoencoder'.
        """        
        super(BaseAutoencoderClassifier, self).__init__(
            name=name, 
            **kwargs)

        self.image_shape = image_shape
        self.n_latent_dims = n_latent_dims
        self.encoder_layer_filters = encoder_layer_filters
        self.decoder_layer_filters = encoder_layer_filters[-1::-1]
        self.classifier_hidden_units = classifier_hidden_units
                
        self.encoder = Encoder(n_latent_dims=n_latent_dims,
                               layer_filters=encoder_layer_filters,
                               return_layer_activations=True)
            
        lsEncoderLayers = [self.encoder.conv0,
            self.encoder.conv1,
            self.encoder.conv2,
            self.encoder.conv3, 
            self.encoder.conv4,
            self.encoder.conv5]
        
        self.decoder_fe = TiedDecoder(lsEncoderLayers, 
                                      image_shape=image_shape, 
                                      layer_filters=self.decoder_layer_filters,
                                      name='decoder_fe')
                
        self.decoder_re = RandomEffectDecoder(layer_filters=self.decoder_layer_filters,
                                              post_loc_init_scale=post_loc_init_scale,
                                              prior_scale=prior_scale,
                                              kl_weight=kl_weight,
                                              name='decoder_re')
        
        self.classifier = MixedEffectAuxClassifier(units=classifier_hidden_units,
                                                   post_loc_init_scale=post_loc_init_scale,
                                                   prior_scale=prior_scale,
                                                   kl_weight=kl_weight)
                
        self.adversary = AdversarialClassifier(image_shape, n_clusters,
                                               layer_filters=adversary_layer_filters)
        
        self.recon_cluster_classifier = ImageClassifier(n_clusters=n_clusters,
                                                        name='recon_classifier')
        
    def call(self, inputs, training=None):
        images, clusters = inputs
        encoder_outs = self.encoder(images, training=training)
        latent = encoder_outs[-1]
        recon_re = self.decoder_re((latent, clusters), training=training)
        recon_fe = self.decoder_fe(latent, training=training)

        classification = self.classifier((latent, clusters), training=training)
        
        pred_cluster = self.adversary(encoder_outs)
        
        return (recon_re, recon_fe, classification, pred_cluster)
    
    def compile(self, 
                loss_recon=tf.keras.losses.MeanSquaredError(), 
                loss_class=tf.keras.losses.BinaryCrossentropy(), 
                loss_adv=tf.keras.losses.CategoricalCrossentropy(), 
                metric_class=tf.keras.metrics.AUC(name='auroc'), 
                metric_adv=tf.keras.metrics.CategoricalAccuracy(name='acc'), 
                opt_autoencoder=tf.keras.optimizers.Adam(lr=0.0001), 
                opt_adversary=tf.keras.optimizers.Adam(lr=0.0001), 
                opt_recon_classifier=tf.keras.optimizers.Adam(lr=0.0001),
                loss_recon_weight=1, 
                loss_recon_fe_weight=1,
                loss_class_weight=0.01, 
                loss_gen_weight=0.2,
                loss_recon_cluster_weight=0.01):
        
        super().compile(loss_recon, 
                        loss_class, 
                        loss_adv, 
                        metric_class, 
                        metric_adv, 
                        opt_autoencoder, 
                        opt_adversary, 
                        loss_recon_weight, 
                        loss_class_weight, 
                        loss_gen_weight)

        self.loss_recon_fe_weight = loss_recon_fe_weight
        self.loss_recon_fe_tracker = tf.keras.metrics.Mean('recon_fe_loss')
        
        self.loss_kl_tracker = tf.keras.metrics.Mean('kld')
        self.loss_recon_cluster_weight = loss_recon_cluster_weight
        self.loss_recon_cluster_tracker = tf.keras.metrics.Mean('recon_clus_loss')
        self.opt_recon_classifier = opt_recon_classifier
    
    @property
    def metrics(self):
        return [self.loss_recon_tracker,
                self.loss_class_tracker,
                self.loss_adv_tracker,
                self.loss_kl_tracker,
                self.loss_recon_cluster_tracker,
                self.loss_total_tracker,
                self.metric_class,
                self.metric_adv]
    
    def _compute_update_loss(self, loss_recon_re, loss_recon_fe, loss_class, 
                             loss_gen, loss_recon_cluster,
                             training=True):
        '''Compute total loss and update loss running means'''
        self.loss_recon_tracker.update_state(loss_recon_re)
        self.loss_recon_fe_tracker.update_state(loss_recon_fe)
        self.loss_class_tracker.update_state(loss_class)
        self.loss_recon_cluster_tracker.update_state(loss_recon_cluster)
        
        if training:
            kld = tf.reduce_mean(self.decoder_re.losses) + tf.reduce_mean(self.classifier.losses)
            self.loss_kl_tracker.update_state(kld)
        else:
            # KLD can't be computed at inference time because posteriors are simplified to 
            # point estimates
            kld = 0
        
        loss_total = (self.loss_recon_weight * loss_recon_re) \
            + (self.loss_recon_fe_weight * loss_recon_fe) \
            + (self.loss_class_weight * loss_class) \
            + (self.loss_gen_weight * loss_gen) \
            + (self.loss_recon_cluster_weight * loss_recon_cluster) \
            + kld
        self.loss_total_tracker.update_state(loss_total)
        
        return loss_total
    
    def train_step(self, data):
        if len(data) == 3:
            (images, clusters), (_, labels), sample_weights = data
        else:
            (images, clusters), (_, labels) = data
            sample_weights = None
            
        # Train adversarial classifier
        encoder_outs = self.encoder(images, training=True)
        with tf.GradientTape() as gt:
            pred_cluster = self.adversary(encoder_outs)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            
        grads_adv = gt.gradient(loss_adv, self.adversary.trainable_variables)
        self.opt_adversary.apply_gradients(zip(grads_adv, self.adversary.trainable_variables))
        
        self.metric_adv.update_state(clusters, pred_cluster)
        self.loss_adv_tracker.update_state(loss_adv)
        
        # Train the recon-cluster classifier on real images
        with tf.GradientTape() as gt2:
            # Predict cluster from real images
            pred_c_image = self.recon_cluster_classifier(images)
            loss_image_cluster = self.loss_adv(clusters, pred_c_image, sample_weight=sample_weights)
        
        grads_rc = gt2.gradient(loss_image_cluster, self.recon_cluster_classifier.trainable_variables)
        self.opt_recon_classifier.apply_gradients(zip(grads_rc, 
                                                      self.recon_cluster_classifier.trainable_variables))
        
        self.loss_recon_cluster_tracker.update_state(loss_image_cluster)
        
        # Train the rest of the model
        with tf.GradientTape(persistent=True) as gt3:
            pred_recon_re, pred_recon_fe, pred_class, pred_cluster = self((images, clusters), training=True)
            loss_class = self.loss_class(labels, pred_class, sample_weight=sample_weights)
            loss_recon_re = self.loss_recon(images, pred_recon_re, sample_weight=sample_weights)
            loss_recon_fe = self.loss_recon(images, pred_recon_fe, sample_weight=sample_weights)
            loss_gen = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            
            pred_recon_cluster = self.recon_cluster_classifier(pred_recon_re)
            loss_recon_cluster = self.loss_adv(clusters, pred_recon_cluster)
            
            total_loss = self._compute_update_loss(loss_recon_re, loss_recon_fe, loss_class, 
                                                   loss_gen, loss_recon_cluster, training=True)
                           
        lsWeights = self.encoder.trainable_variables + self.decoder_re.trainable_variables \
                + self.decoder_fe.trainable_variables + self.classifier.trainable_variables
        grads_aec = gt3.gradient(total_loss, lsWeights)
        self.opt_autoencoder.apply_gradients(zip(grads_aec, lsWeights))
        
        self.metric_class.update_state(labels, pred_class)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        (images, clusters), (_, labels) = data
                        
        pred_recon_re, pred_recon_fe, pred_class, pred_cluster = self((images, clusters), 
                                                                      training=False)
        loss_class = self.loss_class(labels, pred_class)
        loss_recon_re = self.loss_recon(images, pred_recon_re)
        loss_recon_fe = self.loss_recon(images, pred_recon_fe)
        loss_adv = self.loss_adv(clusters, pred_cluster)
        pred_recon_cluster = self.recon_cluster_classifier(pred_recon_re)
        loss_recon_cluster = self.loss_adv(clusters, pred_recon_cluster)
            
        _ = self._compute_update_loss(loss_recon_re, loss_recon_fe, loss_class, 
                                      loss_adv, loss_recon_cluster, training=False)
                    
        self.metric_class.update_state(labels, pred_class)
        self.metric_adv.update_state(clusters, pred_cluster)
        self.loss_adv_tracker.update_state(loss_adv)
        
        return {m.name: m.result() for m in self.metrics}
