import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.legacy import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Reshape, Embedding, Concatenate, LayerNormalization, BatchNormalization, Dropout, Add
import numpy as np

def get_model(model_name="", input_size=None, output_size=None, target="", perc_numeric=0., RS=42):
    # if target not in ["continuous", "binary", "categorical"]:
    #     raise ValueError("target has to be either 'continuous', 'binary' or 'categorical'")
    # if target == "continuous":
    #     activation = 'linear'
    # elif target == "binary":
    #     activation = 'sigmoid'
    # elif target == "categorical":
    #     activation == 'softmax'

    if model_name not in ["tabtransformer", "simchoni_2021", "AutoGluon", "AutoGluon_no_numeric", "Click_prediction_small", "Amazon_employee_access","video-game-sales", "hpc-job-scheduling", "road-safety-drivers-sex", "open_payments", "okcupid-stem",    "Midwest_survey", "Diabetes130US",    "KDDCup09_upselling", "adult", "kdd_internet_usage", "churn", "porto-seguro", "kick", "eucalyptus", "wine-reviews", "medical_charges","avocado-sales", "employee_salaries", "particulate-matter-ukair-2017", "flight-delay-usa-dec-2017", "nyc-taxi-green-dec-2016", "ames-housing","AutoGluon_large", "ResNet"]:
        print("model_name not in list of implemented models, return default model")
    tf.random.set_seed(RS)
    # if dataset_name is not None:
    #     df = pd.read_csv(f"./data/raw/{dataset_name}/{dataset_name}.csv")
    #     df = df.drop("Unnamed: 0", axis=1)
    #     df = df.drop(df.columns[df.isna().sum() / df.shape[0] > 0.95], axis=1)

    if output_size == None:
        raise ValueError("Output size has to be defined")

    if model_name=="Amazon_employee_access":
        model = lambda X: tf.zeros(X.shape[0])
        optimizer = Adam(learning_rate=0.001)

    elif model_name == "simchoni_2021":
        model = Sequential()
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(12, activation='relu'))
        model.add(Dropout(0.25))

        # add linear output layer
        model.add(Dense(output_size, activation='linear'))

        # name model
        model._name = "Simchoni"

        optimizer = Adam(learning_rate=0.001)

    elif model_name == "tabtransformer":
        # dropout = 0.1 ?? SagMaker website say 0.1, so use 0.1
        # All models used early stopping based on the performance on the validation set and the early stopping patience (the number of epochs) is set as 15
        # Hyperparameter default values from: https://docs.aws.amazon.com/sagemaker/latest/dg/tabtransformer-hyperparameters.html
        if input_size is None:
            raise ValueError("Input size has to be defined for TabTransformer")
        model = Sequential()
        model.add(LayerNormalization(epsilon=1e-6))
        model.add(Dense(input_size*4, activation='selu'))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())
        model.add(Dense(input_size*2, activation='selu'))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        # add linear output layer
        model.add(Dense(output_size, activation='linear'))

        # name model
        model._name = "TabTransformer"

        optimizer = AdamW(weight_decay = 0.0001, learning_rate=0.001) # Parameters were tuned in original paper

    elif model_name == "AutoGluon":
    # Copied from Paper & https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/tabular_nn/utils/nn_architecture_utils.py
    ### skip connections need functional API
        if input_size is None:
            raise ValueError("Input size has to be defined for AutoGluon")
        inputs = Input(shape=(input_size,))


        ### Get model layer dimensions
        min_numeric_embed_dim = 32
        max_numeric_embed_dim = 2056
        max_layer_width = 2056
        # Main dense model
        if target == "continuous":
            default_layer_sizes = [256,
                                   128]  # overall network will have 4 layers. Input layer, 256-unit hidden layer, 128-unit hidden layer, output layer.
        else:
            default_sizes = [256, 128]  # will be scaled adaptively
            # base_size = max(1, min(num_net_outputs, 20)/2.0) # scale layer width based on number of classes
            base_size = max(1, min(output_size,
                                   100) / 50)  # TODO: Updated because it improved model quality and made training far faster
            default_layer_sizes = [defaultsize * base_size for defaultsize in default_sizes]
        layer_expansion_factor = 1  # TODO: consider scaling based on num_rows, eg: layer_expansion_factor = 2-np.exp(-max(0,train_dataset.num_examples-10000))
        layers = [int(min(max_layer_width, layer_expansion_factor * defaultsize)) for defaultsize in default_layer_sizes]

        # numeric embed dim
        vector_dim = 0  # total dimensionality of vector features (I think those should be transformed string features, which we don't have)
        prop_vector_features = perc_numeric  # Fraction of features that are numeric
        first_layer_width = layers[0] # 256 as we use the default parameters
        numeric_embedding_size = int(min(max_numeric_embed_dim,
                       max(min_numeric_embed_dim, first_layer_width * prop_vector_features * np.log10(vector_dim + 10))))

        ### Define model
        # Numeric Embedding
        x = Dense(numeric_embedding_size, activation="relu")(inputs)

        # Main network:
        # Dense block 1
        x1 = Dense(layers[0], activation="relu")(x)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)
        # Dense block 2
        x1 = Dense(layers[1], activation="relu")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)
        # Dense block 3
        # dense2 = Dense(128, activation='linear') # which activation function???
        x1 = Dense(output_size, activation='linear')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)

        # Linear skip connection:
        x2 = Dense(output_size, activation='linear')(x)

        # connect 2 parts again
        added = Add()([x1, x2])
        outputs = tf.keras.layers.Activation("linear")(added)

        model = keras.Model(inputs=inputs, outputs=outputs, name="AutoGluon")

        optimizer = Adam(learning_rate=3e-4, decay=1e-6) # from github source code https://github.com/awslabs/autogluon/blob/master/tabular/src/autogluon/tabular/models/tabular_nn/hyperparameters/parameters.py
        # early stopping based on validation performance
        # Other params: epochs=500, patience=20

    elif model_name == "AutoGluon_large":
    # Copied from Paper & https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/tabular_nn/utils/nn_architecture_utils.py
    ### skip connections need functional API
        if input_size is None:
            raise ValueError("Input size has to be defined for AutoGluon")
        inputs = Input(shape=(input_size,))


        ### Get model layer dimensions
        min_numeric_embed_dim = 32
        max_numeric_embed_dim = 2056
        max_layer_width = 2056
        # Main dense model
        if target == "continuous":
            default_layer_sizes = [256,
                                   128]  # overall network will have 4 layers. Input layer, 256-unit hidden layer, 128-unit hidden layer, output layer.
        else:
            default_sizes = [256, 128]  # will be scaled adaptively
            # base_size = max(1, min(num_net_outputs, 20)/2.0) # scale layer width based on number of classes
            base_size = max(1, min(output_size,
                                   100) / 50)  # TODO: Updated because it improved model quality and made training far faster
            default_layer_sizes = [defaultsize * base_size for defaultsize in default_sizes]
        layer_expansion_factor = 1  # TODO: consider scaling based on num_rows, eg: layer_expansion_factor = 2-np.exp(-max(0,train_dataset.num_examples-10000))
        layers = [int(min(max_layer_width, layer_expansion_factor * defaultsize)) for defaultsize in default_layer_sizes]

        # numeric embed dim
        vector_dim = 0  # total dimensionality of vector features (I think those should be transformed string features, which we don't have)
        prop_vector_features = perc_numeric  # Fraction of features that are numeric
        first_layer_width = layers[0] # 256 as we use the default parameters
        numeric_embedding_size = int(min(max_numeric_embed_dim,
                       max(min_numeric_embed_dim, first_layer_width * prop_vector_features * np.log10(vector_dim + 10))))

        ### Define model
        # Numeric Embedding
        x = Dense(numeric_embedding_size, activation="relu")(inputs)

        # Main network:
        # Dense block 1
        x1 = Dense(layers[0], activation="relu")(x)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)
        # Dense block 2
        x1 = Dense(layers[1], activation="relu")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)
        # Dense block 3
        x1 = Dense(layers[1], activation="relu")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)
        # Dense block 4
        x1 = Dense(layers[1], activation="relu")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)
        # Dense block 5
        x1 = Dense(layers[1], activation="relu")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)
        # Dense block 6
        x1 = Dense(layers[1], activation="relu")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)
        # Dense block 7
        x1 = Dense(layers[1], activation="relu")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)
        # Dense block 8
        x1 = Dense(layers[1], activation="relu")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)
        # Dense block 9
        x1 = Dense(layers[1], activation="relu")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)
        # Dense block 10
        # dense2 = Dense(128, activation='linear') # which activation function???
        x1 = Dense(output_size, activation='linear')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)

        # Linear skip connection:
        x2 = Dense(output_size, activation='linear')(x)

        # connect 2 parts again
        added = Add()([x1, x2])
        outputs = tf.keras.layers.Activation("linear")(added)

        model = keras.Model(inputs=inputs, outputs=outputs, name="AutoGluon")

        optimizer = Adam(learning_rate=3e-4, decay=1e-6) # from github source code https://github.com/awslabs/autogluon/blob/master/tabular/src/autogluon/tabular/models/tabular_nn/hyperparameters/parameters.py
        # early stopping based on validation performance
        # Other params: epochs=500, patience=20
        
        
    elif model_name == "AutoGluon_no_numeric":
    # Copied from Paper & https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/tabular_nn/utils/nn_architecture_utils.py
    ### skip connections need functional API 
        if input_size is None:
            raise ValueError("Input size has to be defined for AutoGluon")
        inputs = Input(shape=(input_size,))


        ### Get model layer dimensions
        min_numeric_embed_dim = 32
        max_numeric_embed_dim = 2056
        max_layer_width = 2056
        # Main dense model
        if target == "continuous":
            default_layer_sizes = [256,
                                   128]  # overall network will have 4 layers. Input layer, 256-unit hidden layer, 128-unit hidden layer, output layer.
        else:
            default_sizes = [256, 128]  # will be scaled adaptively
            # base_size = max(1, min(num_net_outputs, 20)/2.0) # scale layer width based on number of classes
            base_size = max(1, min(output_size,
                                   100) / 50)  # TODO: Updated because it improved model quality and made training far faster
            default_layer_sizes = [defaultsize * base_size for defaultsize in default_sizes]
        layer_expansion_factor = 1  # TODO: consider scaling based on num_rows, eg: layer_expansion_factor = 2-np.exp(-max(0,train_dataset.num_examples-10000))
        layers = [int(min(max_layer_width, layer_expansion_factor * defaultsize)) for defaultsize in default_layer_sizes]

        # Main network:
        # Dense block 1
        x1 = Dense(layers[0], activation="relu")(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)
        # Dense block 2
        x1 = Dense(layers[1], activation="relu")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)
        # Dense block 3
        # dense2 = Dense(128, activation='linear') # which activation function???
        x1 = Dense(output_size, activation='linear')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.1)(x1)

        # Linear skip connection:
        x2 = Dense(output_size, activation='linear')(inputs)

        # connect 2 parts again
        added = Add()([x1, x2])
        outputs = tf.keras.layers.Activation("linear")(added)

        model = keras.Model(inputs=inputs, outputs=outputs, name="AutoGluon")

        optimizer = Adam(learning_rate=3e-4, decay=1e-6) # from github source code https://github.com/awslabs/autogluon/blob/master/tabular/src/autogluon/tabular/models/tabular_nn/hyperparameters/parameters.py
        # early stopping based on validation performance
        # Other params: epochs=500, patience=20

    elif model_name=="ResNet":
        input_dim = input_size
        output_dim = output_size
        dim_first = 64
        dim_second = 32


        inputs = Input(shape=(input_dim,))
        # First layer
        x_in = Dense(dim_second, activation='linear')(inputs)
        # First block
        x = tf.keras.layers.LayerNormalization()(x_in)
        x = Dense(dim_first, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = Dense(dim_second, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x_in = x_in+x

        # Second block
        x = tf.keras.layers.LayerNormalization()(x_in)
        x = Dense(dim_first, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = Dense(dim_second, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x_in = x_in+x

        # Third block
        x = tf.keras.layers.LayerNormalization()(x_in)
        x = Dense(dim_first, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = Dense(dim_second, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x_in = x_in+x

        # Fourth block
        x = tf.keras.layers.LayerNormalization()(x_in)
        x = Dense(dim_first, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = Dense(dim_second, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x_in = x_in+x

        # Fifth block
        x = tf.keras.layers.LayerNormalization()(x_in)
        x = Dense(dim_first, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = Dense(dim_second, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x_in = x_in+x

        # Sixth block
        x = tf.keras.layers.LayerNormalization()(x_in)
        x = Dense(dim_first, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = Dense(dim_second, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x_in = x_in+x

        # Seventh block
        x = tf.keras.layers.LayerNormalization()(x_in)
        x = Dense(dim_first, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = Dense(dim_second, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x_in = x_in+x

        # 8th block
        x = tf.keras.layers.LayerNormalization()(x_in)
        x = Dense(dim_first, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = Dense(dim_second, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x_in = x_in+x

        # 9th block
        x = tf.keras.layers.LayerNormalization()(x_in)
        x = Dense(dim_first, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = Dense(dim_second, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x_in = x_in+x

        # 10th block
        x = tf.keras.layers.LayerNormalization()(x_in)
        x = Dense(dim_first, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = Dense(dim_second, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x_in = x_in+x


        # Head 
        x = tf.keras.layers.LayerNormalization()(x_in)
        outputs = Dense(output_dim, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="ResNet")        
        optimizer = AdamW(learning_rate=0.0001,weight_decay=0.00001) # Adam(learning_rate=3e-4, decay=1e-6)
        
    else:
        model = Sequential()
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))

        # add linear output layer
        model.add(Dense(output_size, activation='linear'))

        # name model
        model._name = "Base"

        optimizer = Adam()

    return model, optimizer