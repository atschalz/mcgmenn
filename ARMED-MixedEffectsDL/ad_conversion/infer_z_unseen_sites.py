import pickle
import tensorflow as tf
import tensorflow.keras.layers as tkl

def infer_z_nn(arrXTrain, arrZTrain, arrXVal, arrZVal, arrXUnseen, verbose=0):
    
    xin = tkl.Input(arrXTrain.shape[1])
    x = tkl.Dense(8, activation='relu')(xin)
    x = tkl.Dense(8, activation='relu')(x)
    x = tkl.Dense(4, activation='relu')(x)
    x = tkl.Dense(arrZTrain.shape[1])(x)
    xout = tkl.Softmax()(x)

    model = tf.keras.Model(xin, xout)
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(arrXTrain, arrZTrain,
              epochs=50,
              verbose=verbose)
    
    return model.predict(arrXUnseen)