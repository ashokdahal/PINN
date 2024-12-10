from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, Model,Input
import numpy as np
FTP=tf.keras.metrics.TruePositives()
FFP=tf.keras.metrics.FalsePositives()
FFN=tf.keras.metrics.FalseNegatives()

class DisplacementLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DisplacementLayer, self).__init__(**kwargs)

    def call(self, inputs):
        #https://www.sciencedirect.com/science/article/pii/S0266352X24000387
        # Unpack input
        cohesion_t, friction_angle, slope,pga = inputs[0], inputs[1], inputs[2],inputs[3]
        slope*=0.017453292519943295
        
        pga*=10.0 #to convert to g units
        cohesion_t*=1000.0 #kpa to pa

        # Calculate shear strength using Mohr-Coulomb criterion
        cohesion_t = tf.expand_dims(cohesion_t, 1, name=None)
        friction_angle = tf.expand_dims(friction_angle, 1, name=None)

        safety_factor = (cohesion_t*(1/(2300*9.81*tf.math.sin(slope)))) + ( tf.math.tan(friction_angle)/tf.math.tan(slope))
        
        # safety_factor = tf.nn.relu(safety_factor)
        safety_factor = tf.clip_by_value(safety_factor, 1.2, 15.0)

        ac = (safety_factor-1)*9.81*tf.math.sin(slope)
        
        acpg=ac/pga

        acpg = tf.clip_by_value(acpg, 0.001, 0.999)

        powcomp = tf.math.pow((1-acpg),2.53)*tf.math.pow(acpg,-1.438)
        logds = 0.251+tf.math.log(powcomp)+0.5

        return tf.math.exp(logds)
    
class LandslideModel():
    def __init__(self):
        self.depth=12

    # Custom activation function
    def landslide_activation(self,x):
        # return 1/(1+tf.exp(5-x))
        return x-5.0
    def cohesion_activation(self,x):
        # return 1 / (1 + tf.exp(x- 1))
        return layers.Activation('relu')(x)
    def friction_activation(self,x):
       # turn this into 0-1 radians
       return layers.Activation('sigmoid')(x)#tf.clip_by_value(x, 0.15, 0.75)

    # Define the Mohr-Coulomb safety factor calculation as a custom layer

    def getclassificationModel(self,all_inputs, encoded_features ,in_num=17,out_num=1):

        all_features = tf.keras.layers.concatenate(encoded_features)
        features_only=all_features
        slope=all_inputs[4]
        pga=all_inputs[7]
            
        x=layers.Dense(units=64,name=f'Sus_0',kernel_initializer='random_normal',bias_initializer='random_normal')(features_only)
        for i in range(1,self.depth+1):
            x=layers.Dense(units=64,name=f'Sus_{str(i)}',kernel_initializer='random_normal',bias_initializer='random_normal')(x)
            x= layers.BatchNormalization()(x)
            x=layers.Activation('relu')(x)

        x=layers.Dense(units=2,activation='relu',name='geotechnical_param')(x)
        coh=layers.Lambda(self.cohesion_activation,name='cohesion')(x[...,0])
        ifi=layers.Lambda(self.friction_activation,name='internalFriction')(x[...,1])

        ds= DisplacementLayer()([coh,ifi,slope,pga])
        ds= layers.Activation('relu')(ds)
        
        sus = layers.Lambda(self.landslide_activation)(ds)
        sus= layers.Activation('sigmoid')(sus)
        
        self.model = Model(inputs=all_inputs, outputs=sus)

    def getOptimizer(self,opt=tf.keras.optimizers.Adam,lr=1e-4,decay_steps=10000,decay_rate=0.9):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,decay_steps=decay_steps,decay_rate=decay_rate)
        self.optimizer = opt(learning_rate=lr_schedule)
   
    def compileModel(self,weights=None):
        self.model.compile(optimizer=self.optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[0,1], threshold=0.5),tf.keras.metrics.AUC(),tf.keras.metrics.BinaryAccuracy()])