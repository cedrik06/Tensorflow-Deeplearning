import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('C:/Users/furka/OneDrive/Desktop/Tensorflow DeepLearning/DeepLearning/tms_acceleration.csv', decimal=",", delimiter=';')


raw_data = data.values
data.head()
z_values  = data.iloc[:,2:3].values



scaler = StandardScaler()
data_scaled = scaler.fit_transform(z_values)




class AnomalyDetecter(Model):
    def __init__(self) -> None:
        super(AnomalyDetecter, self).__init__()
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(1,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu')  # Bottleneck boyutunu artırdık
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')  # Çıkış boyutunu 1 tuttuk
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

anomaly_model =AnomalyDetecter()

anomaly_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='mse')  # Mean Squared Error, anomali tespiti için uygun olabilir



# Model eğitimi
history = anomaly_model.fit(data_scaled, data_scaled,
                            epochs=50,
                            batch_size=128,
                            validation_data=(data_scaled, data_scaled),
                            )




plt.plot(history.history["loss"], label = "Training loss")



data_scaled_predict = anomaly_model.predict(data_scaled)
mse = np.mean(np.power(data_scaled - data_scaled_predict, 2), axis=1)
threshold = np.percentile(mse, 95)
anomalies = mse > threshold



anomaly_indices_z = data.index[anomalies == True].tolist()

anomaly_points_z = data.iloc[anomaly_indices_z]





plt.figure(figsize=(20, 12))
plt.scatter(anomaly_points_z.index, anomaly_points_z["max_acc_z"], color='red', label='Anomali')
plt.xlabel('Index')
plt.ylabel('Max Acc Z')
plt.title('Z Axis Anomoly Datas')
plt.legend()
plt.grid(True)
plt.show()
