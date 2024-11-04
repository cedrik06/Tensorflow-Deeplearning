import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from tensorflow.keras import layers, Model
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import base64




# Model ne kadar eğitildiğine dair bilgi
# Excel tarih aralık heatmap tarih aralık
#



# Model oluşturma fonksiyonu
class AnomalyDetecter(Model):
    def __init__(self) -> None:
        super(AnomalyDetecter, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(1,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

anomaly_model = AnomalyDetecter()
anomaly_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')    

    

# Streamlit Arayüzü
st.title("IMS ANOMALI TESPIT MODELI")
st.write("Excel dosyanızı yükleyin ve modeli eğitin veya tahmin yapın.")

# Kullanıcıdan dosya yüklemesini isteyin





uploaded_file_2 = st.file_uploader("Anomali Tespiti İçin Dosya Yükleyin")

uploaded_file = st.file_uploader("Modeli Eğitmek İçin Dosya Yükleyin ", type=["xlsx", "csv"])


# Yeni model oluştur


# Dosya yüklendiğinde işlemler
if uploaded_file:
    excel_Data_educate = pd.read_excel(uploaded_file)
    excel_Data_educate.to_csv("uploaded_file_educate.csv",  
                        index = None, 
                        header=True)
    Educate_data = pd.DataFrame(pd.read_csv("uploaded_file_educate.csv",decimal=",", delimiter=','))
    Educate_z_axis = Educate_data.iloc[:,6:7].values
    scaler = StandardScaler()
    Educate_z_axis_scaled = scaler.fit_transform(Educate_z_axis)

    # Model eğitme düğmesi
    if st.button("Modeli Eğit"):
        history = anomaly_model.fit(Educate_z_axis_scaled, Educate_z_axis_scaled, epochs=50, batch_size=128)
        st.success("Model başarıyla eğitildi.")
        anomaly_model.save('anomali_model.h5')

    # Tahmin yapma düğmesi
if uploaded_file_2:
    
        excel_Data = pd.read_excel(uploaded_file_2)
        excel_Data.to_csv("uploaded_file_predict.csv",  
                        index = None, 
                        header=True)
        new_data = pd.DataFrame(pd.read_csv("uploaded_file_predict.csv",decimal=",", delimiter=','))
        new_z_axis = new_data.iloc[:,6:7].values
        scaler = StandardScaler()
        new_z_axis_data_scaled = scaler.fit_transform(new_z_axis)
    
        data_scaled_predict = anomaly_model.predict(new_z_axis_data_scaled)
        mse = np.mean(np.power(new_z_axis_data_scaled - data_scaled_predict, 2), axis=1)
        threshold = np.percentile(mse, 97)
        anomalies = mse > threshold

        anomaly_indices_z = new_data.index[anomalies == True].tolist()

        anomaly_points_z =(new_data.iloc[anomaly_indices_z])
     
        # Haritayı oluştur
        
        m = folium.Map(location=[40.991826, 29.036553], zoom_start=15)
        anomaly_coordinates = [[row['latitude'], row['longitude']] for index, row in anomaly_points_z.iterrows()]
        HeatMap(anomaly_coordinates).add_to(m)


        # Haritayı kaydet ve göster
        m.save("heatmap_Z.html")
        heatmap_name = st.text_input("Heatmap'e İsim Veriniz")
        st.write("Anomali haritası oluşturuldu. Aşağıdaki bağlantıdan haritayı görebilirsiniz.")
        

        
        with open("heatmap_Z.html", "rb") as file:
            btn = st.download_button(
                label="Haritayı İndir",
                data=file,
                file_name=f"{heatmap_name}",
                mime="text/html"
            )
        
            
                              ####### IMS YAPAY ZEKA MODELİ TMS R&D  ##########
 