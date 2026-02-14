Telemetry Anomaly Detection AI

Telemetry Anomaly Detection AI is a deep learning–based anomaly detection system designed to identify mission-critical irregularities in multivariate satellite telemetry data. The project utilizes a Multivariate LSTM Autoencoder to analyze 25 simultaneous telemetry channels from NASA’s SMAP (Soil Moisture Active Passive) satellite and detect abnormal subsystem behavior that traditional threshold-based monitoring systems may fail to capture.

Project Overview:

Satellite telemetry consists of continuous, high-dimensional time-series sensor data such as voltage, temperature, current, and subsystem measurements. Detecting faults in these systems is challenging due to complex temporal dependencies and cross-sensor relationships. Static threshold systems often fail to capture subtle deviations in system behavior. This project implements an unsupervised deep learning approach where anomalies are detected using reconstruction error from a trained LSTM Autoencoder.

Key Features:

- Multivariate Analysis: Simultaneously monitors 25 telemetry channels.
- Temporal Memory: Uses Long Short-Term Memory (LSTM) layers to learn time-series patterns.
- Automated Flagging: Implements a statistical thresholding system based on Mean Absolute Error (MAE).
- End-to-End Pipeline: Handles data acquisition via Kaggle API, preprocessing, model training, and visualization.

Tech Stack:

Language: Python 3.x  
Deep Learning: TensorFlow, Keras  
Data Processing: NumPy, Scikit-learn  
Visualization: Matplotlib, Seaborn  
Environment: Google Colab (GPU Accelerated)

Methodology:

Data Preprocessing: Telemetry signals are normalized between [0, 1] using MinMaxScaler to ensure uniform feature weighting. The data is transformed into 50-timestep sliding windows to provide the model with temporal context.

Model Architecture: LSTM Autoencoder  
The model follows an Encoder-Decoder structure. The Encoder compresses the 50x25 input tensor into a lower-dimensional latent representation. A RepeatVector layer bridges the compressed summary to the Decoder. The Decoder reconstructs the original 50x25 sequence from the latent embedding.

Anomaly Logic:  
The system is trained exclusively on normal operational data. When presented with anomalous data, the model fails to reconstruct the signal accurately, resulting in increased reconstruction error.

Reconstruction Error:
MAE = 1/n ∑ |X_actual - X_predicted|

Thresholding: Any window where MAE exceeds the defined threshold is flagged as a potential anomaly.

Results:

The model successfully identifies deviations in the NASA SMAP dataset. The final output generates a visualization where anomalies are automatically highlighted, providing a clear monitoring-style dashboard for telemetry analysis.

How to Run:

1. Clone this repository.
2. Upload your kaggle.json file.
3. Run the notebook to download the NASA dataset and train the model.
4. Generate the anomaly detection visualization.

Developed by Abhay R S, Third Year B.Tech, Computer Science (Artificial Intelligence and Machine Learning) Engineering
