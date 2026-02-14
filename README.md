SatHealth-AI: Multivariate Satellite Telemetry Anomaly Detection: SatHealth-AI is a Deep Learning-based health monitoring system designed to detect mission-critical anomalies in satellite subsystems. By utilizing a Multivariate LSTM Autoencoder, the system analyzes 25 simultaneous telemetry channels from NASA’s SMAP (Soil Moisture Active Passive) satellite to identify sensor irregularities that traditional threshold-based systems might miss.

Project Overview: Satellite telemetry is a continuous stream of high-dimensional sensor data (voltage, temperature, current, etc.). Detecting faults in these systems is challenging due to the complex temporal dependencies between different sensors. This project implements an unsupervised learning approach to flag anomalies based on Reconstruction Error.

Key Features: -Multivariate Analysis: Simultaneously monitors 25 telemetry channels. -Temporal Memory: Uses Long Short-Term Memory (LSTM) layers to learn time-series patterns. -Automated Flagging: Implements a dynamic thresholding system based on Mean Absolute Error (MAE). -End-to-End Pipeline: Handles data acquisition via Kaggle API, preprocessing, model training, and visualization.

Tech Stack: Language: Python 3.x Deep Learning: TensorFlow, Keras Data Processing: NumPy, Pandas, Scikit-learn Visualization: Matplotlib, Seaborn Environment: Google Colab (T4 GPU Accelerated)

Methodology:

Data Preprocessing: Normalization: Telemetry signals are scaled between [0, 1] using MinMaxScaler to ensure uniform feature weight. Sliding Window: Data is transformed into 50-minute sequences (timesteps) to provide the model with temporal context.

Model Architecture: LSTM Autoencoder The model follows an Encoder-Decoder structure: Encoder: Compresses the 50x25 input tensor into a lower-dimensional latent representation. RepeatVector: Bridges the compressed summary to the decoder. Decoder: Attempts to reconstruct the original 50x25 sequence from the compressed summary.

Anomaly Logic The system is trained exclusively on Normal operational data. When presented with Anomalous data (faults), the model fails to reconstruct the signal accurately. Reconstruction Error: M AE= 1/n ∑|Xactual - Xpred| Thresholding: Any window where 
M
A
E
>
T
h
r
e
s
h
o
l
d
 is flagged as a potential satellite fault.

Results The model successfully identifies deviations in the NASA SMAP dataset. The final output generates a visualization where anomalies are automatically highlighted in red, providing a clear dashboard for mission control operators. (Note: See results.png above for the visualization of detected anomalies in the telemetry stream.)

How to Run 1.Clone this repository. 2.Upload your kaggle.json to the environment. 3.Run the Jupyter/Colab notebook to download the NASA dataset and train the model. 4.Save the final plot as results.png to display it in this README.

Developed by Abhay R S,Third Year B.Tech,Computer Science(Artificial Intelligence and Machine Learning) Engineering
