# Brain-Tumor-MRI-Classification-with-Multi-Head-Self-Attention-and-Hyperparameter-Tuning

#Brain Tumor MRI Dataset Classification
This repository contains a deep learning model for classifying brain tumor MRI images into four categories: glioma tumor, meningioma tumor, pituitary tumor, and healthy brain tissue. The model is built using TensorFlow and Keras, with a combination of convolutional and recurrent neural networks.

# Dataset
The dataset consists of brain MRI images collected from various sources, including images with different tumor types and images of healthy brain tissue. The dataset is available for download from Kaggle.

Usage
Clone the repository:

git clone https://github.com/your-username/brain-tumor-classification.git
Install the required dependencies:

pip install -r requirements.txt
Download the dataset from Kaggle and place it in the /data directory.

Run the Jupyter Notebook or Python script to train the model:
python train_model.py

# Evaluate the trained model on the test dataset:
python evaluate_model.py
Explore the trained model's performance and visualizations in the notebook or script outputs.

# Model Architecture
The model architecture is based on a combination of convolutional neural networks (CNNs) and recurrent neural networks (RNNs), with attention mechanisms incorporated to enhance feature extraction and classification accuracy. The model utilizes pre-trained VGG16 for feature extraction followed by LSTM and GRU layers for sequential learning.

# Hyperparameter Tuning
Hyperparameters such as LSTM units, batch size, and dropout rates are tuned using the Keras Tuner library. Random search is employed to find the optimal combination of hyperparameters.

# Results
The final trained model achieves high accuracy in classifying brain tumor MRI images, with performance evaluated on both validation and test datasets. Visualizations of training history, hyperparameter tuning results, and model performance metrics are provided.
