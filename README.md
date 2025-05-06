

# BCI Deception Detection System

This repository contains code for a **Brain-Computer Interface (BCI)-based deception detection system**. The system integrates multiple modalities such as **EEG (electroencephalography)**, **facial expressions**, **audio signals**, **text** data, and other physiological signals to detect deception in real-time. The system uses deep learning techniques, including Transformer-based models, for multi-modal data processing and fusion to assess whether a statement is truthful or deceptive.

## Project Overview

The goal of this project is to create an efficient and accurate lie detection system using BCI and other physiological signals. By combining EEG signals, facial expressions, speech features, and text analysis, the system attempts to detect deception with a high level of precision.

### Key Features:

* **Multi-modal Input**: Integrates EEG, facial expression, audio features (MFCC), text, and other physiological signals such as heart rate and blood oxygen levels.
* **Real-time Detection**: Uses real-time data processing to assess whether a statement is true or false.
* **Deep Learning-based Fusion Model**: Combines features from multiple modalities using deep neural networks (e.g., Transformers) to make a final decision.
* **Data Preprocessing**: Preprocessing steps for each type of input data, including standardization, dimensionality reduction, and feature extraction.

## Getting Started

### Prerequisites

To run this project, you'll need the following libraries installed:

* Python 3.x
* TensorFlow (for deep learning models)
* Keras (for building and training neural networks)
* NumPy (for numerical operations)
* pandas (for data manipulation)
* scikit-learn (for preprocessing and machine learning utilities)
* librosa (for audio feature extraction)
* OpenCV/Dlib (for facial expression detection)

You can install the necessary libraries using `pip`:

```bash
pip install tensorflow keras numpy pandas scikit-learn librosa opencv-python dlib
```

### Clone the Repository

To get started with the project, clone this repository:

```bash
git clone https://github.com/yourusername/BCI_Deception_Detection.git
cd BCI_Deception_Detection
```

### Data Preparation

Before running the model, ensure that you have your **EEG data**, **audio features (MFCC)**, **facial expression data**, and **text features** properly preprocessed and stored in CSV files or any other suitable format.

#### Example Data Columns:

* **EEG Data**: `EEG_Channel_1`, `EEG_Channel_2`, ..., `EEG_Channel_7`
* **MFCC Features**: `MFCC_1`, `MFCC_2`, ..., `MFCC_13`
* **Facial Expression Features**: `Left Eye X`, `Mouth X`, `Smile Intensity`, `Frown Intensity`, etc.
* **Text Features**: BERT embeddings or any other textual features
* **Physiological Data**: `Heart Rate`, `Blood Oxygen`, etc.
* **Labels**: `Label` (1 = True, 0 = False)

### Running the System

After preparing your data and dependencies, you can run the BCI-based deception detection system:

```bash
python trans.py
```

This will load your data, preprocess it, train the model, and evaluate the performance on the test dataset.

### Training the Model

If you are training the model from scratch, the training process involves:

1. Loading and splitting the dataset.
2. Preprocessing the input features.
3. Defining and training the multi-modal deep learning model (using Transformer-based architecture).
4. Evaluating the performance using metrics like accuracy, loss, and confusion matrix.

```python
# Example of training a model
model.fit(
    [X_train_eeg, X_train_face, X_train_mfcc, X_train_text],  # Pass in feature arrays
    y_train,  # Labels
    epochs=50,
    batch_size=32,
    validation_data=([X_test_eeg, X_test_face, X_test_mfcc, X_test_text], y_test)  # Validation data
)
```

### Model Architecture

The system uses a **multi-modal fusion model** consisting of:

1. **EEG-based model**: Using **LSTM** and **Transformer** layers to process EEG signals.
2. **Facial Expression Model**: A simple **Dense Layer** model to process facial features.
3. **Audio-based model**: Uses **Transformer** layers to process MFCC features.
4. **Text-based model**: Uses **Transformer** layers to process BERT-like textual features.
5. **Fusion Model**: Combines all individual modality outputs and passes through several **Dense Layers** for final classification.

### Evaluation

The model’s performance is evaluated based on the following metrics:

* **Accuracy**
* **Confusion Matrix** (True Positives, False Positives, True Negatives, False Negatives)
* **Precision and Recall** (for detecting deceptive statements)

### Example of Model Evaluation:

```python
loss, accuracy = model.evaluate(
    [X_test_eeg, X_test_face, X_test_mfcc, X_test_text],  # Testing data
    y_test
)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")
```

## Project Structure

```plaintext
BCI_Deception_Detection/
│
├── data/                   # Contains raw data files (CSV or other formats)
├── models/                 # Saved models and trained weights
├── trans.py                # Main script to run the model (train/evaluate)
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Contributing

If you would like to contribute to this project, feel free to fork the repository and make pull requests. Suggestions and improvements are always welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
