Overview
This project aims to recognize emotions in speech using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. The dataset includes speech and song recordings of actors expressing different emotions. The goal is to classify emotions such as neutral, calm, happy, sad, angry, fear, disgust, and surprise from the given audio files using various machine learning techniques.

The project involves extracting features from the audio data (such as MFCCs, Chroma, Zero Crossing Rate, etc.) and training various classifiers to predict the emotions associated with each audio sample.


Project Workflow
- Data Preprocessing: The audio files from the RAVDESS dataset are loaded and their metadata is parsed to extract the emotion labels.
The dataset includes several emotional categories like neutral, happy, angry, etc.

- Audio Feature Extraction: Waveform and Spectrogram Visualization: Each audio sample's waveform and spectrogram are plotted to visualize the audio content and understand its structure.

- Audio Augmentation: Techniques like adding noise, changing speed, shifting audio, and pitch shifting are applied to augment the data.
Feature Extraction: Features such as MFCCs, Zero Crossing Rate, Chroma, and Mel Spectrogram are extracted from each audio clip to be used as input to machine learning models.

- Model Training and Evaluation:
Several classifiers are trained using the extracted features, including:
Random Forest Classifier
Gradient Boosting Classifier
XGBoost Classifier
Logistic Regression
Support Vector Classifier (SVC)
K-Nearest Neighbors (KNN)
The models are evaluated based on their accuracy on the test set.

- Results: The project evaluates the performance of different classifiers and shows their accuracies on both the training and testing datasets. The best-performing model is chosen based on these results.


Requirements:
- Python 3.x
- Librosa: For audio processing.
- Keras: For building machine learning models.
- Scikit-learn: For machine learning algorithms and metrics.
- Matplotlib: For visualizations.
- Seaborn: For advanced visualizations.


How to Run:
- Download the RAVDESS dataset from the official website and place it in the directory audio_speech_actors_01-24/.
- Run the Jupyter notebook or Python script to start the feature extraction and model training process.
- Explore the results from different models and compare their performance.


License: This project is licensed under the MIT License. 

