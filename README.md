# Sign Language Detection using Deep Learning

## 📌 Overview
This project is a *real-time Sign Language Detection system* using *Convolutional Neural Networks (CNNs)*. The model is trained on American Sign Language (ASL) images and can recognize hand gestures using a webcam.

## 🚀 Features
- *Trains a CNN model using TensorFlow & Keras*
- *Real-time sign language detection using OpenCV*
- *Supports 30 different sign language gestures*
- *Automatically resizes and normalizes input frames*
- *Deployable in Google Colab and local machines*

## 📂 Dataset
The dataset consists of images of different sign language gestures stored in *asl_signs/asl_symbols/*. Each folder represents a unique gesture class.

## 🛠 Installation
### *Step 1: Install Dependencies*
Ensure you have Python 3.11 installed. Then, install the required libraries:
bash
pip install tensorflow opencv-python matplotlib numpy google-colab


### *Step 2: Clone the Repository*
bash
git clone [https://github.com/Aashrith-555/Sign-Language-Detection.git]


### *Step 3: Train the Model (Google Colab or Local Machine)*
Run the following script to train the model:
bash
python train_model.py

This will train a CNN model and save it as sign_language_model.h5.

### *Step 4: Test in Real-Time*
To run the model using your webcam:
bash
python sign_language_detection.py

Press q to exit the webcam window.

## 🏗 Project Structure

Sign-Language-Detection/
│── train_model.py  # Training script
│── sign_language_detection.py  # Real-time detection
│── sign_language_model.h5  # Trained model
│── dataset/  # Folder containing sign language images
│── README.md  # Project documentation


## 🔍 Model Architecture
The CNN model consists of:
- *3 Convolutional Layers* with ReLU activation
- *MaxPooling Layers* to reduce spatial dimensions
- *Flatten Layer* to convert feature maps to a 1D vector
- *Dense Layers* with Dropout for classification

## 📸 Live Demo
To see the model in action, run:
bash
python sign_language_detection.py

Make sure your webcam is enabled.

## 🤝 Contribution
Feel free to fork this repository, improve the model, and submit a Pull Request.

## 📜 License
This project is licensed under the MIT License.

---  
GitHub: [Aashrith-555](https://github.com/Aashrith-555/Sign-Language-Detection.git)
