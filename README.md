# 🖼️ Face Detection Classifier (Python + TensorFlow/Keras)

A deep learning **binary classifier** that detects whether an image contains a face or not.  
Trained on **Labeled Faces in the Wild (LFW)** for positive samples (faces) and **CIFAR‑10** for negative samples (non-faces).  
Built using **Python**, **OpenCV**, **TensorFlow/Keras**, and **scikit-learn**.

---

## 🚀 Features
- Downloads and preprocesses the **LFW** face dataset from scikit-learn.
- Loads **CIFAR‑10** dataset (via Keras) as non-face examples.
- Image preprocessing: grayscale conversion, resizing (50×37), pixel normalization.
- Splits dataset into **train / validation / test** sets with stratification.
- Convolutional Neural Network (CNN) architecture for binary classification.
- Evaluation with accuracy, precision, recall, F1-score, and confusion matrix.
- Predicts on **new images** with confidence values.

---

## 📊 Example Results
**Test Set Performance:**
Accuracy: 1.000
Precision: 1.000
Recall: 1.000
F1-score: 1.000

Confusion matrix heatmap example:

![Confusion Matrix] (<img width="387" height="393" alt="confusionmatrix" src="https://github.com/user-attachments/assets/40a0798e-cbb9-4bb7-ac99-f84f71ed6dfb" />)

---

## 🗂️ Project Structure
📁 face-detection-classifier
│── face_detection_notebook.ipynb # Full Jupyter Notebook with step-by-step code
│── requirements.txt # Python dependencies
│── README.md # Project overview
│── images/ # Sample images and results


---

## 🛠️ Installation
1. **Clone this repository**:
https://github.com/parvathykrishna05/face-detection-ml/tree/main
cd face-detection-classifier
2. **Install required packages**:
pip install -r requirements.txt
> Ensure you have Python 3.8+ and pip installed.

---

## 📦 Dependencies
- Python 3.x
- NumPy  
- Pandas  
- scikit-learn  
- TensorFlow / Keras  
- OpenCV  
- Matplotlib  
- Seaborn  

You can install them with:
pip install numpy pandas scikit-learn tensorflow opencv-python matplotlib seaborn

---

## 🖥️ Usage

### 1️⃣ Train the Model
Open the Jupyter notebook and run all cells:
jupyter notebook face_detection_notebook.ipynb

### 2️⃣ Predict on a New Image
from detect import preprocess_image
input_image = preprocess_image('path_to_image.jpg')
prob = model.predict(input_image)

if prob > 0.5:
print(f"Face detected with confidence {prob:.2f}")
else:
print(f"No face detected, confidence {1-prob:.2f}")

---

## 🧠 Model Architecture
Conv2D(32, 3x3, ReLU) → MaxPooling(2x2)
Conv2D(64, 3x3, ReLU) → MaxPooling(2x2)
Conv2D(128, 3x3, ReLU) → MaxPooling(2x2)
Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)

---

## 📌 Future Improvements
- Add **data augmentation** for more robust detection.
- Test on more challenging datasets (e.g., CelebA, random internet images).
- Deploy model as a real-time webcam application with OpenCV.

---

## 📜 License
This project is licensed under the MIT License.

---

## ✨ Acknowledgements
- **[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)**
- **[CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)**
- TensorFlow/Keras and scikit-learn teams
