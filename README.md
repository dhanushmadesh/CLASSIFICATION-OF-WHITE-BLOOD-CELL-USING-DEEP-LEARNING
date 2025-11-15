# 🧬 White Blood Cell Classification Using Deep Learning

A deep learning–based system to classify **four types of white blood cells** from microscopic blood smear images:

- **Eosinophil**
- **Lymphocyte**
- **Monocyte**
- **Neutrophil**

This project trains a custom CNN model and deploys it through a **Streamlit web app**, enabling users to upload a WBC image and instantly get the predicted cell type along with its biological role.

---

# 📂 Dataset Used

🔗 **Dataset Link:**  
https://www.kaggle.com/datasets/paultimothymooney/blood-cells

The dataset contains **12,500+ augmented images** distributed across 4 classes, with balanced sample sizes of ~3000 per class.

Each image is a high-quality microscopic view of a single white blood cell — making it ideal for deep learning classification tasks.


# ⭐ Final Model Performance

### ✔ **Final Test Accuracy: 86.01%**

<img src="FINAL TEST ACCURACY .png" width="350"/>

A solid accuracy for a traditional CNN model trained on medical-image data.

---

# 📈 Training Curves

### **Accuracy & Loss Graphs**

<p align="center">
  <img src="ACCURACY_AND_LOSS.png" width="800"/>
</p>

- Smooth convergence  
- Low overfitting  
- Good generalization  
- Validation accuracy closely follows training accuracy  

---

# 🧪 Classification Report

<p align="center">
  <img src="CLASSIFICATION REPORT .png" width="700"/>
</p>

### 🔍 Highlights

| Cell Type | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| EOSINOPHIL | 0.90 | 0.84 | 0.87 |
| LYMPHOCYTE | 0.99 | 0.94 | 0.97 |
| MONOCYTE | 0.98 | 0.75 | 0.85 |
| NEUTROPHIL | 0.67 | 0.91 | 0.77 |

- **Lymphocytes**: Highest overall performance  
- **Neutrophils**: High recall but lower precision  
- Very strong overall macro F1 score of **0.86**

---

# 🔍 Confusion Matrix

<p align="center">
  <img src="CONFUSION_MATRIX.png" width="500"/>
</p>

This visualizes prediction performance across all four classes and highlights common misclassifications.

---

# 🏗️ Model Architecture

The CNN model includes:

- **4 Convolution Blocks**  
  - Conv2D  
  - LeakyReLU  
  - BatchNormalization  
  - MaxPooling2D  

- **Dense Layers**  
  - Fully connected layers (512 → 256 → 128)  
  - Dropout (0.5 / 0.4 / 0.3)  

- **Output Layer**  
  - Softmax activation for 4-class prediction  

### Training Features

- Class weights to handle imbalance  
- Data augmentation  
- Early stopping  
- Model checkpointing  
- Adam optimizer (1e-4 LR)  
- 50 epochs  

---

# 🎛️ Streamlit Web App

The web app:

- Accepts JPG/PNG microscopic images  
- Preprocesses automatically  
- Predicts the WBC type  
- Shows model confidence  
- Displays the biological description of each cell  
- Provides a clean visual layout  


