🩺 Skin_disease_detection_2025

This project focuses on developing a deep learning based model for the classification of various skin conditions using pretrained Convulutional Neural Networks(CNN) specifically EfficientNetB0 along with a custom classifier for Feature Extraction. The primary goal is to assist dermatologists and healthcare professionals by providing AI-driven efficient diagonostic tool for identifying the disease.🤖
Following 15 classes were diagonosed:
    🌟 Light Diseases and Disorders of Pigmentation
    😷 Acne and Rosacea Photos
    🧬 Systemic Disease
    🩸 Vascular Tumors
    🌿 Atopic Dermatitis Photos
    🩹 Bullous Disease Photos
    🍄 Tinea, Ringworm, Candidiasis, and other Fungal Infections
    🔴 Psoriasis Pictures, Lichen Planus, and Related Diseases
    🧪 Melanoma, Skin Cancer, Nevi, and Moles
    🐛 Scabies, Lyme Disease, and other Infestations and Bites
    🧴 Eczema Photos
    ⚪ Seborrheic Keratoses and other Benign Tumors
    🔬 Actinic Keratosis, Basal Cell Carcinoma, and other Malignant Lesions
    🩹 Vasculitis Photos
    🦠 Cellulitis, Impetigo, and other Bacterial Infections
  
** 🔄 Workflow **
Steps in the Pipeline:
1) 🗂 Data Preprocessing
      🧹 Cleaning raw data
      ❓ Handling missing values
      🔄 Converting images from BGR → RGB
2) 🔍 Feature Extraction
      ✨ Extracting meaningful features from input images
      🏋️ Model Training
      Base Model: EfficientNetB0 (pretrained on ImageNet)
      Custom Classifier: Dense layers for final classification
3) 📊 Validation
    🔁 K-Fold Cross Validation for robust evaluation
4) 📈 Performance Evaluation
    ✅ Accuracy
    🎯 Precision
    🔁 Recall
    🏆 F1-Score
   
🧑‍💻 Tech Stack
      Python 3.10
      TensorFlow / Keras (EfficientNetB0, custom classifier)
      NumPy, Pandas, scikit-learn, OpenCV
      Matplotlib, Seaborn

📊 Performance / Results
✅ Training Accuracy: 95.85%
✅ Testing Accuracy: 55.92%

🤝 Contributors:
      Heli Mevada (Team Leader)
      Vishwa Parmar
      Krisha Patel
      Drashti Raiyani
