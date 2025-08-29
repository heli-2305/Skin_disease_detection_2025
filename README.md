🩺 Skin Disease Detection 2025
This project focuses on developing a deep learning model for the classification of various skin conditions using pretrained Convolutional Neural Networks (CNNs), specifically EfficientNetB0, along with a custom classifier for feature extraction. The primary goal is to assist dermatologists and healthcare professionals by providing an AI-driven, efficient diagnostic tool for identifying skin diseases. 🤖
The following classes are diagnosed:
        - 🌟 Light Diseases and Disorders of Pigmentation<br>
        - 😷 Acne and Rosacea Photos<br>
        - 🧬 Systemic Disease<br>
        - 🩸 Vascular Tumors<br>
        - 🌿 Atopic Dermatitis Photos<br>
        - 🩹 Bullous Disease Photos<br>
        - 🍄 Tinea, Ringworm, Candidiasis, and other Fungal Infections<br>
        - 🔴 Psoriasis Pictures, Lichen Planus, and Related Diseases<br>
        - 🧪 Melanoma, Skin Cancer, Nevi, and Moles<br>
        - 🐛 Scabies, Lyme Disease, and other Infestations and Bites<br>
        - 🧴 Eczema Photos<br>
        - ⚪ Seborrheic Keratoses and other Benign Tumors<br>
        - 🔬 Actinic Keratosis, Basal Cell Carcinoma, and other Malignant Lesions<br>
        - 🩹 Vasculitis Photos<br>
        - 🦠 Cellulitis, Impetigo, and other Bacterial Infections<br>

🔄 Workflow<br>
Steps in the Pipeline:<br>
1) 🗂 Data Preprocessing<br>
        🧹 Cleaning raw data<br>
        ❓ Handling missing values<br>
        🔄 Converting images from BGR → RGB<br>
2) 🔍 Feature Extraction<br>
        ✨ Extracting meaningful features from input images<br>
3) 🏋️ Model Training<br>
        Base Model: EfficientNetB0 (pretrained on ImageNet)<br>
        Custom Classifier: Dense layers for final classification<br>
4) 📊 Validation<br>
        🔁 K-Fold Cross Validation for robust evaluation<br>
5) 📈 Performance Evaluation<br>
           ✅ Accuracy<br>
           🎯 Precision<br>
           🔁 Recall<br>
           🏆 F1-Score<br>

🧑‍💻 Tech Stack<br>
        🐍 Python 3.10<br>
        🤖 TensorFlow / Keras (EfficientNetB0, custom classifier)<br>
        🔢 NumPy, Pandas, scikit-learn, OpenCV<br>
        📊 Matplotlib, Seaborn<br>
📊 Performance / Results<br>
        ✅ Training Accuracy: 95.85%<br>
        ✅ Testing Accuracy: 55.92%<br>
🤝 Contributors<br>
        Heli Mevada (Team Leader)<br>
        Vishwa Parmar<br>
        Krisha Patel<br>
        Drashti Raiyani<br>
