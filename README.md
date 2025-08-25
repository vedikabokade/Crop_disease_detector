# 🌱 Crop Disease Detection using CNN and Django

## 📌 Project Overview
This project is a **Django-based web application** integrated with a **custom Convolutional Neural Network (CNN) model** for detecting crop diseases.  
Users can upload an image of a plant leaf, and the trained CNN model predicts whether the crop is healthy or diseased.  

- **Training Accuracy:** 83%  
- **Validation Accuracy:** 88%  

This application is designed to help farmers, agronomists, and researchers detect crop diseases at an early stage, enabling timely intervention and reducing losses.  

---

## 🚀 Features
- 🌿 Upload plant leaf images for disease detection  
- 🤖 Custom-trained CNN model for classification  
- 📊 Display of model predictions with damage( high or low)  
- 🌐 Web interface built using Django  
- 🗂 Organized dataset for model training  

---

## 🛠️ Tech Stack
- **Backend:** Django (Python)  
- **Frontend:** HTML, CSS, JavaScript (Django Templates)  
- **Deep Learning:** TensorFlow / Keras (Custom CNN)  
- **Database:** SQLite (default Django database)
- **datset:** Taken from Kaggle

---

## 📂 Project Structure
-crop_disease_app/                     # Root project folder
-│
-├── crop_disease_app/                 # Django project settings package
-│   ├── __pycache__/                  # Compiled Python cache files
-│   ├── __init__.py
-│   ├── asgi.py                       # ASGI config
-│   ├── settings.py                   # Project settings (DB, apps, static, etc.)
-│   ├── urls.py                       # Project-level URL routing
-│   ├── wsgi.py                       # WSGI config for deployment
-│
-├── crop_disease_app1/                # Main Django application
-│   ├── __pycache__/                  # Compiled cache
-│   ├── migrations/                   # Database migrations
-│   ├── templates/                    # HTML templates for frontend
-│   ├── __init__.py
-│   ├── admin.py                      # Django admin configurations
-│   ├── apps.py                       # App configuration
-│   ├── models.py                     # Django models (DB schema)
-│   ├── tests.py                      # Unit tests
-│   ├── urls.py                       # App-level URL routing
-│   ├── views.py                      # Main business logic (handles requests)
-│
-├── dataset/                          # Training dataset (should be gitignored)
-├── media/                            # Uploaded images (runtime storage)
-├── model/                            # Saved ML models (optional folder)
-├── static/                           # CSS, JS, Images (for frontend UI)
-│
-├── db.sqlite3                        # SQLite database (default Django DB)
-├── disease_info.csv                  # CSV file with disease details/info
-├── manage.py                         # Django management script
-├── modelfinal.py                     # Script for running the CNN model
-├── plantvillage-dataset.zip          # Zipped dataset (should be gitignored)
-├── train_model.py                    # Script for training CNN model
-│
-├── crop_disease_app_backup/          # Backup copy of the project (old version?)
-├── .gitignore                        # Git ignore rules
-└── crop_env/                         # Python virtual environment
