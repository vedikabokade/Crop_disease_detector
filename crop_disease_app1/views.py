from django.shortcuts import render

# Create your views here.
import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# model = load_model(os.path.join(settings.BASE_DIR,"model","plant_disease_model.h5"))

# Lazy-load model
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model(os.path.join(settings.BASE_DIR, "model", "plant_disease_model.h5"))
    return _model

# disease_df = pd.read_csv(os.path.join(settings.BASE_DIR, 'disease_info.csv'))

# class_names = sorted(disease_df['label'].unique())
disease_df = None
class_names = None
def get_disease_data():
    global disease_df, class_names
    if disease_df is None:
        csv_path = os.path.join(settings.BASE_DIR, 'disease_info.csv')
        disease_df = pd.read_csv(csv_path)
        class_names = sorted(disease_df['label'].unique())
    return disease_df, class_names




def get_disease_info(label):
    row = disease_df[disease_df['label'] == label]
    if not row.empty:
        row_data = row.iloc[0]
        return {
            'crop': row_data['crop'],
            'disease': row_data['disease'],
            'description': row_data['description'],
            'symptoms': row_data['symptoms'],
            'cause': row_data['cause'],
            'solution': row_data['solution'],
            'prevention': row_data['prevention'],
            'danger_level': row_data['danger_level'],
        }
    return {
        'crop': 'Unknown',
        'disease': 'Unknown',
        'description': 'No info available.',
        'symptoms': 'N/A',
        'cause': 'N/A',
        'solution': 'Consult an expert.',
        'prevention': 'N/A',
        'danger_level': 'Unknown'
    }

def index(request):
    return render(request, 'index.html')


def predict(request):
    if request.method == 'POST' and 'image' in request.FILES:
        img = request.FILES['image']

        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        img_path = os.path.join(upload_dir, img.name)
        with open(img_path, 'wb+') as f:
            for chunk in img.chunks():
                f.write(chunk)

        img_obj = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img_obj) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model = get_model()
        disease_df, class_names = get_disease_data()

        prediction = model.predict(img_array)
        predicted_label = class_names[np.argmax(prediction)]
        disease_details = get_disease_info(predicted_label, disease_df)

        return render(request, 'index.html', {
            'image_path': settings.MEDIA_URL +'uploads/' + img.name,
            **disease_details
        })
    return render(request, 'index.html')

