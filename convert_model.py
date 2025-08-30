import tensorflow as tf

# Load your existing Keras model
model = tf.keras.models.load_model("model/plant_disease_model.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("model/plant_disease_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion successful: plant_disease_model.tflite created")
