import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout ,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint ,ReduceLROnPlateau
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam


img_height, img_width = 128 , 128
batch_size = 32

#augmentation
datagen = ImageDataGenerator(rescale=1./255, 
                            validation_split=0.2,
                            rotation_range=30,
                            zoom_range=0.2,
                            brightness_range=[0.8,1.2],
                            horizontal_flip=True,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.15,
                            fill_mode="nearest")
    

train = datagen.flow_from_directory(
    'dataset/plantvillage dataset/color',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val = datagen.flow_from_directory(
    'dataset/plantvillage dataset/color',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)


model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.1),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train.num_classes, activation='softmax')
])
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

os.makedirs("model", exist_ok=True)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,           # reduce learning rate by 50%
    patience=5,           # after 2 epochs of no improvement
    verbose=1,
    min_lr=1e-6           # minimum learning rate
)
checkpoint = ModelCheckpoint('model/best_model.keras', save_best_only=True)

# Training
model.fit(
    train,
    validation_data=val,
    epochs=30,
    callbacks=[early_stop, reduce_lr,checkpoint],

)

model.save("model/plant_disease_model.keras")
print("Model trained and saved.")





