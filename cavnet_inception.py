import os
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Parameters
batch_size = 20
image_size = (224, 224)
epochs = 20  # Increased number of epochs
learning_rate = 0.00001  # Smaller learning rate

# Paths
train_dir = r'C:\Users\Greg\Dog\train'
validation_dir = r'C:\Users\Greg\Dog\validation'
test_dir = r'C:\Users\Greg\Dog\test'

# Data generators with enhanced augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Compute class weights to handle class imbalance
classes = list(train_generator.class_indices.keys())
class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weights_dict = dict(enumerate(class_weights))

# Load InceptionV3 model pre-trained on ImageNet, without the top layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom top layers with dropout
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Combine base model and custom layers into a new model
model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze the last few layers of the base InceptionV3 model for fine-tuning
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Compile the model with a smaller learning rate
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    class_weight=class_weights_dict,
    callbacks=[early_stopping]
)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")

# Save the model
model.save(r'C:\Users\Greg\Dog\DogNet_InceptionV3.h5')

# Data generator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of image data for the test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=20,
    class_mode='binary',
    shuffle=False
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Predict on the test set
test_predictions = model.predict(test_generator)
predicted_classes = (test_predictions > 0.5).astype(int).flatten()

# Map the predicted classes back to the class names
index_to_label = {v: k for k, v in test_generator.class_indices.items()}
predicted_labels = [index_to_label[i] for i in predicted_classes]

# Print the results
for filename, predicted_label in zip(test_generator.filenames, predicted_labels):
    print(f"{filename}: {predicted_label}")
