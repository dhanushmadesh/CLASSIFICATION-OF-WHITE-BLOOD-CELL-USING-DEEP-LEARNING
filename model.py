import os, numpy as np, tensorflow as tf, matplotlib.pyplot as plt, seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization, LeakyReLU)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, f1_score


TRAIN_PATH = r"C:\Users\dhanu\Desktop\BLOOD CELL CLASSIFIER\DATASET\TRAIN"
TEST_PATH  = r"C:\Users\dhanu\Desktop\BLOOD CELL CLASSIFIER\DATASET\TEST"
SAVE_PATH  = r"C:\Users\dhanu\Desktop\BLOOD CELL CLASSIFIER\wbc_classifier_model.h5"

IMG_SIZE   = (224, 224)   # resizing all input images
BATCH_SIZE = 32           # Number of images processed in one batch
EPOCHS     = 50           # Full passes over the dataset

train_datagen = ImageDataGenerator(   # normalizes + augments images to improve generalization
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    brightness_range=[0.7,1.3],
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(   # 80% of train folder
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical'
)

val_gen = train_datagen.flow_from_directory(     # 20% of train folder
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical'
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(  # only normalization
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# CLASS NAMES (required)
CLASS_NAMES = list(train_gen.class_indices.keys())
print("Classes:", CLASS_NAMES)

# CLASS WEIGHTS to handle imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))


model = Sequential([

    # Block 1 – Edges, lines, color gradients
    Conv2D(32, (3,3), kernel_regularizer=l2(1e-4), input_shape=IMG_SIZE+(3,)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Block 2 – Curves, blobs, textures
    Conv2D(64, (3,3), kernel_regularizer=l2(1e-4)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Block 3 – Nucleus shape, granules, borders
    Conv2D(128, (3,3), kernel_regularizer=l2(1e-4)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Block 4 – Full WBC morphology & class-specific patterns
    Conv2D(256, (3,3), kernel_regularizer=l2(1e-4)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Flatten – convert feature maps to 1D vector
    Flatten(),

    # Dense Layers – final decision-making
    Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(len(CLASS_NAMES), activation='softmax')  # final 4-class output
])


model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint(SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

loss, acc = model.evaluate(test_gen, verbose=1)
print(f"\n✅ Final Test Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")

Y_pred = model.predict(test_gen)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(test_gen.classes, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(test_gen.classes, y_pred, target_names=CLASS_NAMES))

print("Macro F1 Score:", f1_score(test_gen.classes, y_pred, average="macro"))
