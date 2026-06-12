# train.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==================================
# DATA AUGMENTATION
# ==================================

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_gen = ImageDataGenerator(
    rescale=1./255
)

test_gen = ImageDataGenerator(
    rescale=1./255
)

# ==================================
# LOAD DATASET
# ==================================

train_data = train_gen.flow_from_directory(
    "dataset_split/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    "dataset_split/validation",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    "dataset_split/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\n========== CLASS MAPPING ==========")
print(train_data.class_indices)
print("Jumlah kelas:", train_data.num_classes)

# ==================================
# MOBILENETV2
# ==================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Fine Tuning
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# ==================================
# CLASSIFICATION HEAD
# ==================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(0.4)(x)

predictions = Dense(
    train_data.num_classes,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# ==================================
# COMPILE
# ==================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==================================
# CALLBACKS
# ==================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "model/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ==================================
# TRAINING
# ==================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ]
)

# ==================================
# EVALUASI TEST
# ==================================

print("\n========== TESTING ==========")

loss, acc = model.evaluate(
    test_data,
    verbose=1
)

print(
    f"\nAkurasi Test : {acc*100:.2f}%"
)

print(
    f"Loss Test : {loss:.4f}"
)

# ==================================
# SAVE MODEL
# ==================================

model.save(
    "model/rice_disease_model.keras"
)

print("\n========== TRAINING SELESAI ==========")
print("Model tersimpan di:")
print("model/rice_disease_model.keras")
print("Model terbaik:")
print("model/best_model.keras")