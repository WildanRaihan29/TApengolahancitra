import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model hasil training terbaru
model = load_model("model/best_model.keras")

classes = [
    "Blast",
    "Blight",
    "Brown Spot",
    "Hispa",
    "Normal",
    "Tungro"
]

CONFIDENCE_THRESHOLD = 85

def predict_disease(path):

    # Load gambar
    img = image.load_img(
        path,
        target_size=(224, 224)
    )

    img = image.img_to_array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # Prediksi
    predictions = model.predict(
        img,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    confidence = float(
        predictions[predicted_index] * 100
    )

    result = classes[predicted_index]

    print("\n===== HASIL PREDIKSI =====")

    for i, class_name in enumerate(classes):
        print(
            f"{class_name:<12}: "
            f"{predictions[i] * 100:.2f}%"
        )

    print(
        f"\nPrediksi : {result}"
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )

    # Jika confidence terlalu rendah
    if confidence < CONFIDENCE_THRESHOLD:
        return (
            "Gambar Tidak Dikenali",
            confidence
        )

    return (
        result,
        confidence
    )