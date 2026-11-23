from datetime import timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from llm import VARIANTES, generar_recomendacion, llm_cargado, precargar_llm
from rl_feedback import CLASES, actualizar, mejor_accion

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR.parent / "notebooks" / "mejor_modelo.keras").resolve()

IMG_SIZE = (180, 180)
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

CLASS_NAMES = CLASES

CLASS_NAMES_ES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "Sin tumor detectable",
    "pituitary": "Tumor hipofisario",
}


app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES
# El navegador guarda el CSS y no lo revalida en cada carga.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = timedelta(days=30)


def asset_version(nombre):
    """Fecha de modificación para invalidar la caché cuando cambia el archivo."""
    return int((BASE_DIR / "static" / nombre).stat().st_mtime)


app.jinja_env.globals["asset_version"] = asset_version


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


cnn_model = None

def load_models() -> None:
    global cnn_model

    if cnn_model is not None:
        return

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Verifica que exista Proyecto/notebooks/mejor_modelo.keras."
        )

    print(f"[CNN] Cargando modelo desde {MODEL_PATH} ...", flush=True)
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    print("[CNN] Modelo listo.", flush=True)


def preprocess_image(contents: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("El archivo no es una imagen válida.") from error

    image = image.resize(IMG_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


@app.route("/")
def home():
    precargar_llm()
    return render_template("index.html")


@app.route("/health")
def health_check():
    return jsonify(
        {
            "status": "ok",
            "cnn_loaded": str(cnn_model is not None),
            "llm_loaded": str(llm_cargado()),
            "message": "API de clasificación de tumores cerebrales con LLM activa",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Clasifica la imagen y usa la predicción como INPUT del LLM."""
    file = request.files.get("image")

    if file is None or not file.filename:
        return (
            render_template("index.html", error="Debes seleccionar una imagen."),
            400,
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        return (
            render_template(
                "index.html",
                error="El archivo enviado no es una imagen válida.",
            ),
            400,
        )

    if cnn_model is None:
        return (
            render_template(
                "index.html",
                error="El modelo CNN no está cargado. Intenta de nuevo.",
            ),
            503,
        )

    try:
        image = preprocess_image(file.read())
    except ValueError as error:
        return render_template("index.html", error=str(error)), 400

    predictions = cnn_model.predict(image, verbose=0)
    probabilities = predictions[0]

    print("\nProbabilidades:")
    for clase, probabilidad in zip(CLASS_NAMES, probabilities):
        print(f"{clase}: {probabilidad * 100:.2f}%")

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    variant_idx = mejor_accion(predicted_class)

    # LLM: la predicción entra como INPUT y genera el texto de recomendación.
    recommendation = generar_recomendacion(predicted_class, confidence, variant_idx)

    if not recommendation:
        return (
            render_template(
                "index.html",
                error=(
                    "El modelo de lenguaje no pudo generar la recomendación. "
                    "Intenta de nuevo en unos momentos."
                ),
                prediction=predicted_class,
                prediction_es=CLASS_NAMES_ES.get(predicted_class, predicted_class),
                confidence=confidence,
            ),
            503,
        )

    return render_template(
        "index.html",
        prediction=predicted_class,
        prediction_es=CLASS_NAMES_ES.get(predicted_class, predicted_class),
        confidence=confidence,
        recommendation=recommendation,
        variant=VARIANTES[variant_idx]["nombre"],
        variant_idx=variant_idx,
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    """Calificación del usuario (1-5 estrellas) que entrena la Q-table."""
    datos = request.get_json(silent=True) or {}

    clase = datos.get("clase")
    variante = datos.get("variante")
    estrellas = datos.get("estrellas")

    if clase not in CLASS_NAMES or variante is None or estrellas is None:
        return jsonify({"ok": False, "error": "Datos incompletos"}), 400

    try:
        variante = int(variante)
        estrellas = int(estrellas)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Valores no numéricos"}), 400

    if not 1 <= estrellas <= 5 or not 0 <= variante < len(VARIANTES):
        return jsonify({"ok": False, "error": "Rango inválido"}), 400

    q_actualizada = actualizar(clase, variante, estrellas)

    return jsonify(
        {
            "ok": True,
            "mensaje": "¡Gracias por tu calificación! La recomendación se "
            "ajustará con tus próximos intentos.",
            "q_valor": round(q_actualizada, 4),
        }
    )

load_models()

if __name__ == "__main__":
    app.run(debug=True)
