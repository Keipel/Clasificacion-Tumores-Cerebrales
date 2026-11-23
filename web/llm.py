import re
import threading

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MIN_CHARS = 200
MAX_NEW_TOKENS = 450

VARIANTES = [
    {
        "nombre": "Descripción",
        "enfoque": (
            "Describe de forma declarativa qué es la clase detectada, sus "
            "características generales y qué significa haberla obtenido en una "
            "resonancia. Después incluye recomendaciones para manejarla. No hagas "
            "preguntas ni ofrezcas ayuda adicional."
        ),
    },
    {
        "nombre": "Primeros pasos",
        "enfoque": (
            "Explica de forma declarativa qué conviene hacer para manejar la "
            "situación y refuerza con recomendaciones prácticas: conservar la "
            "imagen, acudir a un profesional, llevar los estudios previos y anotar "
            "la información relevante. No hagas preguntas ni pidas respuestas al "
            "usuario."
        ),
    },
]

DESCRIPCIONES = {
    "glioma": "glioma (tumor cerebral originado en células de la glía)",
    "meningioma": "meningioma (tumor originado en las meninges, suele ser benigno)",
    "notumor": "ausencia de tumor detectable en la imagen",
    "pituitary": "tumor hipofisario (adenoma de la hipófisis)",
}

_SISTEMA = (
    "Eres un asistente de divulgación médica que colabora en un proyecto académico "
    "de inteligencia artificial. Recibes la predicción de un modelo que clasifica "
    "imágenes de resonancia magnética cerebral y debes redactar una recomendación "
    "clara, informativa y responsable en español. Reglas obligatorias: responde "
    "únicamente con el texto de la recomendación, sin saludos, encabezados, listas "
    "ni despedidas; extiéndete siempre más de 200 caracteres; usa lenguaje "
    "divulgativo para personas sin formación médica; no formules preguntas, no uses "
    "signos de interrogación y no ofrezcas ayuda adicional ni pidas al usuario que "
    "responda; escribe en modo declarativo, explicando qué es la clase detectada y, "
    "justo después, cómo manejarla; incluye recomendaciones concretas para manejar la "
    "situación, como acudir a un especialista, llevar los estudios e informes previos, "
    "anotar los síntomas y su evolución, no automedicarse y acudir a urgencias si "
    "aparecen síntomas graves; si la clase es 'notumor', aclara que no se detectó "
    "tumor y da solo indicaciones generales de prevención y cuidado, sin decir que la "
    "condición se debe a un tumor; nunca afirmes que exista "
    "un diagnóstico ni que la " 
    "predicción sea un resultado clínico: aclara que es una orientación experimental "
    "y que la interpretación corresponde a un profesional de la salud. No uses la "
    "palabra 'confirmado' ni afirmes que la persona padece la enfermedad; usa "
    "fórmulas como 'la predicción sugiere' o 'el modelo clasificó'. No inventes "
    "estadísticas, pronósticos, probabilidades de cura ni tratamientos; no recetes "
    "medicamentos, dosis, cirugías ni tratamientos específicos y no estimes "
    "probabilidades ni afirmes que la clase sea cancerosa o maligna. Limítate a "
    "explicar qué significa la predicción, cómo manejar la situación y la "
    "importancia del control médico profesional. Tampoco inventes detalles visuales "
    "de la imagen ni afirmes haberla visto: solo conoces la clase predicha y su "
    "confianza. No escribas meta-instrucciones ni títulos; evita frases como "
    "'Aquí hay información', 'Estamos evaluando' o 'Técnica de redacción'. Habla "
    "directamente a la persona que consulta, en segunda persona, pero sin hacerle "
    "preguntas."
)

_pipe = None
_lock = threading.Lock()
_arrancado = False
_intentos = 0
_error = None


def _obtener_pipeline():
    global _pipe, _intentos, _error

    with _lock:
        if _pipe is not None:
            return _pipe
        if _error is not None and _intentos >= 3:
            return None

        _intentos += 1
        try:
            from transformers import pipeline

            print(f"[LLM] Cargando modelo {MODEL_ID} ...", flush=True)
            _pipe = pipeline("text-generation", model=MODEL_ID)
            print("[LLM] Modelo listo.", flush=True)
        except Exception as exc:
            _error = exc
            print(f"[LLM] No se pudo cargar el modelo: {exc}", flush=True)
            return None

        return _pipe


def precargar_llm():
    global _arrancado

    with _lock:
        if _arrancado:
            return
        _arrancado = True

    threading.Thread(target=_obtener_pipeline, daemon=True).start()


def llm_cargado() -> bool:
    return _pipe is not None


def _mensajes(clase, confianza, variante_idx):
    descripcion = DESCRIPCIONES.get(clase, clase)
    aviso = (
        "REGLA PARA 'notumor': no se detectó tumor. No afirmes ni insinúes que la "
        "condición se deba a un tumor ni que pueda surgir por uno; no inventes "
        "hallazgos (tejido, lesiones, enfermedades); describe el resultado como "
        "ausencia de tumor y da solo indicaciones generales de prevención y cuidado.\n"
        if clase == "notumor" else ""
    )
    usuario = (
        f"Predicción del modelo de IA: {clase} — {descripcion} "
        f"(confianza {confianza:.1f}%).\n"
        f"Enfoque de la recomendación: {VARIANTES[variante_idx]['enfoque']}\n\n"
        "Usa la descripción dada de la clase y no traduzcas literalmente el "
        "término en inglés.\n"
        "Estructura: primero indica qué es la clase detectada y qué significa; "
        "inmediatamente después presenta las recomendaciones para manejarla; y "
        "cierra aclarando que el resultado es experimental y debe interpretarlo un "
        "profesional. Escribe en párrafos continuos, sin listas ni numeración.\n"
        f"{aviso}"
        "Redacta ahora la recomendación sin hacer preguntas ni ofrecer ayuda."
    )
    return [
        {"role": "system", "content": _SISTEMA},
        {"role": "user", "content": usuario},
    ]


def _limpiar_texto(texto: str) -> str:
    if not texto:
        return ""

    texto = re.sub(r"\s+", " ", texto).strip()
    # Quita numeración, viñetas y negritas para dejar párrafos continuos.
    texto = re.sub(r"^\s*(?:\d+[.)]|[-•*]+)\s+", "", texto)
    texto = re.sub(r"(?<=[.:])\s+(?:\d+[.)]|[-•*]+)\s+", " ", texto)
    texto = texto.replace("**", "")
    oraciones = re.split(r"(?<=[.!?])\s+", texto)

    limpias = []
    vistas = set()
    for oracion in oraciones:
        oracion = oracion.strip()
        if not oracion:
            continue
        # Descarta cualquier oración interrogativa.
        if "?" in oracion or "¿" in oracion:
            continue
        # Descarta oraciones con caracteres extraños o de otros alfabetos.
        if re.search(r"[^\x00-\x7FáéíóúüñÁÉÍÓÚÜÑ¿¡.,;:()\-%'\" ]", oracion):
            continue
        # Evita que el modelo repita la misma oración varias veces.
        clave = re.sub(r"\s+", " ", oracion).lower()
        if clave in vistas:
            continue
        vistas.add(clave)
        limpias.append(oracion)

    texto = " ".join(limpias).strip()
    texto = re.sub(r"\bpor favor,?\s*", "", texto, flags=re.IGNORECASE).strip()
    if texto:
        texto = texto[0].upper() + texto[1:]
    return texto


def _extraer_texto(salida):
    """Normaliza la salida del pipeline (lista de mensajes o texto plano)."""
    gen = salida[0]["generated_text"]

    if isinstance(gen, list):  # pipeline con plantilla de chat
        ultimo = gen[-1]
        return ultimo.get("content", "") if isinstance(ultimo, dict) else str(ultimo)
    if isinstance(gen, dict):
        return gen.get("content", "")
    return str(gen).strip()


def _generar(mensajes, max_new_tokens):
    pipe = _obtener_pipeline()
    if pipe is None:
        return None

    kwargs = dict(
        max_new_tokens=max_new_tokens,
        # Evita que el modelo se corte antes de llegar a los 200 caracteres
        min_new_tokens=160,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    try:
        salida = pipe(mensajes, **kwargs)
    except Exception:
        # Plantilla de chat no soportada: prompt en texto plano
        plano = mensajes[0]["content"] + "\n\n" + mensajes[1]["content"]
        try:
            salida = pipe(plano, return_full_text=False, **kwargs)
        except Exception as exc:
            print(f"[LLM] Error al generar: {exc}", flush=True)
            return None

    return _extraer_texto(salida)


def generar_recomendacion(clase, confianza, variante_idx):
    variante_idx = max(0, min(int(variante_idx), len(VARIANTES) - 1))
    mensajes = _mensajes(clase, confianza, variante_idx)

    motivo = "sin respuesta del modelo"
    for max_tokens in (MAX_NEW_TOKENS, MAX_NEW_TOKENS + 250):
        generado = _generar(mensajes, max_tokens)
        if not generado:
            motivo = "sin respuesta del modelo"
            continue

        if "?" in generado or "¿" in generado:
            print("[LLM] Se eliminaron preguntas del texto.", flush=True)

        texto = _limpiar_texto(generado)
        if clase == "notumor" and re.search(
            r"(?:debido|asociad[oa]|causad[oa]|provocad[oa])\s+(?:a|con|por)\s+"
            r"(?:la\s+)?(?:presencia\s+de\s+)?(?:un\s+)?tumor",
            texto,
            re.IGNORECASE,
        ):
            motivo = "el texto atribuye la condición a un tumor"
            continue

        if len(texto) > MIN_CHARS:
            print(f"[LLM] Recomendación generada ({len(texto)} caracteres).", flush=True)
            return texto

        motivo = f"texto insuficiente tras la limpieza ({len(texto)} caracteres)"

    print(f"[LLM] No se pudo generar la recomendación: {motivo}.", flush=True)
    return None
