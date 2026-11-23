import json
import os
import random

CLASES = ["glioma", "meningioma", "notumor", "pituitary"]
N_ACCIONES = 2

ALPHA = 0.1      # tasa de aprendizaje
GAMMA = 0.9      # factor de descuento
EPSILON = 0.2    # probabilidad de explorar otra variante

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RUTA_Q = os.path.join(DATA_DIR, "q_table.json")

_q_table = None


def _nueva_q_table():
    return {clase: [0.0] * N_ACCIONES for clase in CLASES}


def _cargar():
    """Carga la Q-table desde disco (la crea si no existe)."""
    global _q_table

    if _q_table is not None:
        return _q_table

    _q_table = _nueva_q_table()

    if os.path.exists(RUTA_Q):
        try:
            with open(RUTA_Q, "r", encoding="utf-8") as archivo:
                datos = json.load(archivo)
            for clase in CLASES:
                valores = datos.get("q_table", {}).get(clase)
                if isinstance(valores, list) and len(valores) == N_ACCIONES:
                    _q_table[clase] = [float(v) for v in valores]
        except Exception as exc:
            print(f"[RL] No se pudo leer la Q-table: {exc}")

    return _q_table


def _guardar():
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RUTA_Q, "w", encoding="utf-8") as archivo:
        json.dump(
            {"q_table": _q_table, "alpha": ALPHA, "gamma": GAMMA},
            archivo,
            ensure_ascii=False,
            indent=2,
        )


def mejor_accion(clase):
    q_table = _cargar()
    if clase not in q_table:
        clase = CLASES[0]

    if random.random() < EPSILON:
        return random.randrange(N_ACCIONES)

    valores = q_table[clase]
    maximo = max(valores)
    candidatos = [i for i, v in enumerate(valores) if v == maximo]
    return random.choice(candidatos)


def recompensa(estrellas):
    estrellas = max(1, min(int(estrellas), 5))
    return (estrellas - 1) / 4.0


def actualizar(clase, accion, estrellas):
    """Actualiza Q[s][a] con la regla de Q-learning y guarda """
    q_table = _cargar()
    if clase not in q_table:
        clase = CLASES[0]

    accion = max(0, min(int(accion), N_ACCIONES - 1))
    r = recompensa(estrellas)

    q_anterior = q_table[clase][accion]
    q_siguiente = max(q_table[clase])         
    q_table[clase][accion] = q_anterior + ALPHA * (r + GAMMA * q_siguiente - q_anterior)

    _guardar()

    print(
        f"[RL] {clase} | acción {accion} | estrellas {estrellas} "
        f"| recompensa {r:.2f} | Q: {q_anterior:.3f} -> {q_table[clase][accion]:.3f}",
        flush=True,
    )
    return q_table[clase][accion]
