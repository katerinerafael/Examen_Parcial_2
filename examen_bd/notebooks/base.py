# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# =========================================================
# MODELO
# =========================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================================================
# DATASET INICIAL
# =========================================================

base = [

    # -------------------------
    # ANIMALES
    # -------------------------

    "gato doméstico pequeño",
    "perro guardián grande",
    "felino salvaje africano",
    "ave tropical colorida",
    "pez marino rápido",

    # -------------------------
    # VEHÍCULOS
    # -------------------------

    "coche deportivo rápido",
    "vehículo eléctrico moderno",
    "bicicleta urbana ligera",
    "camión de carga pesado",
    "motocicleta de carreras",

    # -------------------------
    # COMIDA
    # -------------------------

    "pizza italiana tradicional",
    "sushi japonés fresco",
    "hamburguesa americana clásica",
    "ensalada saludable verde",
    "postre dulce de chocolate",

    # -------------------------
    # TECNOLOGÍA
    # -------------------------

    "computadora portátil moderna",
    "teléfono inteligente avanzado",
    "servidor empresarial seguro",
    "base de datos distribuida",
    "aplicación web interactiva",

    # -------------------------
    # AMBIGÜEDAD
    # -------------------------

    "banco financiero internacional",
    "banco de madera parque",
    "ratón inalámbrico computadora",
    "ratón pequeño gris",

    # -------------------------
    # RUIDO SEMÁNTICO
    # -------------------------

    "galaxia espiral distante",
    "planeta rocoso habitable",
    "filosofía existencial moderna"

]

# =========================================================
# EMBEDDING
# =========================================================

def embedding(texto):
    """
    Genera embedding usando SentenceTransformer
    """

    vector = model.encode(texto)

    return vector.astype("float32")

# =========================================================
# CONSTRUCCIÓN DEL ÍNDICE
# =========================================================

def construir_indice(base):
    """
    Construye índice FAISS
    desde una lista de textos
    """

    vectores = np.array([

        embedding(texto)

        for texto in base

    ])

    dimension = vectores.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(vectores)

    return index, vectores

# =========================================================
# SCORE
# =========================================================

def calcular_score(distancia):
    """
    Convierte distancia en score
    """

    return 1 / (1 + distancia)

# =========================================================
# BÚSQUEDA SEMÁNTICA
# =========================================================

def buscar(query, base, index, k=5):
    """
    Realiza búsqueda semántica
    usando FAISS
    """

    q = embedding(query).reshape(1, -1)

    distancias, indices = index.search(
        q,
        k
    )

    resultados = []

    for j, i in enumerate(indices[0]):

        distancia = float(
            distancias[0][j]
        )

        score = calcular_score(
            distancia
        )

        resultados.append({

            "texto": base[i],

            "distancia": distancia,

            "score": score,

            "indice": int(i)

        })

    return resultados

# =========================================================
# BÚSQUEDA FILTRADA
# =========================================================

def buscar_filtrado(
    query,
    base,
    index,
    umbral=0.3,
    k=5
):
    """
    Retorna resultados
    con score mínimo
    """

    resultados = buscar(
        query,
        base,
        index,
        k
    )

    filtrados = [

        r for r in resultados

        if r["score"] >= umbral

    ]

    return filtrados

# =========================================================
# CLASE PRINCIPAL
# =========================================================

class BuscadorSemantico:

    def __init__(self, base):

        self.base = base

        self.index, self.vectores = (
            construir_indice(base)
        )

    # -----------------------------------------------------

    def buscar(self, query, k=5):

        return buscar(
            query,
            self.base,
            self.index,
            k
        )

    # -----------------------------------------------------

    def buscar_filtrado(
        self,
        query,
        umbral=0.3,
        k=5
    ):

        return buscar_filtrado(
            query,
            self.base,
            self.index,
            umbral,
            k
        )

    # -----------------------------------------------------

    def agregar(self, texto):
        """
        Inserta nuevo texto
        y reconstruye índice
        """

        self.base.append(texto)

        self.index, self.vectores = (
            construir_indice(
                self.base
            )
        )

# =========================================================
# SISTEMA INICIAL
# =========================================================

sistema = BuscadorSemantico(base)

# =========================================================
# EJEMPLOS
# =========================================================

print("\n===== FELINO =====\n")

print(
    sistema.buscar("felino")
)

print("\n===== VEHÍCULO RÁPIDO =====\n")

print(
    sistema.buscar("vehículo rápido")
)

print("\n===== BANCO =====\n")

print(
    sistema.buscar("banco")
)

print("\n===== BÚSQUEDA FILTRADA =====\n")

print(
    sistema.buscar_filtrado(
        "tecnología moderna",
        umbral=0.4
    )
)