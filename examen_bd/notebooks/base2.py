# =========================================================
# IMPORTS
# =========================================================

# Numpy maneja el almacenamiento y las operaciones matemáticas de matrices/vectores.
import numpy as np

# FAISS (Facebook AI Similarity Search) es la librería/motor que optimiza
# la búsqueda de similitud en espacios vectoriales de alta dimensión.
import faiss

# SentenceTransformer nos permite cargar modelos de lenguaje (LLMs) locales
# entrenados específicamente para mapear oraciones a embeddings vectoriales.
from sentence_transformers import SentenceTransformer

# =========================================================
# MODELO
# =========================================================

# Inicializa el modelo "all-MiniLM-L6-v2". 
# ¡EXAMEN!: Este modelo transforma texto en vectores con una dimensión FIJA de 384 números.
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================================================
# DATASET INICIAL (Tu base de conocimientos / Corpus de texto)
# =========================================================

base = [

    # -------------------------
    # ANIMALES
    # -------------------------
    "gato doméstico pequeño",       # Índice 0
    "perro guardián grande",        # Índice 1
    "felino salvaje africano",      # Índice 2
    "ave tropical colorida",        # Índice 3
    "pez marino rápido",            # Índice 4

    # -------------------------
    # VEHÍCULOS
    # -------------------------
    "coche deportivo rápido",       # Índice 5
    "vehículo eléctrico moderno",   # Índice 6
    "bicicleta urbana ligera",      # Índice 7
    "camión de carga pesado",       # Índice 8
    "motocicleta de carreras",      # Índice 9

    # -------------------------
    # COMIDA
    # -------------------------
    "pizza italiana tradicional",    # Índice 10
    "sushi japonés fresco",         # Índice 11
    "hamburguesa americana clásica",# Índice 12
    "ensalada saludable verde",     # Índice 13
    "postre dulce de chocolate",    # Índice 14

    # -------------------------
    # TECNOLOGÍA
    # -------------------------
    "computadora portátil moderna",  # Índice 15
    "teléfono inteligente avanzado", # Índice 16
    "servidor empresarial seguro",   # Índice 17
    "base de datos distribuida",     # Índice 18
    "aplicación web interactiva",    # Índice 19

    # -------------------------
    # AMBIGÜEDAD (Palabras iguales, significados según el contexto)
    # -------------------------
    "banco financiero internacional",# Índice 20 (Entidad de dinero)
    "banco de madera parque",        # Índice 21 (Mueble para sentarse)
    "ratón inalámbrico computadora", # Índice 22 (Hardware / Periférico)
    "ratón pequeño gris",            # Índice 23 (Animal / Roedor)

    # -------------------------
    # RUIDO SEMÁNTICO (Textos alejados de las categorías principales)
    # -------------------------
    "galaxia espiral distante",      # Índice 24
    "planeta rocoso habitable",      # Índice 25
    "filosofía existencial moderna"  # Índice 26

]

# =========================================================
# EMBEDDING (Conversión de Texto a Números)
# =========================================================

def embedding(texto):
    """
    Genera un embedding denso usando SentenceTransformer.
    Toma un string de texto y devuelve una representación numérica de su significado.
    """

    # El modelo codifica el texto. Retorna un array de floats.
    vector = model.encode(texto)

    # ¡EXAMEN!: FAISS está escrito en C++ y exige estrictamente flotantes de 32 bits.
    # .astype("float32") es OBLIGATORIO para evitar errores de tipo de dato.
    return vector.astype("float32")

# =========================================================
# CONSTRUCCIÓN DEL ÍNDICE (Creación de la Estructura de Datos de Búsqueda)
# =========================================================

def construir_indice(base):
    """
    Construye un índice FAISS en memoria desde una lista de textos.
    """

    # Comprensión de lista: Convierte CADA texto del dataset en su respectivo vector.
    # np.array() agrupa todos los vectores individuales en una matriz bidimensional (Matriz de Embeddings).
    vectores = np.array([
        embedding(texto)
        for texto in base
    ])

    # .shape[1] extrae el número de columnas de la matriz, es decir, la dimensión (384).
    dimension = vectores.shape[1]

    # ¡EXAMEN!: Inicializa un índice tipo 'Flat' usando la métrica 'L2' (Distancia Euclidiana).
    # 'Flat' significa indexación por fuerza bruta: almacena los vectores completos sin comprimir.
    index = faiss.IndexFlatL2(
        dimension
    )

    # Añade la matriz de vectores al índice FAISS para habilitar las búsquedas sobre ellos.
    index.add(vectores)

    # Devuelve el índice listo para buscar y la matriz de vectores generada.
    return index, vectores

# =========================================================
# SCORE (Métrica de confianza humana)
# =========================================================

def calcular_score(distancia):
    """
    Convierte la distancia matemática L2 en un score de similitud entendible (0 a 1).
    - Distancia = 0 (Vectores idénticos) -> Score = 1 / (1 + 0) = 1.0 (Máxima similitud)
    - Distancia Alta (Muy diferentes) -> Score tiende a 0.
    """
    return 1 / (1 + distancia)

# =========================================================
# BÚSQUEDA SEMÁNTICA (El Motor de Inferencia)
# =========================================================

def buscar(query, base, index, k=5):
    """
    Realiza la búsqueda semántica de los 'k' elementos más cercanos en el índice FAISS.
    """

    # Genera el embedding de la consulta y usa .reshape(1, -1) para transformarlo en una 
    # matriz de 1 fila y N columnas, ya que FAISS espera un lote (batch) de consultas.
    q = embedding(query).reshape(1, -1)

    # El método index.search realiza el cálculo geométrico y retorna dos matrices numpy:
    # 1. distancias: El valor de distancia Euclidiana de los k mejores resultados.
    # 2. indices: Las posiciones (IDs de fila) de esos k resultados en nuestro dataset original.
    distancias, indices = index.search(
        q,
        k
    )

    resultados = []

    # Iteramos sobre la lista de índices encontrados (accedemos a indices[0] porque pasamos una sola query)
    for j, i in enumerate(indices[0]):

        # Extrae el valor numérico de la distancia matemática para ese elemento.
        distancia = float(
            distancias[0][j]
        )

        # Mapea la distancia a un score porcentual/humano.
        score = calcular_score(
            distancia
        )

        # Construye un diccionario estructurado con toda la metadata del resultado.
        resultados.append({
            "texto": base[i],       # El texto original asociado al vector recuperado.
            "distancia": distancia, # Qué tan lejos está geométricamente.
            "score": score,         # Similitud normalizada (0 a 1).
            "indice": int(i)        # ID de posición original en la lista 'base'.
        })

    return resultados

# =========================================================
# BÚSQUEDA FILTRADA (Post-procesamiento / Umbral de Calidad)
# =========================================================

def buscar_filtrado(
    query,
    base,
    index,
    umbral=0.3,
    k=5
):
    """
    Primero busca los k elementos más cercanos y luego aplica un filtro
    para descartar resultados de mala calidad (menores al umbral).
    """

    # Ejecuta la búsqueda estándar (esto siempre devuelve hasta k elementos).
    resultados = buscar(
        query,
        base,
        index,
        k
    )

    # Filtrado por list comprehension: Conserva sólo los diccionarios donde score >= umbral.
    # ¡EXAMEN!: Si k=5 pero sólo 2 elementos superan el umbral, esta función devolverá sólo 2 elementos.
    filtrados = [
        r for r in resultados
        if r["score"] >= umbral
    ]

    return filtrados

# =========================================================
# CLASE PRINCIPAL (Abstracción e Interfaz del Sistema)
# =========================================================

class BuscadorSemantico:

    def __init__(self, base):
        """
        Constructor: Guarda el corpus y genera el índice FAISS automáticamente al instanciar.
        """
        self.base = base

        # Llama a la función global para inicializar y poblar el índice con los textos de 'base'.
        self.index, self.vectores = (
            construir_indice(base)
        )

    # -----------------------------------------------------

    def buscar(self, query, k=5):
        """
        Encapsula la función global de búsqueda utilizando las variables de la instancia.
        """
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
        """
        Encapsula la función global de búsqueda filtrada por umbral.
        """
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
        Inserta un nuevo elemento en la base de datos de manera dinámica.
        ¡EXAMEN / CRÍTICA!: Este diseño añade el texto al array 'base' y RECONSTRUYE
        el índice FAISS desde cero. Esto es altamente ineficiente para bases de datos reales en producción, 
        pero funcional para este ejercicio didáctico.
        """
        # Agrega físicamente el string a la lista de textos.
        self.base.append(texto)

        # Borra el índice anterior y calcula todos los embeddings de nuevo.
        self.index, self.vectores = (
            construir_indice(
                self.base
            )
        )

# =========================================================
# SISTEMA INICIAL (Instanciación del motor en memoria)
# =========================================================

# Creamos el objeto 'sistema' pasándole la lista 'base' definida al inicio.
# En este momento exacto se calculan los 27 embeddings iniciales de 384 dimensiones cada uno.
sistema = BuscadorSemantico(base)

# =========================================================
# EJEMPLOS DE PRUEBA (Análisis de ejecución)
# =========================================================

print("\n===== FELINO =====\n")
# Buscará "felino". El Top 1 debería ser "felino salvaje africano" e incluirá "gato doméstico pequeño"
# debido a la cercanía semántica conceptual de la familia de animales (felinos).
print(
    sistema.buscar("felino")
)

print("\n===== VEHÍCULO RÁPIDO =====\n")
# Evaluará la combinación de dos conceptos: "vehículo" y "rápido". 
# Debería rankear en los primeros puestos "coche deportivo rápido" y "motocicleta de carreras".
print(
    sistema.buscar("vehículo rápido")
)

print("\n===== BANCO =====\n")
# ¡EXAMEN - DESAFÍO DE AMBIGÜEDAD!: La palabra "banco" sola no tiene contexto. 
# El modelo traerá tanto "banco financiero internacional" como "banco de madera parque". 
# Observa las distancias y scores, suelen estar muy parejos porque el modelo no sabe a qué te refieres.
print(
    sistema.buscar("banco")
)

print("\n===== BÚSQUEDA FILTRADA =====\n")
# Solicita buscar "tecnología moderna" limitando el resultado estrictamente a los que tengan
# un score mayor o igual a 0.4. Los textos sobre comida, animales o astronomía serán descartados.
print(
    sistema.buscar_filtrado(
        "tecnología moderna",
        umbral=0.4
    )
)
