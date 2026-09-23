# Plantilla de respuestas escritas — Pregunta 2

Esto es SOLO lo que se escribe como texto/markdown en el notebook para cada punto. Sin código — el código va en `guia_codigo_y_decisiones.md`. Genérico, para rellenar con el caso que toque el día del examen.

---

## 1. Entendimiento del negocio y enfoque analítico

### 1a. Objetivo de negocio (3%)
```
[Empresa] busca ___(objetivo: predecir / clasificar / optimizar / detectar)___
a partir de ___(datos disponibles)___. El uso de aprendizaje automático se
justifica porque se cuenta con datos históricos que relacionan
___(variables de entrada)___ con ___(resultado/variable objetivo conocida)___,
permitiendo identificar patrones difíciles de capturar manualmente y
aplicarlos de forma escalable a casos nuevos.
```

### 1b. Enfoque analítico (7%)
```
- Tipo de aprendizaje: ___(supervisado/no supervisado)___, porque
  ___(existe/no existe una variable objetivo etiquetada: nómbrala)___.
- Tarea: ___(regresión/clasificación binaria/multiclase/clustering)___,
  porque la variable objetivo `___` es ___(numérica continua/categórica
  con N clases/no existe)___.
- Algoritmo(s) propuesto(s): ___(nombre)___, justificado por
  ___(criterio del caso: interpretabilidad/no linealidad/desbalance/
  tamaño de datos)___.
```
Chuleta de algoritmo según lo que pida el caso: interpretabilidad → regresión lineal/logística o árbol poco profundo · no linealidad sin requisito de interpretar → Random Forest/Gradient Boosting · clases desbalanceadas → + SMOTE/undersampling o `class_weight='balanced'` · necesita probabilidades → regresión logística · sin variable objetivo → clustering/PCA.

---

## 2. Análisis exploratorio y calidad de datos (10%)
```
Se realizó un análisis exploratorio evaluando las siguientes dimensiones de
calidad de datos:

- Completitud: la(s) columna(s) ___ presentan ___% de valores nulos.
  Corrección propuesta: ___(imputar media/mediana/moda | eliminar
  filas/columna)___, justificado por ___(% de faltantes, tipo de
  variable)___.

- Unicidad: se identificaron ___ registros duplicados / IDs repetidos
  en ___. Corrección propuesta: eliminar con drop_duplicates() /
  investigar origen.

- Exactitud: la columna ___ contiene valores físicamente/lógicamente
  imposibles, como ___(valor)___, fuera del rango esperado [___, ___].
  Corrección propuesta: ___(capear | tratar como nulo e imputar |
  transformar por unidad/escala | eliminar fila)___.

- Consistencia/Validez: la columna ___ presenta categorías equivalentes
  escritas de forma distinta (ej. ___ y ___).
  Corrección propuesta: unificar mediante mapeo antes de codificar.
```

---

## 3. Transformaciones (10%)
```
Dado que el enfoque propuesto es ___(algoritmo del punto 1b)___, se
requieren las siguientes transformaciones:

- Variables categóricas ordinales (`___`): se codifican con
  OrdinalEncoder respetando el orden ___, ya que ese orden es
  informativo para el modelo.
- Variables categóricas nominales (`___`): se codifican con
  OneHotEncoder, al no existir una relación de orden entre sus
  categorías.
- Variable binaria (`___`): se mapea a 0/1 (o con OrdinalEncoder de
  2 niveles si prefieres mantenerla en el mismo flujo que las
  ordinales).
- Variables numéricas: se escalan con StandardScaler, necesario
  porque ___(el algoritmo es sensible a la escala / usa distancias /
  regulariza coeficientes)___.
- [Si aplica] La variable objetivo ___ presenta una distribución
  sesgada, por lo que se propone ___.
```

---

## 4. Pipeline (20%)
No requiere tanto texto — el peso está en el código (ver `guia_codigo_y_decisiones.md`, sección Pipeline). Si quieres una frase de introducción antes del código:
```
A continuación se construye el pipeline de preparación de datos con base
en las decisiones justificadas en los puntos 2 y 3, separando las
estadísticas de imputación y escalado para que se aprendan únicamente
del conjunto de entrenamiento, evitando fuga de información hacia el
conjunto de prueba. Se muestra su aplicación sobre los datos de test
como ejemplo.
```

---

## Nota sobre el "Punto 5"
El "Simulacro completo" no es parte de lo que califica el enunciado — es tu propio checkpoint de práctica (correr todo con un caso ficticio distinto para comprobar que puedes aplicar la plantilla sin ayuda). No necesita su propia plantilla de respuesta; cuando lo hagamos, va a ser resolver estas mismas 4 secciones desde cero con un caso nuevo.
