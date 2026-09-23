# Guía de código y decisiones — Parcial práctico 1

Esto es SOLO el "cómo": checklist, tabla de decisión, y patrones de código. Lo que se ESCRIBE como respuesta va en `plantilla_respuestas_escritas.md`. Genérico y reusable para cualquier caso nuevo.

---

## Checklist (para ir tachando durante el examen)

### General, al inicio
- [ ] `df.shape`
- [ ] `df.info()` — ¿dtypes correctos?
- [ ] `df.duplicated().sum()` — ¿filas 100% duplicadas?
- [ ] `df['id'].duplicated().sum()` — ¿ID repetido? (si da más que el número de filas 100% duplicadas → hay conflictos de ID que investigar, no solo duplicados exactos)
- [ ] `df.isnull().sum()` — ¿qué columnas tienen nulos y en qué %?

### Por columna NUMÉRICA
- [ ] min/max de `describe()` → ¿físicamente posible?
- [ ] ¿valor aislado o hay muchos con un patrón? (percentiles, múltiplos)
- [ ] % de nulos
- [ ] histograma → ¿forma razonable o barra aislada?

### Por columna CATEGÓRICA
- [ ] `.unique()`/`.value_counts()` → ¿número de categorías esperado?
- [ ] ¿misma categoría escrita distinto? (mayúsc/minúsc, abreviaturas)
- [ ] ¿categoría con muy pocas observaciones (typo)?
- [ ] % de nulos
- [ ] si es ordinal, ¿está claro el orden para codificarla después?

### Relación con el target
- [ ] ¿el target tiene nulos? (si sí → eliminar esas filas, no imputar)
- [ ] clasificación → `value_counts()` del target (¿balanceado?)
- [ ] regresión → `describe()` del target (¿rango razonable?) + correlación de Pearson con numéricas
- [ ] scatter (numéricas) / boxplot (categóricas) vs target

### Antes de pasar a transformaciones
- [ ] ¿ya corregiste todo lo que es error (no estadística)?
- [ ] ¿dejaste solo la imputación estadística para el pipeline?
- [ ] ¿volviste a correr `isnull().sum()`/`duplicated().sum()` para confirmar?

---

## Guía de decisión por dimensión — con código pegado a cada situación

### Completitud (valores faltantes)

Detectar:
```python
df.isnull().sum()
(df.isnull().sum() / df.shape[0] * 100).round(2).sort_values(ascending=False)
```

| Situación | Qué hacer | Código |
|---|---|---|
| Nulos <5-10%, numérica | Imputar mediana — **dentro del pipeline**, no aquí manual | Ya está en `numeric_transformer` → `SimpleImputer(strategy='median')` |
| Nulos <5-10%, categórica | Imputar moda — dentro del pipeline | `SimpleImputer(strategy='most_frequent')` |
| Nulos >40-50% en una columna | Evaluar eliminar la columna completa | `df = df.drop(columns=['col'])` |
| Nulos en el **target** | Eliminar esas filas (no se puede entrenar sin etiqueta) | `df = df.dropna(subset=['target'])` |
| Nulos concentrados en una categoría específica | Imputar por grupo en vez de con la mediana global | `df['num'] = df.groupby('cat')['num'].transform(lambda x: x.fillna(x.median()))` |
| Falta un ID | Generar uno nuevo que no choque con los existentes | `nuevo_id = df['id'].max()+1; df.loc[df['id'].isna(), 'id'] = nuevo_id` (ver nota abajo si son varios nulos) |

Nota: si necesitas asignar un ID nuevo a **varias** filas nulas a la vez (no solo una), cada una necesita un valor distinto:
```python
n_nulos = df['id'].isna().sum()
nuevos_ids = range(int(df['id'].max())+1, int(df['id'].max())+1+n_nulos)
df.loc[df['id'].isna(), 'id'] = list(nuevos_ids)
```

### Unicidad (duplicados)

Detectar:
```python
df.duplicated().sum()                     # filas 100% idénticas
df['id'].duplicated().sum()               # IDs repetidos (puede incluir conflictos, no solo exactos)
```

| Situación | Qué hacer | Código |
|---|---|---|
| Fila 100% idéntica repetida | Eliminar la copia | `df = df.drop_duplicates(subset=[...], keep='first')` |
| Mismo ID, datos distintos (conflicto real) | Investigar antes de decidir | Ver bloque de investigación abajo |

Investigar un conflicto de ID (mismo ID, filas con datos distintos):
```python
repetidos = df[df['id'].duplicated(keep=False)]
for id_, grupo in repetidos.groupby('id'):
    print(id_, 'versiones distintas:', grupo.drop(columns='id').drop_duplicates().shape[0])
    # 1 versión = duplicado exacto normal; >1 = conflicto real, hay que decidir
```

Si es un conflicto real sin explicación (no se resuelve con una corrección de otro tipo, como pasó con Edad/meses), tres salidas posibles, a elegir con criterio y justificando:
```python
# Opción A: quedarte con la fila más completa (menos nulos)
conflicto = df[df['id'] == id_problema]
fila_mas_completa = conflicto.loc[conflicto.notna().sum(axis=1).idxmax()]

# Opción B: eliminar ambas versiones (si el conflicto es irresoluble)
df = df[df['id'] != id_problema]

# Opción C: son personas/registros distintos, solo el ID chocó por error — asignar uno nuevo a una de las dos
nuevo_id = df['id'].max() + 1
df.loc[conflicto.index[1], 'id'] = nuevo_id
```

### Validez / Consistencia (categóricas y tipos)

Detectar:
```python
df['col'].unique()
df['col'].value_counts(dropna=False)
```

| Situación | Qué hacer | Código |
|---|---|---|
| Misma categoría escrita distinto (ej. "M" y "Masculino") | Unificar con un mapeo | `df['col'] = df['col'].replace({'M': 'Masculino', 'F': 'Femenino'})` |
| Numérica cargada como texto | Forzar conversión, lo inválido se vuelve NaN | `df['col'] = pd.to_numeric(df['col'], errors='coerce')` |
| Categórica con muy pocas observaciones (posible typo) | Revisar caso por caso si es válida o un error | inspeccionar con `df[df['col'] == 'valor_raro']` antes de decidir |

### Exactitud (valores imposibles/fuera de rango)

Detectar:
```python
df['col'].describe()   # revisar min/max contra lo físicamente posible
```

| Situación | Cómo diferenciarla | Qué hacer | Código |
|---|---|---|---|
| Valor aislado, sin patrón (outlier real, posiblemente legítimo) | Pocos casos, sin relación entre sí | Evaluar con IQR; dejar si es plausible, o capear si distorsiona mucho el modelo | `Q1, Q3 = df['col'].quantile([.25,.75]); IQR = Q3-Q1` luego filtrar con esos límites |
| Muchos valores fuera de rango con un patrón (ej. múltiplos de algo) | Revisar percentiles — salto brusco, muchos casos iguales | Sospechar error sistemático de unidad — corregir con una operación, no borrar | `df.loc[condición, 'col'] = df.loc[condición, 'col'] / factor` |
| Valor claramente inválido, sin causa recuperable (ej. negativo donde no puede serlo) | Pocos casos, sin fórmula que los explique | Tratar como inválido, no inventar un valor | `df.loc[condición, 'col'] = np.nan` (se imputa después en el pipeline) |

**Regla de oro outliers:** estadísticamente atípico ≠ error. Imposible → corregir/invalidar. Solo inusual pero posible → dejarlo y documentarlo (puede ser un caso real, ej. un atleta excepcional).

**Algoritmo mental:** EDA general → por cada hallazgo raro, ¿aislado o sistemático? → si sistemático, buscar causa (unidad, formato) antes de borrar → si aislado sin causa, `NaN` y documentar → imputación estadística siempre dentro del pipeline (nunca sobre el dataset completo antes del split, es fuga de información).

### Eliminar filas o registros — referencia rápida

```python
# Por condición directa
df = df[df['col'] >= 0]                 # te quedas con lo que cumple
df = df[~(df['col'] < 0)]               # eliminas lo que cumple (negando con ~)

# Varias condiciones (cada una entre paréntesis, & = y, | = o)
df = df[~((df['col'] < 15) | (df['col'] > 60))]

# Por índice, cuando ya identificaste las filas problemáticas aparte
indices_malos = df[df['col'] < 0].index
df = df.drop(index=indices_malos)

# Filas con nulo en una columna específica (target, ID)
df = df.dropna(subset=['col'])
```

---

## Encoders y Scalers — cuál usar

| Encoder | Cuándo | Riesgo si te equivocas |
|---|---|---|
| `OrdinalEncoder` | Categórica CON orden real | Usarlo en nominal → inventas un orden falso, distorsiona coeficientes |
| `OneHotEncoder` | Categórica nominal, sin orden | Usarlo en ordinal → pierdes la información de orden |
| `.map({...})` | Binaria | Bajo riesgo, equivalente a OneHot de 1 columna |

| Scaler | Cuándo |
|---|---|
| `StandardScaler` | Default para regresión lineal/Ridge/Lasso, KNN |
| `RobustScaler` | Si quedan outliers reales sin corregir (usa mediana/IQR, no media/std) |
| `MinMaxScaler` | Si necesitas rango acotado (ej. redes neuronales) |
| Ninguno | Modelos basados en árboles (Random Forest, árbol de decisión) — no lo necesitan |

**Ridge vs Lasso:** Ridge si hay multicolinealidad y quieres conservar todas las variables. Lasso si además quieres selección de variables (coeficientes en cero). Elastic Net si quieres ambos.

---

## Librería de patrones de código

```python
# 1. Detectar
df['col'].isnull().sum()
df.duplicated().sum()
df['col'].unique() / df['col'].value_counts(dropna=False)
(df.isnull().sum() / df.shape[0] * 100).round(2).sort_values(ascending=False)

# 2. Recodificar (normalizar categorías)
df['col'] = df['col'].replace({'viejo': 'nuevo'})
df['col'] = df['col'].map({'No': 0, 'Sí': 1})

# 3. Transformar por condición (errores de unidad/escala)
df.loc[condición, 'col'] = df.loc[condición, 'col'] / factor

# 4. Invalidar (a NaN cuando no hay fórmula de corrección)
df.loc[condición, 'col'] = np.nan

# 5. Forzar tipo
df['col'] = pd.to_numeric(df['col'], errors='coerce')
df['col'] = df['col'].round().astype(int)   # si aún hay NaN, usar 'Int64' en vez de int

# 6. Duplicados
df = df.drop_duplicates(subset=[...], keep='first')

# Investigar conflicto de ID (mismo ID, filas distintas)
repetidos = df[df['id'].duplicated(keep=False)]
for id_, grupo in repetidos.groupby('id'):
    print(id_, grupo.drop(columns='id').drop_duplicates().shape[0])  # >1 = conflicto real

# 7. Eliminar filas (target/ID nulo, o por condición)
df = df.dropna(subset=['col'])
df = df[condición]                 # te quedas con lo que SÍ cumple
df = df[~condición]                # eliminas lo que cumple (niega con ~)
df = df[~((df['col'] < a) | (df['col'] > b))]   # varias condiciones con & / |, cada una entre paréntesis
df = df.drop(index=df[condición].index)          # alternativa usando índices
```

**Notas importantes:**
- `.loc[condición, col] = valor`: se repite la condición en ambos lados cuando el valor también depende de filtrar — lado derecho calcula, lado izquierdo dice dónde escribir.
- `.replace()`/`.dropna()`/`.drop_duplicates()`/`.map()` NO modifican en el sitio — hay que reasignar (`df = ...`).
- ⚠️ Evitar `df.fillna(valor_escalar, inplace=True)` sin especificar columna — rellena TODO el DataFrame con ese valor.
- `describe()` cuenta como "count" solo los valores NO nulos — por eso cada columna tiene un count distinto.
- Percentil 50% en `describe()` = mediana. 25%/75% definen el IQR.
- `object` en `df.info()` = pandas no sabe más que "es texto"; no indica si es ordinal/nominal/binaria — eso lo decides tú con el diccionario de datos.
- Convertir a `int`/redondear NO depende del split (no es una estadística aprendida) — se puede hacer antes o después, la misma operación se aplica igual a train y test.
- Imputación (media/mediana/moda) SÍ debe aprenderse solo del train, nunca del dataset completo antes del split.

---

## Veredicto sobre los códigos de la plantilla oficial del examen

La plantilla que da la universidad trae mucho código de referencia, pero no todo vale la pena usarlo tal cual. Clasificación después de haber pasado por todo:

### Usar siempre (memorizar, aplica a cualquier caso)
- Imports básicos (pandas, numpy, sklearn, matplotlib, seaborn) — quitar SMOTE/RandomUnderSampler si el caso es regresión
- `pd.read_csv`, `df.copy()`
- `df.shape`, `df.info()`, `df.describe(include='all')` / `describe(include='object')`
- `df.isnull().sum()` (con `isna()` es un alias exacto, no hace falta usar los dos)
- `df.duplicated().sum()`, `.unique()`/`.nunique()`, `value_counts()`
- Filtrado condicional (`df[condición]`, `~condición`, `&`/`|`)
- `.replace()`, `.map()`, `.loc[condición, col] = valor`
- `np.where(...)` para columnas condicionales
- `drop_duplicates(subset=[...], keep=...)`
- Patrón `groupby('id').size()` para detectar IDs repetidos con conteo (el que usamos para el conflicto de `log_id`)
- `train_test_split`
- El bloque final de `ColumnTransformer` + `Pipeline` — esqueleto del punto 4

### Situacional (solo si el caso nuevo lo pide)
- `pd.to_datetime` + `sort_values` — solo si hay columna de fecha
- `pd.crosstab` — cruzar dos categóricas, no siempre necesario
- Crear columnas nuevas (`df['nueva'] = ...`) — solo si hay feature engineering real que justificar
- SMOTE/RandomUnderSampler — solo si es clasificación con clases desbalanceadas; en regresión no aplica
- `pairplot`/heatmap con muchas variables — útil pero puede saturarse, usar selectivamente

### Evitar o usar con cuidado
- `df.fillna(df['col2'].median(), inplace=True)` — bug: rellena TODO el DataFrame con la mediana de esa columna. Corregir a `df['col2'] = df['col2'].fillna(df['col2'].median())`.
- Bloque de eliminación de outliers por IQR que borra filas automáticamente para TODAS las numéricas — peligroso sin pensar, borra outliers reales junto con errores. Aplicar columna por columna, con justificación explícita de que ES un error.
- Ejemplos con columnas del dataset de vivienda (`Calificación`, `AreaHabitable`, `Cuartos`, `Baños`, `FrenteMar`) — solo sirven como referencia de sintaxis, hay que reescribirlos con las columnas del caso real.
- Imputación por razón entre columnas (`k = data['Calificación']/data['AreaHabitable']`) — muy específico del ejemplo de vivienda; la idea (imputar usando una relación conocida entre dos columnas) puede servir si el caso nuevo tiene algo similar, pero el código no es copiable tal cual.
- Bloques de gráficos que se repiten con ligeras variaciones — quedarse con una sola versión.

---

## Pipeline (punto 4, 20%)

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# 1. Separar X/y (data ya limpio del punto 2). Excluir target Y cualquier columna ID (no predictiva).
X = data.drop(columns=['target', 'id'])
y = data['target']

# 2. Split ANTES de imputar/escalar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Grupos de columnas según el punto 3
numeric_cols = [...]
ordinal_cols = [...]   # incluye binarias si las tratas como ordinal de 2 niveles
nominal_cols = [...]

# 4. Un sub-pipeline por tipo: imputar primero, transformar después
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[[...], [...]]))  # una lista de categorías por columna, en el mismo orden que ordinal_cols
])

nominal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 5. Juntar todo
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('ord', ordinal_transformer, ordinal_cols),
    ('nom', nominal_transformer, nominal_cols)
])

# 6. fit_transform SOLO en train
X_train_prep = preprocessor.fit_transform(X_train)

# 7. transform (sin fit) sobre test — ejemplo de aplicación pedido en el enunciado
X_test_prep = preprocessor.transform(X_test)

X_test_prep_df = pd.DataFrame(X_test_prep, columns=preprocessor.get_feature_names_out())
display(X_test_prep_df.head())
```

`handle_unknown='ignore'` evita que truene si en test aparece una categoría que no vio en train.

⚠️ **Verificación obligatoria después de armar los grupos de columnas:** `ColumnTransformer` descarta en silencio (sin error) cualquier columna que no asignes a ningún transformer. Antes de dar por bueno el pipeline, siempre correr:
```python
assert len(numeric_cols) + len(ordinal_cols) + len(nominal_cols) == X_train.shape[1]
```
Si no coincide, te falta clasificar alguna columna (nos pasó con `Frecuencia de competencia` en el caso de OlimpiAlpes — se quedó fuera de `numeric_cols` y se perdía silenciosamente).

### El mismo pipeline, anotado — qué cambia si es clasificación/KNN

Mismo código de arriba, con comentarios marcando exactamente los 2 puntos que cambian (todo lo demás es idéntico):

```python
# 1. Separar X/y — IGUAL en regresión y clasificación
X = data.drop(columns=['target', 'id'])
y = data['target']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y   # <- CAMBIA: SOLO clasificación. Mantiene la misma proporción de
                 #    clases en train/test. En regresión no existe (target continuo,
                 #    no hay "clases" que estratificar).
)

# 3. Grupos de columnas — IGUAL en ambos casos
numeric_cols = [...]
ordinal_cols = [...]
nominal_cols = [...]

# 4. Sub-pipelines — IGUAL en ambos casos (la preparación no depende del modelo final)
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())   # con KNN esto es AÚN MÁS crítico: usa distancias
                                    # directas entre puntos, sin escalar una variable
                                    # de rango grande domina toda la distancia
])

ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[[...], [...]]))
])

nominal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 5. ColumnTransformer — IGUAL en ambos casos
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('ord', ordinal_transformer, ordinal_cols),
    ('nom', nominal_transformer, nominal_cols)
])

# 6. fit_transform en train — IGUAL en ambos casos
X_train_prep = preprocessor.fit_transform(X_train)

# --- CAMBIA: bloque que solo existe en clasificación, y solo si hay desbalance ---
# from imblearn.over_sampling import SMOTE
# X_train_prep, y_train = SMOTE(random_state=42).fit_resample(X_train_prep, y_train)
# (NUNCA se aplica a test — el test debe reflejar la distribución real de clases)
# ------------------------------------------------------------------------------

# 7. transform sobre test — IGUAL en ambos casos
X_test_prep = preprocessor.transform(X_test)

X_test_prep_df = pd.DataFrame(X_test_prep, columns=preprocessor.get_feature_names_out())
display(X_test_prep_df.head())
```

Resumen: solo 2 cosas cambian dentro del código del pipeline (`stratify=y` en el split, y el bloque opcional de resampling después de `fit_transform`) — todo lo demás, letra por letra, es igual.

---

## Si el caso es clasificación en vez de regresión (ej. regresión logística o KNN)

La mayoría del trabajo NO cambia. Esto es lo que sí cambia:

### No cambia
- EDA y calidad de datos (punto 2): completitud, unicidad, validez, exactitud se detectan y corrigen exactamente igual.
- Encoders (`OrdinalEncoder`/`OneHotEncoder`): codificar categóricas no depende del algoritmo, depende de la variable.
- Imputación (mediana/moda dentro del pipeline, fit solo en train).
- Estructura del `ColumnTransformer` — mismo esqueleto.

### Sí cambia

**1b — Enfoque analítico:** target categórico (binario/multiclase) en vez de continuo → tarea = clasificación.
- Regresión logística: interpretable (coeficientes = log-odds), buena opción si se pide interpretabilidad.
- KNN: **no da coeficientes ni reglas interpretables** — si el caso exige interpretabilidad, es más difícil de justificar que logística o un árbol.

**Punto 2 — ahora sí aplica el balance de clases:**
```python
data[target].value_counts()
data[target].value_counts(normalize=True) * 100
```
Si está desbalanceado (ej. 90%/10%), es un hallazgo que hay que mencionar y justificar cómo se maneja.

**Punto 2 — relación con el target cambia de forma:**
- Antes (regresión): scatter numérica vs target continuo.
- Ahora (clasificación): boxplot de cada numérica **agrupada por clase del target** (mismo tipo de gráfico que usábamos para categóricas vs target, pero ahora el agrupador es el target):
```python
sns.boxplot(data=data, x=target, y='numerica')
```
- Categórica vs target categórico → `pd.crosstab(data['categorica'], data[target])` en vez de boxplot.

**Punto 3 — el escalado se vuelve más crítico con KNN:**
- Logística: sensible a escala (igual que regresión lineal) → `StandardScaler` sigue siendo necesario.
- KNN es **aún más sensible** — calcula distancias directas entre puntos; una variable con rango 0-7000 domina la distancia frente a una de rango 0-1 si no se escala. Con KNN, escalar no es opcional.
- Si hay desbalance de clases, aquí se justifica SMOTE/undersampling como parte de las transformaciones.

**Punto 4 — Pipeline:**
- Construcción del `ColumnTransformer` idéntica.
- Si se incluye balanceo de clases (SMOTE), no se puede meter dentro de un `sklearn.pipeline.Pipeline` normal — usar `imblearn.pipeline.Pipeline` en su lugar, para que SMOTE encadene bien con los demás pasos:
```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

pipeline_completo = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    # ('modelo', LogisticRegression())  # si el punto pidiera entrenar el modelo
])
```

---

## Pendiente
- Simulacro completo con un caso ficticio distinto (aplicar todo este archivo + `plantilla_respuestas_escritas.md` desde cero, sin ver las respuestas de OlimpiAlpes).
