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

## Tabla de decisión por dimensión de calidad

| Dimensión | Pregunta | Cómo detectar |
|---|---|---|
| Completitud | ¿faltan datos? | `isnull().sum()`, `% nulos` |
| Unicidad | ¿duplicados/IDs repetidos? | `duplicated().sum()` |
| Validez | ¿valores en el dominio esperado? | `.unique()`, tipos |
| Exactitud | ¿valores física/lógicamente posibles? | `describe()` min/max |
| Consistencia | ¿misma categoría escrita distinto? | comparar `.unique()` |

| Situación | Qué hacer |
|---|---|
| Nulos <5-10%, numérica | Imputar mediana (dentro del pipeline) |
| Nulos <5-10%, categórica | Imputar moda (dentro del pipeline) |
| Nulos >40-50% en una columna | Evaluar eliminar la columna |
| Nulos en el target | Eliminar filas (`dropna`) |
| Nulos concentrados en una categoría | Imputar por grupo (`groupby().transform()`) |
| Fila 100% duplicada | `drop_duplicates()` directo |
| Mismo ID, datos distintos (conflicto real) | Investigar causa; si no se resuelve: quedarse con la fila más completa, eliminar ambas, o asignar ID nuevo si son personas distintas — justificando la elección |
| Categoría escrita distinto | Normalizar con `.replace()`/`.map()` |
| Numérica cargada como texto | `pd.to_numeric(errors='coerce')` |
| Valor aislado sin patrón (outlier real) | Evaluar IQR/z-score; capear o dejar si el modelo lo tolera |
| Muchos valores fuera de rango con un patrón | Sospechar error sistemático de unidad — buscar el factor antes de descartar |
| Valor inválido sin causa recuperable | Convertir a `NaN`, imputar en el pipeline |

**Regla de oro outliers:** estadísticamente atípico ≠ error. Imposible → corregir/invalidar. Solo inusual pero posible → dejarlo y documentarlo (puede ser un caso real).

**Algoritmo mental:** EDA general → por cada hallazgo raro, ¿aislado o sistemático? → si sistemático, buscar causa (unidad, formato) antes de borrar → si aislado sin causa, `NaN` y documentar → imputación estadística siempre dentro del pipeline (nunca sobre el dataset completo antes del split, es fuga de información).

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

---

## Pendiente
- Simulacro completo con un caso ficticio distinto (aplicar todo este archivo + `plantilla_respuestas_escritas.md` desde cero, sin ver las respuestas de OlimpiAlpes).
