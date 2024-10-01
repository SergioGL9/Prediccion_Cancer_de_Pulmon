import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Cargar el conjunto de datos
df = pd.read_csv('cancer_pulmon.csv')

# Separar características y etiquetas
X = df.drop('Cancer de pulmon', axis=1)
y = df['Cancer de pulmon']

# Normalizar las características
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# Dividir el conjunto de datos en entrenamiento y prueba con estratificación
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X_escalado, y, test_size=0.2, random_state=42, stratify=y)

# Entrenar el modelo de regresión logística con validación cruzada
modelo = LogisticRegression()
puntuaciones = cross_val_score(modelo, X_entrenamiento, y_entrenamiento, cv=5, scoring='accuracy')
print("Puntuaciones de exactitud para cada fold: ", puntuaciones)
print("Media de la exactitud con validación cruzada: ", puntuaciones.mean())

# Entrenar el modelo final con todos los datos de entrenamiento
modelo.fit(X_entrenamiento, y_entrenamiento)

# Evaluar el modelo
y_pred_entrenamiento = modelo.predict(X_entrenamiento)
y_pred_prueba = modelo.predict(X_prueba)
print("Datos de Entrenamiento - Informe de Clasificación")
print(classification_report(y_entrenamiento, y_pred_entrenamiento))
print("Datos de Prueba - Informe de Clasificación")
print(classification_report(y_prueba, y_pred_prueba))
print("Datos de Prueba - Exactitud:", accuracy_score(y_prueba, y_pred_prueba))

# Guardar el modelo y el escalador
with open('modelo/modelo_regresion_logistica.pkl', 'wb') as f:
    pickle.dump(modelo, f)
with open('modelo/escalador.pkl', 'wb') as f:
    pickle.dump(escalador, f)

print('Modelo y escalador entrenados y guardados exitosamente.')