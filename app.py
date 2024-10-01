from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo y el escalador
with open('modelo/modelo_regresion_logistica.pkl', 'rb') as f:
    model = pickle.load(f)
with open('modelo/escalador.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('indice.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = [
        data['Genero'], data['Edad'], data['Fuma'], data['Dedos_amarillos'], 
        data['Ansiedad'], data['Presion_de_grupo'], data['Enfermedad_cronica'], 
        data['Fatiga'], data['Alergia'], data['Jadeos'], data['Consumo_de_alcohol'], 
        data['Tos'], data['Falta_de_aire'], data['Dificultad_para_tragar'], data['Dolor_de_pecho']
    ]
    
    # Convertir datos a formato adecuado
    gender = 1 if features[0] == 'M' else 0
    features = [gender] + [int(features[1])] + [1 if x == 'Sí' else 0 for x in features[2:]]
    
    # Normalizar las características
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Realizar la predicción
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][prediction]

    result = 'SI' if prediction == 1 else 'NO'
    return render_template('resultado.html', result=result, proba=proba)

if __name__ == '__main__':
    app.run(debug=True)