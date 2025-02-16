#                                                                                                        #   
#    Semana 5 - Curso Visión por computador - Especialización en Intrligencia Artificial                 #                                                                                   #   
#    DVF                                                                                                 #
##########################################################################################################
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2

app = Flask(__name__)

print("Cargando el modelo...")
modelo = tf.keras.models.load_model('modelo_mnist.h5')
print("Modelo cargado exitosamente!")

def segmentar_digitos(imagen_array):
    """
    Segmenta múltiples dígitos en una imagen.
    Retorna una lista de imágenes, cada una conteniendo un dígito.
    """
    # Convertir a escala de grises si no lo está
    if len(imagen_array.shape) == 3:
        gris = cv2.cvtColor(imagen_array, cv2.COLOR_RGB2GRAY)
    else:
        gris = imagen_array

    # Binarización
    _, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar contornos de izquierda a derecha
    contornos = sorted(contornos, key=lambda x: cv2.boundingRect(x)[0])

    digitos_recortados = []
    for contorno in contornos:
        # Obtener el rectángulo que encierra el contorno
        x, y, w, h = cv2.boundingRect(contorno)
        
        # Filtrar contornos muy pequeños
        if w > 10 and h > 10:
            # Recortar el dígito
            digito = binaria[y:y+h, x:x+w]
            
            # Agregar padding para mantener la proporción
            alto, ancho = digito.shape
            if alto > ancho:
                diff = alto - ancho
                padding = diff // 2 + 5
                digito = cv2.copyMakeBorder(digito, 5, 5, padding, padding, cv2.BORDER_CONSTANT, value=0)
            else:
                diff = ancho - alto
                padding = diff // 2 + 5
                digito = cv2.copyMakeBorder(digito, padding, padding, 5, 5, cv2.BORDER_CONSTANT, value=0)
            
            # Redimensionar a 28x28
            digito = cv2.resize(digito, (28, 28))
            
            digitos_recortados.append(digito)

    return digitos_recortados

def procesar_imagen(datos_imagen):
    """
    Procesa la imagen para detectar y predecir múltiples dígitos.
    """
    # Decodificar la imagen base64
    datos_imagen = base64.b64decode(datos_imagen.split(',')[1])
    
    # Convertir a formato numpy
    imagen = Image.open(io.BytesIO(datos_imagen))
    imagen_array = np.array(imagen)
    
    # Segmentar los dígitos
    digitos = segmentar_digitos(imagen_array)
    
    resultados = []
    for digito in digitos:
        # Preparar para la predicción
        digito = digito.reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        # Realizar predicción
        prediccion = modelo.predict(digito)
        digito_predicho = np.argmax(prediccion[0])
        confianza = float(prediccion[0][digito_predicho])
        
        resultados.append({
            'digito': int(digito_predicho),
            'confianza': confianza
        })
    
    return resultados

@app.route('/')
def inicio():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        datos_imagen = request.json['imagen']
        resultados = procesar_imagen(datos_imagen)
        
        return jsonify({
            'exito': True,
            'resultados': resultados,
            'total_digitos': len(resultados)
        })

    except Exception as e:
        return jsonify({
            'exito': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)