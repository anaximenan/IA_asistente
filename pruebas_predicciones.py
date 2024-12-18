import mysql.connector
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Cargar los datos desde el archivo CSV
archivo_csv = 'datos_ventas.csv'
try:
    data = pd.read_csv(archivo_csv, encoding='utf-8')
    print(f"Datos cargados exitosamente desde {archivo_csv}")
except FileNotFoundError:
    print(f"Error: No se pudo encontrar el archivo {archivo_csv}")
    exit()

# Crear una lista para almacenar los resultados
resultados = []

# Obtener los artículos únicos
articulos = data['articulo_id'].unique()
lecturas = 0

for articulo_id in articulos:
    # Filtrar el dataframe por artículo
    data_articulo = data.loc[data['articulo_id'] == articulo_id].copy()
    data_articulo_12 = data_articulo.tail(12).copy()

    # Preparar los datos para la regresión lineal
    data_articulo_12['ejercicio'] = data_articulo_12['ejercicio']  # Mantener el año original
    data_articulo_12['semana'] = range(1, 13)  # Reasignar las semanas del 1 al 12
    
    X_lineal = data_articulo_12[['ejercicio', 'semana']]
    Y_lineal = data_articulo_12['Unidades']
    
    Y_lineal_nocero = Y_lineal[Y_lineal != 0]
    if Y_lineal_nocero.sum() == 0 or len(Y_lineal_nocero) < 4:
        print(f"No se puede realizar la regresión para el artículo {articulo_id}: Datos insuficientes o ventas son ceros.")
        continue

    # Ajustar el modelo de regresión lineal
    modelolineal = LinearRegression()
    modelolineal.fit(X_lineal, Y_lineal)
    
    coeficientes = modelolineal.coef_
    intercepto = modelolineal.intercept_
    regresionlinealy = coeficientes[1] * 13 + intercepto
    
    # Calcular promedios
    promedio = data_articulo_12['Unidades'].sum() / 12
    promedio_sin_extremos = (data_articulo_12['Unidades'].sum() - data_articulo_12['Unidades'].max() - data_articulo_12['Unidades'].min()) / 10

    # Dividir los datos en entrenamiento y prueba
    X = data_articulo[['articulo_id', 'ejercicio', 'semana']]  # Mantener los valores originales
    y = data_articulo['Unidades']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear el modelo Random Forest y entrenar
    estimadores = 100
    modelo = RandomForestRegressor(n_estimators=estimadores, random_state=42)
    modelo.fit(X_train, y_train)
    
    # Hacer predicciones y evaluar
    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Predicción para la siguiente semana
    ultimo_anio = data_articulo['ejercicio'].max()
    df_ultimo_anio = data_articulo[data_articulo['ejercicio'] == ultimo_anio]
    ultima_semana = df_ultimo_anio['semana'].max()
    
    if ultima_semana < 52:
        semana_siguiente = ultima_semana + 1
        anio_siguiente = ultimo_anio
    else:
        semana_siguiente = 1
        anio_siguiente = ultimo_anio + 1

    # Crear un DataFrame para la nueva entrada
    nueva_entrada = pd.DataFrame([[articulo_id, anio_siguiente, semana_siguiente]], 
                                 columns=['articulo_id', 'ejercicio', 'semana'])
    
    prediccion = modelo.predict(nueva_entrada)
    unidades_predichas = prediccion[0]

    # Imprimir las tres tablas:
    print(f"Artículo ID: {articulo_id}")
    
    # Primera tabla: data_articulo (15 registros)
    print("Primera tabla: data_articulo")
    print(data_articulo.tail(12))
    
    # Segunda tabla: data_articulo_12 (12 registros)
    print("Segunda tabla: data_articulo_12")
    print(data_articulo_12.tail(12))
    
    # Tercera tabla: data original con filtrado de artículo (12 registros)
    print("Tercera tabla: data original filtrado")
    print(data.loc[data['articulo_id'] == articulo_id].tail(12))

    lecturas += 1
    if lecturas > 2:  
        break