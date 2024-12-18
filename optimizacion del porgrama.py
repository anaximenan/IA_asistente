import pandas as pd
from prophet import Prophet  # Cambio aquí: usamos el nuevo paquete `prophet`
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Función para leer los datos desde el archivo CSV
def obtener_datos_csv(filename):
    """
    Lee los datos del archivo CSV.
    :param filename: Nombre del archivo CSV con el historial.
    :return: DataFrame con los datos.
    """
    return pd.read_csv(filename)

# Función para preparar los datos
def preparar_datos(data_articulo):
    """
    Prepara los datos para el modelo Prophet.
    :param data_articulo: Subconjunto de datos para un artículo específico.
    :return: DataFrame preparado.
    """
    # Crear una columna de fechas a partir de año y semana
    data_articulo['date'] = pd.to_datetime(data_articulo['ejercicio'].astype(str) + 
                                           data_articulo['semana'].astype(str) + '0', format='%Y%U%w')
    # Renombrar las columnas para Prophet
    data_articulo = data_articulo.rename(columns={'date': 'ds', 'Unidades': 'y'})
    return data_articulo[['ds', 'y']]

# Función para entrenar y predecir
def entrenar_y_predecir(data_articulo, semanas_futuras=4):
    """
    Entrena un modelo Prophet y realiza predicciones.
    :param data_articulo: Subconjunto de datos para un artículo específico.
    :param semanas_futuras: Número de semanas a predecir.
    :return: Métricas de evaluación y predicciones.
    """
    # Preparar los datos
    data_articulo = preparar_datos(data_articulo)

    # Separar datos en entrenamiento y prueba (últimas 10 semanas para prueba)
    data_train = data_articulo[:-10]
    data_test = data_articulo[-10:]

    # Inicializar y entrenar el modelo Prophet
    modelo = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    modelo.fit(data_train)

    # Realizar predicciones para las fechas de prueba
    future_test = data_test[['ds']]
    predicciones_test = modelo.predict(future_test)

    # Calcular métricas
    mae = mean_absolute_error(data_test['y'], predicciones_test['yhat'])
    rmse = np.sqrt(mean_squared_error(data_test['y'], predicciones_test['yhat']))

    # Predecir semanas futuras
    future = modelo.make_future_dataframe(periods=semanas_futuras, freq='W')
    forecast = modelo.predict(future)

    # Extraer predicciones futuras
    predicciones_futuras = forecast[['ds', 'yhat']].tail(semanas_futuras)

    return mae, rmse, predicciones_futuras

# Guardar las predicciones en un archivo CSV
def guardar_en_csv(resultados, filename='predicciones.csv'):
    """
    Guarda las predicciones en un archivo CSV.
    :param resultados: Lista de diccionarios con las predicciones.
    :param filename: Nombre del archivo CSV.
    """
    df = pd.DataFrame(resultados)
    df.to_csv(filename, index=False)
    print(f"Predicciones guardadas en {filename}.")

# Flujo principal
def main():
    # Leer los datos desde el archivo CSV
    data = obtener_datos_csv('datos_ventas.csv')

    # Lista para almacenar los resultados
    resultados = []

    # Generar predicciones para cada artículo
    for articulo_id in data['articulo_id'].unique():
        data_articulo = data[data['articulo_id'] == articulo_id]
        mae, rmse, predicciones_futuras = entrenar_y_predecir(data_articulo)

        # Guardar métricas y predicciones
        for _, row in predicciones_futuras.iterrows():
            resultados.append({
                'articulo_id': articulo_id,
                'mae': mae,
                'rmse': rmse,
                'fecha': row['ds'],
                'prediccion': row['yhat']
            })

    # Guardar las predicciones en un archivo CSV
    guardar_en_csv(resultados)

if __name__ == "__main__":
    main()