import mysql.connector
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# conexion a base de datos
cnx = mysql.connector.connect(
    host = 'qc-megafresh.cgjn6lnuo5l2.us-east-1.rds.amazonaws.com',
    user = 'admin',
    password = 'Ferp1102',
    database='predicciones_truper'
)
#consulta a base de datos
consulta = "call diario_ventas_semanal_todos('2021-01-03','2024-10-19');"
#guardar los datos en un dataframe
data = pd.read_sql(consulta, cnx)

articulos = data['articulo_id'].unique()
lecturas=0
query_acumulado = []


for articulo_id in articulos:
    # filtar el dataframe historico al solo la infomración del articulo
    data_articulo = data.loc[data['articulo_id'] == articulo_id]
    data_articulo_12 = data.loc[data['articulo_id'] == articulo_id].head(12)
    
    data_articulo_12.loc[:,'ejercicio'] = 1
    data_articulo_12.loc[:,'semana'] = range(1,13)
    #if data_articulo_12['Unidades'].sum() == 0 or len(data_articulo_12) < 2 or (data_articulo_12['Unidades'].sum() - data_articulo_12['Unidades'].max()- data_articulo_12['Unidades'].min())<=0:
    #    print(f"No se puede realizar la regresión para el artículo {articulo_id}: Datos insuficientes o todas las ventas son ceros.")
    #    continue
    # Variables independientes: 'ejercicio' (año) y 'semana'
    X_lineal = data_articulo_12[['ejercicio', 'semana']]  # Variables independientes (ejercicio y semana) 
    # Variables dependientes: 'unidades'
    Y_lineal = data_articulo_12['Unidades']       # Variable dependiente (unidades vendidas)
    
    Y_lineal_nocero = Y_lineal[Y_lineal !=0]
    # Asegúrate de que y tenga más de un valor único
    #if Y_lineal_nocero.nunique() < 4:
    #    print(f"No se puede realizar la regresión para el artículo {articulo_id}: Ventas constantes o insuficientes.")
    #    continue
    # Ajustamos el modelo de regresión lineal
    modelolineal = LinearRegression()
    modelolineal.fit(X_lineal,Y_lineal)    
    # Cálculos de regresión lineal
    coeficientes = modelolineal.coef_  # Coeficientes de la regresión (para ejercicio y semana) o pendiente (m)
    intercepto = modelolineal.intercept_  # Intercepto (punto donde la línea cruza el eje Y)
    # Cálculos de promedios
    promedio = data_articulo_12['Unidades'].sum() /12   # Promedio de unidades vendidas se cambio por calculo manual necetas un curso de sklearn.linear_model import LinearRegression

    # Cálculo del promedio sin el valor máximo y mínimo
    promedio_sin_extremos = (data_articulo_12['Unidades'].sum()-data_articulo_12['Unidades'].max()-data_articulo_12['Unidades'].min()) /10  # Promedio sin extremos

    regresionlinealy = coeficientes[1]*13 + intercepto
    
    
    # Codificar las variables categóricas (articulo_id, cliente_id, codigo_postal)
    label_encoder = LabelEncoder()
    data_articulo['articulo_id'] = label_encoder.fit_transform(data_articulo['articulo_id'])
    data_articulo['ejercicio'] = label_encoder.fit_transform(data_articulo['ejercicio'])
    data_articulo['semana'] = label_encoder.fit_transform(data_articulo['semana'])
    # seleccionar variables caracteristicas y objetivo
    X = data_articulo[['articulo_id','ejercicio','semana']]
    y = data_articulo['Unidades']
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Crear el modelo Random Forest
    estimadores = 100
    modelo = RandomForestRegressor(n_estimators=estimadores, random_state=42)

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = modelo.predict(X_test)

    # Evaluar el modelo
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # Predicción para la siguiente semana
    ultimo_anio = data_articulo['ejercicio'].max()
    
    # Filtrar las filas que corresponden al último año
    df_ultimo_anio = data_articulo[data_articulo['ejercicio'] == ultimo_anio]
    ultima_semana = df_ultimo_anio['semana'].max()

    # Calcular la siguiente semana y el año
    if ultima_semana < 52:
        semana_siguiente = ultima_semana + 1
        anio_siguiente = ultimo_anio
    else:
        semana_siguiente = 1
        anio_siguiente = ultimo_anio + 1

    nueva_entrada = [[articulo_id, anio_siguiente, semana_siguiente]]
    prediccion = modelo.predict(nueva_entrada)
    unidades_predichas = prediccion[0]
    lecturas = lecturas+1
    query = f"""
        insert into predicciones (articulo_id,mean_absolute_error, root_mean_squared_error, ejercicio, semana,estimators, random_state, fecha_hora, prediccion, promedio, regresion_lineal) 
            values({articulo_id}, {mae}, {rmse}, {anio_siguiente}, {semana_siguiente},{estimadores}, 42, current_timestamp(),{unidades_predichas}, {promedio_sin_extremos},{regresionlinealy})
    """
    print(articulo_id)
    print(data_articulo.head(15))
    print(data_articulo_12.head(12))
    print(data.loc[data['articulo_id'] == articulo_id].head(12))
    if lecturas>2:
        break
    continue
    cnx1 = mysql.connector.connect(
        host = 'qc-megafresh.cgjn6lnuo5l2.us-east-1.rds.amazonaws.com',
        user = 'admin',
        password = 'Ferp1102',
        database='predicciones_truper'
    )
    cursor = cnx1.cursor()
    cursor.execute(query)
    cnx1.commit()
    cursor.close()
    cnx1.close()
    
    