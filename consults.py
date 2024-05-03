import sys, json
import psycopg2
import pandas as pd
import numpy as np

dates = []; values = []; number = 0
data = eval(sys.argv[1])
dates = data[2]; values = ["Date","Time"] + data[0]; number = int(data[1])

listToStr = '\",\"'.join([str(elem) for elem in values])
route = "C:\\Users\\Santiago D\\Documents\\Universidad\\2024-1\\Procesos 2\\Proyecto\\JSON\\data.json"

#Coneccion con la base de datos
conexion = psycopg2.connect(database="EnergyConsumption", user="postgres", password="postgres")
cursor = conexion.cursor()

query = f"""
    SELECT "{listToStr}" FROM "LogSensor" 
    WHERE "Date" >= %s AND "Date" <= %s 
    ORDER BY "Date" 
"""
#Ejecutamos la seleccion para hallar los datos 
cursor.execute(query,(dates[0], dates[1]))

# Obtener los resultados
resultados = cursor.fetchall()

# Convertir los resultados a un DataFrame de Pandas
df = pd.DataFrame(resultados, columns=values)

descripcion = df.describe()
mean =  ""; min = ""; max = ""; std = ""
for index in data[0]:
    mean1 = round(descripcion.loc['mean', index],4)
    min1  = round(descripcion.loc['min', index],4)
    max1  = round(descripcion.loc['max', index],4)
    std1  = round(descripcion.loc['std', index],4)
    
    mean = mean + index + " = " + f"{mean1}" + "\n"
    min  = min  + index + " = " + f"{min1}"  + "\n"
    max  = max  + index + " = " + f"{max1}"  + "\n"
    std  = std  + index + " = " + f"{std1}"  + "\n"

def calcular_promedio_por_hora(df, columnas):
    # Convertir la columna "Date" a formato de fecha
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convertir la columna "Time" a formato de tiempo si es necesario
    if isinstance(df['Time'].iloc[0], str):
        df['Time'] = pd.to_datetime(df['Time']).dt.time
    
    # Combinar las columnas "Date" y "Time" para crear una columna de fecha y hora completa
    df['datetime'] = df['Date'] + pd.to_timedelta(df['Time'].astype(str))

    # Calcular el promedio de las columnas especificadas por hora
    df['hour'] = df['datetime'].dt.hour
    hourly_avg = df.groupby([df['hour']])[columnas].mean()

    # Crear una lista de listas para almacenar los resultados
    promedio_por_hora = [[] for _ in range(len(data[0]))]
    for _, value in hourly_avg.iterrows():
        for i, col in enumerate(columnas):
            promedio_por_hora[i].append(value[col])

    return promedio_por_hora

def calcular_promedio_por_semana(df, columnas):
    # Convertir la columna "Date" a formato de fecha
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convertir la columna "Time" a formato de tiempo si es necesario
    if isinstance(df['Time'].iloc[0], str):
        df['Time'] = pd.to_datetime(df['Time']).dt.time
    
    # Combinar las columnas "Date" y "Time" para crear una columna de fecha y hora completa
    df['datetime'] = df['Date'] + pd.to_timedelta(df['Time'].astype(str))

    # Calcular el promedio de las columnas especificadas por semana
    df['weekday'] = df['datetime'].dt.dayofweek  # Lunes: 0, Domingo: 6
    df['week'] = df['datetime'].dt.isocalendar().week
    weekly_avg = df.groupby(['week', 'weekday'])[columnas].mean().reset_index()

    # Crear una lista de listas para almacenar los resultados
    promedio_por_semana = [[] for _ in range(len(columnas))]
    for i, col in enumerate(columnas):
        for weekday in range(7):
            promedio_por_semana[i].append(weekly_avg[weekly_avg['weekday'] == weekday][col].mean())

    return promedio_por_semana

def calcular_promedio_por_mes(df, columnas):
    # Convertir la columna "Date" a formato de fecha
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convertir la columna "Time" a formato de tiempo si es necesario
    if isinstance(df['Time'].iloc[0], str):
        df['Time'] = pd.to_datetime(df['Time']).dt.time
    
    # Combinar las columnas "Date" y "Time" para crear una columna de fecha y hora completa
    df['datetime'] = df['Date'] + pd.to_timedelta(df['Time'].astype(str))

    # Calcular el promedio de las columnas especificadas por mes
    df['month'] = df['datetime'].dt.month
    monthly_avg = df.groupby(['month', df['datetime'].dt.day])[columnas].mean()

    # Crear una lista de listas para almacenar los resultados
    promedio_por_mes = [[] for _ in range(len(columnas))]

    # Iterar sobre las columnas y calcular el promedio por mes
    for i, col in enumerate(columnas):
        column_avg = []
        for month, month_data in monthly_avg.groupby(level=0):
            if month in df['month'].unique():
                avg_per_day = month_data[col].values.tolist()
                column_avg.extend(avg_per_day)

        # Limitar el resultado a 30 días (si hay menos, se repiten los datos)
        column_avg = column_avg[:30]
        promedio_por_mes[i].extend(column_avg)

    return promedio_por_mes

payload = {}

if number == 1: 
    promedio_por_hora = calcular_promedio_por_hora(df, data[0])

    # Crear un vector con los valores del 0 al 23 representando las horas
    horas = list(range(len(promedio_por_hora[0])))
    
    payload = {"labels": horas,
               "data": promedio_por_hora,
               "series":data[0] }
        
elif number == 7:
    promedio_por_semana = calcular_promedio_por_semana(df, data[0])

    # Crear un vector con los valores del 0 al 7 representando de lunes a domingo
    dias = list(range(1, len(promedio_por_semana[0]) + 1))
    
    payload = {"labels": dias,
               "data": promedio_por_semana,
               "series":data[0] }
        
elif number == 30:
    promedio_por_mes = calcular_promedio_por_mes(df, data[0])
    
    # Crear un vector con los valores del 0 al 30 representando los dias del mes
    diasM = list(range(1, len(promedio_por_mes[0] ) + 1))
    
    payload = {"labels": diasM,
               "data": promedio_por_mes,
               "series":data[0]}

payload["Promedio"] = mean
payload["Max"] = max
payload["Min"] = min
payload["STD"] = std

# Escribir los datos en el archivo JSON
with open(route, "w") as json_file:
    json.dump(payload, json_file, indent=4)

conexion.close()
cursor.close()