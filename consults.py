import sys
import psycopg2
import json
import re

data = []; data2 = []; dates = []
data = sys.argv[1].split("\"")

for j in range(len(data)):
    if not j % 2 == 0:
        data2.append(data[j])
    j += 1
      
listToStr = '\",\"'.join([str(elem) for elem in data2])
route = "C:\\Users\\Santiago D\\Documents\\Universidad\\2024-1\\Procesos 2\\Proyecto\\JSON\\data.json"

#Coneccion con la base de datos
conexion1 = psycopg2.connect(database="EnergyConsumption", user="postgres", password="KinKon28")
cursor1 = conexion1.cursor()
cursor2 = conexion1.cursor()
#Ejecutamos la seleccion para hallar los datos 
cursor1.execute(f"SELECT \"{listToStr}\"  FROM \"LogSensor\" ORDER BY \"Date\" LIMIT 100")
cursor2.execute(f"SELECT \"Date\"  FROM \"LogSensor\" ORDER BY \"Date\" LIMIT 20")

for index in cursor2:
    # Convertir la tupla a una cadena de texto
    fecha_str = str(index)

    # Extraer los componentes de la fecha utilizando expresiones regulares
    match = re.match(r"\(datetime\.date\((\d+), (\d+), (\d+)\)\,\)", fecha_str)

    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        
        # Formatear los componentes en un string con el formato deseado
        fecha_string = "{:02d}/{:02d}/{:02d}".format(day, month, year % 100)

        dates.append(fecha_string)  # Output: '16/12/06'

# Lista de listas para almacenar los datos resultantes
datos_resultantes = []

# Iterar sobre los datos del cursor1 y almacenar cada posición en una lista diferente
for fila in cursor1:
    # Si es la primera fila, crear las sublistas vacías
    if not datos_resultantes:
        datos_resultantes = [[] for _ in fila]
    
    # Iterar sobre los elementos de la fila y agregarlos a las sublistas correspondientes
    for i, dato in enumerate(fila):
        datos_resultantes[i].append(dato)
    
payload = {"labels": dates,
           "data": datos_resultantes,
           "series":data2 }

# Escribir los datos en el archivo JSON
with open(route, "w") as json_file:
    json.dump(payload, json_file, indent=4)

conexion1.close()