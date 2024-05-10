import psycopg2
import sys, json
import pandas as pd

dates = []; 
values = ["Date","Time",'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1','Sub_metering_2', 'Sub_metering_3']
data = eval(sys.argv[1])

#Coneccion con la base de datos
conexion = psycopg2.connect(database="EnergyConsumption", user="postgres", password="postgres")
cursor = conexion.cursor()

query = f"""
    SELECT * FROM "LogSensor" 
    WHERE "Date" >= %s AND "Date" <= %s 
    ORDER BY "Date" 
"""
#Ejecutamos la seleccion para hallar los datos 
cursor.execute(query,(data[0], data[1]))

# Obtener los resultados
resultados = cursor.fetchall()

# Convertir los resultados a un DataFrame de Pandas
df = pd.DataFrame(resultados, columns=values)

#select just the columns needed
df3 = df.loc[:int(len(df)), ['Global_active_power', 'Global_reactive_power', 
                               'Voltage',	'Global_intensity', 'Sub_metering_1',	'Sub_metering_2',
                               'Sub_metering_3']]
    
df3['Global_apparent_power'] = (df3['Global_active_power']*2 + df3['Global_reactive_power']*2)*(1/2)
df3['Power_factor'] = df3['Global_active_power']/df3['Global_apparent_power']

# Iterar sobre cada columna del DataFrame
for columna in df3.columns:
    
    rounded = df3[columna].round(1)
    # Obtener los valores únicos y sus frecuencias
    vc = rounded.value_counts().sort_index()

    # Extraer los valores y sus frecuencias como listas
    valores = vc.index.tolist()
    frecuencias = vc.values.tolist()
    # Filtrar los valores y sus sumas de frecuencias para eliminar los valores con frecuencia igual a 1
    valores_filtrados = [grupo for grupo, suma in zip(valores, frecuencias) if suma > 100]
    frecuencias_filtradas = [suma for suma in frecuencias if suma > 100]

    
    payload = {"labels": valores_filtrados,
               "data": frecuencias_filtradas,
               "series":[]}

    # Agregar los valores y sus frecuencias a la lista
    with open(f"C:\\Users\\Santiago D\\Documents\\Universidad\\2024-1\\Procesos 2\\Proyecto\\JSON\\{columna}.json", "w") as json_file:
        json.dump(payload, json_file, indent=4)
        
conexion.close()
cursor.close()