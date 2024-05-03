from collections import Counter
import sys, json
import psycopg2
import pandas as pd
import numpy as np

dates = []; 
values = ["Date","Time",'Global_reactive_power', 'Global_active_power', 'Voltage', 'Global_intensity', 'Sub_metering_1','Sub_metering_2', 'Sub_metering_3']
data = eval(sys.argv[1])

route1 = "C:\\Users\\Santiago D\\Documents\\Universidad\\2024-1\\Procesos 2\\Proyecto\\JSON\\pie1.json"
route2 = "C:\\Users\\Santiago D\\Documents\\Universidad\\2024-1\\Procesos 2\\Proyecto\\JSON\\pie2.json"
route3 = "C:\\Users\\Santiago D\\Documents\\Universidad\\2024-1\\Procesos 2\\Proyecto\\JSON\\pie3.json"

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
    
#asignación de nuevas etiquetas con rangos mas flexibles
S1avg = df3['Sub_metering_1'].mean()
S2avg = df3['Sub_metering_2'].mean()
S3avg = df3['Sub_metering_3'].mean()
S1 = df3['Sub_metering_1']
S2 = df3['Sub_metering_2']
S3 = df3['Sub_metering_3']

conditions1 = [
    ((S2>0)&(S2<(S3+0.25*S3avg))) | ((S3>0)&(S3<(S2+0.25*S2avg))), #Etiqueta 0:El consumo de ambas zonas es similar
    S2 > (S3+0.25*S3avg), #Etiqueta 1: consumo en zona 2 considerablemente mayor a zona 3
    S3 > (S2+0.25*S2avg), #Etiqueta 2: consumo en zona 3 considerablemente mayor a zona 2
    (S2 == 0) & (S3==0) #Etiqueta 3: el consumo es despreciable en ambas zonas    
]
conditions2 = [
    ((S2>0)&(S2<(S1+0.25*S1avg))) | ((S1>0)&(S1<(S2+0.25*S2avg))), #Etiqueta 0:El consumo de ambas zonas es similar
    S2 > (S1+0.25*S1avg), #Etiqueta 1: consumo en zona 2 considerablemente mayor a zona 1
    S1 > (S2+0.25*S2avg), #Etiqueta 2: consumo en zona 1 considerablemente mayor a zona 2
    (S2 == 0) & (S1==0) #Etiqueta 3: el consumo es despreciable en ambas zonas    
]
conditions3 = [
    ((S1>0)&(S1<(S3+0.25*S3avg))) | ((S3>0)&(S3<(S1+0.25*S1avg))), #Etiqueta 0:El consumo de ambas zonas es similar
    S1 > (S3+0.25*S3avg), #Etiqueta 1: consumo en zona 1 considerablemente mayor a zona 3
    S3 > (S1+0.25*S1avg), #Etiqueta 2: consumo en zona 3 considerablemente mayor a zona 1
    (S1 == 0) & (S3==0) #Etiqueta 3: el consumo es despreciable en ambas zonas    
]
    
choices = [0, 1, 2, 3]
y1 = np.select(conditions1, choices, default=4)
y1 = y1[y1 != 3]
y2 = np.select(conditions2, choices, default=4)
y2 = y2[y2 != 3]
y3 = np.select(conditions3, choices, default=4)
y3 = y3[y3 != 3]        

counter1 = Counter(y1)
counter2 = Counter(y2)
counter3 = Counter(y3)
distribution1 = [counter1.get(2, 0), counter1.get(1, 0), counter1.get(0, 0)]
distribution2 = [counter2.get(2, 0), counter2.get(1, 0), counter2.get(0, 0)]
distribution3 = [counter3.get(2, 0), counter3.get(1, 0), counter3.get(0, 0)]

payload1 = {"labels": ["Z1 > Z2", "Z2 > Z1", "Z1 = Z2"],
           "data": distribution2,
           "series":[]}
payload2 = {"labels": ["Z3 > Z1", "Z1 > Z3", "Z1 = Z3"],
           "data": distribution3,
           "series":["1","2","3"]}
payload3 = {"labels": ["Z2 > Z3", "Z3 > Z2", "Z2 = Z3"],
           "data": distribution1,
           "series":["1","2","3"]}

# Escribir los datos en el archivo JSON
with open(route1, "w") as json_file:
    json.dump(payload1, json_file, indent=4)
with open(route2, "w") as json_file:
    json.dump(payload2, json_file, indent=4)
with open(route3, "w") as json_file:
    json.dump(payload3, json_file, indent=4)

conexion.close()
cursor.close()

