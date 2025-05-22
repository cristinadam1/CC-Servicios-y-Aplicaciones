# PRACTICA 3 - CC
## Objetivos de la práctica
Los objetivos formativos de esta práctica se centran en que el alumno adquiera los siguientes conocimientos y habilidades:

1. Conocimiento de la infraestructura de cómputo para Big Data y su despliegue en plataformas de cloud computing.
2. Manejo del sistema HDFS.
3. Conocimiento de frameworks para el procesamiento de los datos de gran volumen y aplicación de técnicas de Minería de Datos para la extracción de conocimiento.
4. Uso de las herramientas básicas proporcionadas por la MLLib de Spark.

## Configuración del entorno con Docker
Primero creo mi estructura de carpetas

    mkdir practica3
    cd practica3
    mkdir spark

### docker-compose.yaml
A continuación creo el archivo docker-compose.yaml, en el que se va a explicar cómo levantar varios contenedores y cómo se comunican entre ellos.
En este archivo defino lo siguiente:

- 1 contenedor NameNode (HDFS)
- 1 contenedor DataNode (HDFS)
- 1 contenedor Spark, con Python instalado

### Dockerfile
A continuación creo el archivo Dockerfile en la carpeta /spark (spark/Dockerfile), aquí explicamos a Docker cómo construir el contenedor Spark con Python y las librerías que van a hacer falta.

1. Parto de una imagen con Spark ya instalado
2. Añado Python 3 y pip
3. Instalo numpy 

## Levantar los contenedores
Desde la carpeta practica3 utilizo el siguiente comando:

    docker compose up -d

Lo cual va a levantar los siguientes contenedores:

- namenode: donde se gestiona el sistema de archivos HDFS
- datanode: almacén de los bloques de datos en el sistema HDFS
- spark: entorno para ejecutar tus scripts de análisis con PySpark

## Acceso a los contenedores
Para conectarme al contenedor spark utilizo:

    docker exec -it spark bash

Para conectarme al contenedor namenode utilizo:

    docker exec -it namenode bash

## Subir datos a HDFS
Descargo el dataset y lo copio en ~/data

    cp half_celestial.csv ~/data/

En el contenedor namenode hago lo siguiente:

    hdfs dfs -mkdir -p /user/CCSA2425/cristinadam
    hdfs dfs -put /data/half_celestial.csv /user/CCSA2425/cristinadam/
    hdfs dfs -ls /user/CCSA2425/cristinadam

## Creación del clasificador y explicación del código
Creo el archivo clasificador.py

1. Inicio la sesión de Spark
Primero, creo una sesión de Spark con el nombre "Practica3-MLlib", para usar las funcionalidades de procesamiento distribuido de PySpark. También uso la siguiente línea para reducir el nivel de los mensajes del sistema a solo errores para evitar información innecesaria durante la ejecución.

![cap4](/img/c4.png)

2. Cargar el conjunto de datos
Leo el archivo CSV que había almacenado en HDFS, con cabecera y separación por punto y coma.

![cap5](/img/c5.png)

3. Exploración inicial
Imprimo la estructura del DataFrame para conocer los tipos de cada columna. También calculo los valores nulos por columna y se enseño la distribución de clases (galaxy y star) en la variable objetivo.

![cap6](/img/c6.png)

4. Preprocesamiento
Transformo la variable "type", de tipo cadena, en una columna numérica label usando StringIndexer. Después, defino un conjunto de 10 variables numéricas como características (features).

Mediante VectorAssembler, combino esas columnas en un único vector features_raw.

Para los modelos que toleran valores negativos (como regresión logística o Random Forest), aplico StandardScaler, que centra los datos en media cero y varianza uno. En cambio, para Naive Bayes (que exige entradas no negativas), uso MinMaxScaler, que transforma los datos al rango [0,1].

![cap7](/img/c7.png)

Así, genero dos conjuntos distintos: data_std para los dos modelos y data_minmax exclusivamente para Naive Bayes.

5. División del conjunto de datos
Divido los dos conjuntos (data_std y data_minmax) en entrenamiento (80%) y prueba (20%) como se indica en el guión de prácticas, usando una semilla fija para garantizar reproducibilidad.

![cap8](/img/c8.png)

6. Definición de modelos
Defino seis modelos en total, agrupados en tres técnicas distintas:

- Regresión logística con dos configuraciones distintas de regularización (regParam).
- Random Forest, variando el número de árboles y la profundidad máxima.
- Naive Bayes, con dos valores de suavizado (smoothing).

![cap3](/img/c3.png)

7. Entrenamiento y evaluación
Recorro cada modelo y lo entreno sobre el conjunto de datos correspondiente (train_std o train_minmax). Luego, calculo la métrica AUC sobre el conjunto de prueba.

![cap9](/img/c9.png)

8. Finalización
Cierro la sesión de Spark para liberar los recursos utilizados

    spark.stop()

### Para ejecutarlo
Como no quería tener que estar copiando el archivo "clasificador.py" cada vez que lo modificaba, he optado por montar mi carpeta en el contenedor. 
Para esto hago lo siguiente:

1. En mi docker-compose.yaml añado la siguiente línea

    volumes:
      - ./:/workspace

Esto hace que todo lo que tengo en practica3/ (mi carpeta local) aparezca en el contenedor Spark como /workspace.

2. Reinicio los contenedores 

    docker compose down
    docker compose up -d

Esto aplica los cambios del docker-compose.yaml

3. Ejecuto el script directamente desde la carpeta montada
Entro al contenedor Spark

    docker exec -it spark bash

y ejecuta el script desde /workspace

De esta forma, cada vez que edite clasificador.py en mi máquina local, el contenedor verá los cambios automáticamente, sin necesidad de volver a copiarlo.

## Evaluación de los modelos y resultados obtenidos
El conjunto de datos utilizado contiene 1.000.000 de instancias, distribuidas casi perfectamente entre las dos clases objetivo: galaxias (500.002) y estrellas (499.998). Todas las variables numéricas están completas, sin presencia de valores nulos, lo que ha facilitado el preprocesamiento sin necesidad de imputaciones.

![cap2](/img/c2.png)

Para abordar el problema de clasificación binaria, he usado 3 técnicas de construcción de clasificadores, probando 2 parametrizaciones de cada uno de los algoritmos. He usado como métrica principal el Área Bajo la Curva ROC (AUC), que permite comparar el rendimiento de los clasificadores teniendo en cuenta tanto la sensibilidad como la especificidad del modelo.

![cap3](/img/c3.png)

Los resultados obtenidos han los siguientes:

![cap10](/img/c10.png)

Con estos resultados, podemos decir que:

1. Los modelos de Random Forest son los que tienen mejor rendimiento. El segundo modelo (RandomForest_2), con más árboles (30) y mayor profundidad (10), ha alcanzado un AUC de 0.9686, lo que indica una capacidad muy alta para distinguir entre galaxias y estrellas. El incremento en el número de árboles y la profundidad ha mejorado claramente la capacidad de generalización del modelo.

2. La regresión logística también ha tenido un buen rendimiento, especialmente en su configuración más simple (LogisticRegression_1). Sin embargo, aumentar la regularización en LogisticRegression_2 ha disminuido el rendimiento, lo que sugiere que un exceso de penalización ha limitado la capacidad del modelo para ajustar los datos.

3. Los modelos de Naive Bayes son los que han tenido los peores resultados, con un AUC de apenas 0.6109 en ambas configuraciones. A pesar de usar MinMaxScaler para asegurarme de que las variables tuvieran valores no negativos, la suposición de independencia entre variables que realiza este clasificador no se ajusta adecuadamente a la complejidad del conjunto de datos.

Así que, como resumen puedo afirmar que el modelo RandomForest_2 es el más eficaz en este caso, seguido por LogisticRegression_1. Naive Bayes, por su parte, no es el más adecuado para este problema en particular.

## Problemas encontrados
### Problema 1:
En el clasificador estaba obteniendo error por no separar bien las columanas. Para solucionarlo añado en la línea 15:

    sep=';'

### Problema 2:
No encontraba los errores porque me aparecía demasiada información en la terminal. 
Para solucionarlo añado la siguiente línea de código a clasificador.py spark.
    
    sparkContext.setLogLevel("ERROR")

### Problema 3
Al ejecutar el clasificador acababa en "Killed" esto era debido a que se estaba quedando sin memoria justo durante el entrenamiento de uno de los modelos.
Para solucionarlo ejecuto el clasificador con este comando:

    spark-submit --master local[2] --executor-memory 4g /workspace/clasificador.py

- local[2]: usa solo 2 núcleos (evita sobrecargar).
- --executor-memory 4g: da más RAM al proceso.

Además, también me he dado cuenta de que RandomForest_2 (50 árboles, maxDepth=10) avanzaba bastante y casi terminaba, pero acababa fallando con el mensaje Killed, los mensajes indicaban un problema de recursos. Para solucionaro disminuyo el número de árboles a 30

### Problema 4
"requirement failed: Naive Bayes requires nonnegative feature values"

Este problema está relacionado con que el modelo Naive Bayes, exige que todas las features sean >= 0
En mi caso, esto aparecía porque estaba usando StandardScaler para todos los modelos. 
Para solucionarlo, hago que NaiveBayes use MinMaxScaler en lugar de StandardScaler, mientras el resto de modelos (LogisticRegression y RandomForest) siguen usando StandardScaler