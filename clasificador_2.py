from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, isnan, count
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# 1. Crear sesión de Spark
spark = SparkSession.builder.appName("Practica3-MLlib").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. Cargo el dataset desde HDFS
path = "hdfs://namenode:8020/user/CCSA2425/cristinadam/half_celestial.csv"  
df = spark.read.csv(path, header=True, inferSchema=True, sep=';')

# 3. Exploración de datos
print("\n Estructura del dataset:")
df.printSchema()

print("\n Valores nulos por columna:")
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

print("\n Distribución de clases:")
df.groupBy("type").count().show()

# 4. Preprocesamiento
# Convierto la variable de salida (galaxy/star) a numérica
indexer = StringIndexer(inputCol="type", outputCol="label")

# Columnas numéricas que uso como features
features = [
    "expAB_z", "i", "q_r", "modelFlux_r", "expAB_i",
    "expRad_u", "q_g", "psfMag_z", "dec", "psfMag_r"
]

# Ensamblo y normalizo features
assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features")

# Pipeline para modelos que admiten negativos
pipeline_std = Pipeline(stages=[indexer, assembler, StandardScaler(inputCol="features_raw", outputCol="features")])
data_std = pipeline_std.fit(df).transform(df).select("features", "label")

# 5. Partición de datos
# Para el resto de modelos
train_std, test_std = data_std.randomSplit([0.8, 0.2], seed=42)

# 6. Modelos
modelos = {
    "GBTClassifier_1": GBTClassifier(maxIter=10, maxDepth=5),
    "GBTClassifier_2": GBTClassifier(maxIter=20, maxDepth=5),
}


# Evaluador
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

# 7. Entrenamiento y evaluación
print("\n Resultados:")
for nombre, modelo in modelos.items():

    m = modelo.fit(train_std)
    predicciones = m.transform(test_std)

    auc = evaluator.evaluate(predicciones)
    print(f"{nombre}: AUC = {auc:.4f}")


# 8. Cierre de Spark
spark.stop()
