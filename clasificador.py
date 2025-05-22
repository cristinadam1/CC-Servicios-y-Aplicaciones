from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, isnan, count
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Creo la sesión de Spark
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

# Pipeline para NaiveBayes: features no negativas
pipeline_minmax = Pipeline(stages=[indexer, assembler, MinMaxScaler(inputCol="features_raw", outputCol="features")])
data_minmax = pipeline_minmax.fit(df).transform(df).select("features", "label")

# 5. Partición de datos
# Para el resto de modelos
train_std, test_std = data_std.randomSplit([0.8, 0.2], seed=42)

# Para NaiveBayes
train_minmax, test_minmax = data_minmax.randomSplit([0.8, 0.2], seed=42)

# 6. Modelos
modelos = {
    "LogisticRegression_1": LogisticRegression(maxIter=10, regParam=0.01),
    "LogisticRegression_2": LogisticRegression(maxIter=20, regParam=0.1),
    "RandomForest_1": RandomForestClassifier(numTrees=10, maxDepth=5),
    "RandomForest_2": RandomForestClassifier(numTrees=30, maxDepth=10),
    "DecisionTree_1": DecisionTreeClassifier(maxDepth=20),
    "DecisionTree_2": DecisionTreeClassifier(maxDepth=10),
    "NaiveBayes_1": NaiveBayes(smoothing=1.0),
    "NaiveBayes_2": NaiveBayes(smoothing=0.5)
}


# Evaluadores
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1")
evaluator_precision = MulticlassClassificationEvaluator(metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(metricName="weightedRecall")


# 7. Entrenamiento y evaluación
print("\n Resultados:")
for nombre, modelo in modelos.items():
    if "NaiveBayes" in nombre:
        m = modelo.fit(train_minmax)
        predicciones = m.transform(test_minmax)
    else:
        m = modelo.fit(train_std)
        predicciones = m.transform(test_std)

    auc = evaluator.evaluate(predicciones)
    f1 = evaluator_f1.evaluate(predicciones)
    precision = evaluator_precision.evaluate(predicciones)
    recall = evaluator_recall.evaluate(predicciones)
    print(f"{nombre}: AUC={auc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    #print(f"{nombre}: AUC = {auc:.4f}")


# 8. Cierre de Spark
spark.stop()
