# Databricks notebook source
from pyspark.sql import SparkSession
import pyspark.sql as sparksql
spark = SparkSession.builder.appName('stroke').getOrCreate()
df = spark.read.csv('/FileStore/tables/train.csv', inferSchema=True,header=True)
df.printSchema()

# COMMAND ----------

df = df.na.fill('Unknown', subset=['smoking_status'])

# COMMAND ----------

import pandas
data = df.toPandas()
print(data.describe())
print(data.dtypes)
print('\nCount missing value of each column:')
print(data.isnull().sum())

# COMMAND ----------

data.groupby('stroke').size()   #check whether is an imbalanced dataset

# COMMAND ----------

import sklearn
print(sklearn.__version__)

# COMMAND ----------

# MAGIC %sh
# MAGIC #need to run ***ONCE*** to install SMOTE package
# MAGIC /home/ubuntu/databricks/python/bin/pip install 'imbalanced-learn<0.2.1'
# MAGIC pip freeze | grep imbalanced-learn

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install --upgrade pip

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

categoricalColumns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df_indexed = df

for categoricalCol in categoricalColumns:
  stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
  df_indexed = stringIndexer.fit(df_indexed).transform(df_indexed)
  
display(df_indexed)

# COMMAND ----------

df_new = df_indexed.select(['genderIndex',
                            'age',
                            'hypertension',
                            'heart_disease',
                            'ever_marriedIndex',
                            'work_typeIndex',
                            'Residence_typeIndex',
                            'avg_glucose_level',
                            'bmi',
                            'smoking_statusIndex',
                            'stroke'])
display(df_new)

# COMMAND ----------

d = df_new.toPandas()
print(d.shape)
print(d.dtypes)

# COMMAND ----------

import numpy
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=7)
x_val = d.values[:,0:10]
y_val = d.values[:,10]
X_res, y_res = sm.fit_sample(x_val, y_val)

feature=['genderIndex', 'age', 'hypertension', 'heart_disease', 'ever_marriedIndex', 'work_typeIndex', 'Residence_typeIndex', 'avg_glucose_level', 'bmi', 'smoking_statusIndex']
oversampled_df = pandas.DataFrame(X_res)
oversampled_df.columns = feature
oversampled_df = oversampled_df.assign(label = numpy.asarray(y_res))
oversampled_df = oversampled_df.sample(frac=1).reset_index(drop=True)

oversampling_attr = oversampled_df.values[:,0:10]
oversampling_label = oversampled_df.values[:,10]

print("oversampled_df", oversampled_df.groupby('label').size()) 


# COMMAND ----------

oversampled_df.describe()

# COMMAND ----------

spark_df = spark.createDataFrame(oversampled_df)
spark_df.printSchema()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

stages = []
assembler = VectorAssembler(inputCols=['genderIndex',
 'age',
 'hypertension',
 'heart_disease',
 'ever_marriedIndex',
 'work_typeIndex',
 'Residence_typeIndex',
 'avg_glucose_level',
 'bmi',
 'smoking_statusIndex'],outputCol='features')

stages += [assembler]

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(spark_df)
df_pipeline = pipelineModel.transform(spark_df)

# COMMAND ----------

train,test = df_pipeline.randomSplit([0.8,0.2])
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Measure the impact of core number

# COMMAND ----------

# Decision Tree (maxDepth=5, impurity="gini")
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

dt = DecisionTreeClassifier(labelCol='label',featuresCol='features')

# COMMAND ----------

dt_model = dt.fit(train)
dt_predictions = dt_model.transform(test)
dt_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
dt_acc = dt_acc_evaluator.evaluate(dt_predictions)
print(dt_acc)

# COMMAND ----------

# Random Forest (maxDepth=5, impurity="gini", numTrees=20, subsamplingRate=1.0)
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')

# COMMAND ----------

rf_model = rf.fit(train)
rf_predictions = rf_model.transform(test)
rf_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
rf_acc = rf_acc_evaluator.evaluate(rf_predictions)
print(rf_acc)

# COMMAND ----------

# Gradient-Boosted Tree Classifier (maxDepth=5, maxIter=5, subsamplingRate=1.0)

from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(featuresCol = 'features', labelCol = 'label', maxIter=5)

# COMMAND ----------

gbt_model = gbt.fit(train)
gbt_predictions = gbt_model.transform(test)
gbt_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
gbt_acc = gbt_acc_evaluator.evaluate(gbt_predictions)
print(gbt_acc)

# COMMAND ----------

# Logistic Regression Classifier (threshold=0.5, maxIter=10)

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)

# COMMAND ----------

lr_model = lr.fit(train)
lr_predictions = lr_model.transform(test)
lr_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
lr_acc = lr_acc_evaluator.evaluate(lr_predictions)
print(lr_acc)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Measure the impact of scale

# COMMAND ----------

tr1 = train.limit(15000)
tr2 = train.limit(30000)
tr3 = train.limit(45000)
tr4 = train.limit(60000)

# COMMAND ----------

# Decision Tree (maxDepth=5, impurity="gini")
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

dt = DecisionTreeClassifier(labelCol='label',featuresCol='features')

# COMMAND ----------

dt_model = dt.fit(tr1)
dt_predictions = dt_model.transform(test)
dt_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
dt_acc = dt_acc_evaluator.evaluate(dt_predictions)
print(dt_acc)

# COMMAND ----------

dt_model = dt.fit(tr2)
dt_predictions = dt_model.transform(test)
dt_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
dt_acc = dt_acc_evaluator.evaluate(dt_predictions)
print(dt_acc)

# COMMAND ----------

dt_model = dt.fit(tr3)
dt_predictions = dt_model.transform(test)
dt_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
dt_acc = dt_acc_evaluator.evaluate(dt_predictions)
print(dt_acc)

# COMMAND ----------

dt_model = dt.fit(tr4)
dt_predictions = dt_model.transform(test)
dt_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
dt_acc = dt_acc_evaluator.evaluate(dt_predictions)
print(dt_acc)

# COMMAND ----------

# Random Forest (maxDepth=5, impurity="gini", numTrees=20, subsamplingRate=1.0)
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')

# COMMAND ----------

rf_model = rf.fit(tr1)
rf_predictions = rf_model.transform(test)
rf_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
rf_acc = rf_acc_evaluator.evaluate(rf_predictions)
print(rf_acc)

# COMMAND ----------

rf_model = rf.fit(tr2)
rf_predictions = rf_model.transform(test)
rf_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
rf_acc = rf_acc_evaluator.evaluate(rf_predictions)
print(rf_acc)

# COMMAND ----------

rf_model = rf.fit(tr3)
rf_predictions = rf_model.transform(test)
rf_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
rf_acc = rf_acc_evaluator.evaluate(rf_predictions)
print(rf_acc)

# COMMAND ----------

rf_model = rf.fit(tr4)
rf_predictions = rf_model.transform(test)
rf_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
rf_acc = rf_acc_evaluator.evaluate(rf_predictions)
print(rf_acc)

# COMMAND ----------

# Gradient-Boosted Tree Classifier (maxDepth=5, maxIter=5, subsamplingRate=1.0)

from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(featuresCol = 'features', labelCol = 'label', maxIter=5)

# COMMAND ----------

gbt_model = gbt.fit(tr1)
gbt_predictions = gbt_model.transform(test)
gbt_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
gbt_acc = gbt_acc_evaluator.evaluate(gbt_predictions)
print(gbt_acc)

# COMMAND ----------

gbt_model = gbt.fit(tr2)
gbt_predictions = gbt_model.transform(test)
gbt_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
gbt_acc = gbt_acc_evaluator.evaluate(gbt_predictions)
print(gbt_acc)

# COMMAND ----------

gbt_model = gbt.fit(tr3)
gbt_predictions = gbt_model.transform(test)
gbt_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
gbt_acc = gbt_acc_evaluator.evaluate(gbt_predictions)
print(gbt_acc)

# COMMAND ----------

gbt_model = gbt.fit(tr4)
gbt_predictions = gbt_model.transform(test)
gbt_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
gbt_acc = gbt_acc_evaluator.evaluate(gbt_predictions)
print(gbt_acc)

# COMMAND ----------

# Logistic Regression Classifier (threshold=0.5, maxIter=10)

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)

# COMMAND ----------

lr_model = lr.fit(tr1)
lr_predictions = lr_model.transform(test)
lr_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
lr_acc = lr_acc_evaluator.evaluate(lr_predictions)
print(lr_acc)

# COMMAND ----------

lr_model = lr.fit(tr2)
lr_predictions = lr_model.transform(test)
lr_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
lr_acc = lr_acc_evaluator.evaluate(lr_predictions)
print(lr_acc)

# COMMAND ----------

lr_model = lr.fit(tr3)
lr_predictions = lr_model.transform(test)
lr_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
lr_acc = lr_acc_evaluator.evaluate(lr_predictions)
print(lr_acc)

# COMMAND ----------

lr_model = lr.fit(tr4)
lr_predictions = lr_model.transform(test)
lr_acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
lr_acc = lr_acc_evaluator.evaluate(lr_predictions)
print(lr_acc)

# COMMAND ----------


