'''
Created on 29-Jan-2020

@author: akhil
'''

data = spark.read.csv('./datasets/titanic.csv', inferSchema=True, header=True)

data = data.select(['Survived', 'Pclass', 'Gender', 'Age', 'SibSp', 'Parch', 'Fare'])

'''
Calculating Age Missing Values
Age is an important feature; it is not wise to drop it because of some missing values. What we could do is to 
fill missing values with the help of existing ones. This process is called Data Imputation. There are many
 available strategies, but we will follow a simple one that fills missing values with the mean value calculated 
 from the sample.Spark ML makes the job easy using the Imputer class. First, we define the estimator, fit it 
 to the model, then we apply the transformer on the data.
'''

from pyspark.ml.feature import Imputer
imputer = Imputer(strategy='mean', inputCols=['Age'], outputCols=['AgeImputed'])
imputer_model = imputer.fit(data)
data = imputer_model.transform(data)


from pyspark.ml.feature import StringIndexer
gender_indexer = StringIndexer(inputCol='Gender', outputCol='GenderIndexed')
gender_indexer_model = gender_indexer.fit(data)
data = gender_indexer_model.transform(data)

'''
Creating the Features VectorWe learned previously that Spark ML expects data to be represented in two
 columns: a features vector and a label column. We have the label column ready (Survived), 
so let us prepare the features vector.
'''

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['Pclass', 'SibSp', 'Parch', 'Fare', 'AgeImputed', 'GenderIndexed'], outputCol='features')
data = assembler.transform(data)


###Model Training 

from pyspark.ml.classification import RandomForestClassifier
algo = RandomForestClassifier(featuresCol='features', labelCol='Survived')
model = algo.fit(data)
predictions = model.transform(data)
predictions.select(['Survived','prediction', 'probability']).show()


####''''Model Evaluation '''''

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol='Survived', metricName='areaUnderROC')
evaluator.evaluate(predictions)



#####'''''Evaluation using scikit-learn''''

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))


