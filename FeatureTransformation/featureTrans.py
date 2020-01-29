'''
Created on 29-Jan-2020

@author: akhil
'''
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="color", outputCol="color_indexed")

#Next we call the fit() method to initiate the learning process.
indexer_model = indexer.fit(data)

indexed_data= indexer_model.transform(data)
# to view the data
indexed_data.show()



###########################################One Hot Encoding#######################################

from pyspark.ml.feature import OneHotEncoderEstimator
ohe = OneHotEncoderEstimator(inputCols=["color_indexed"], outputCols=["color_ohe"])
ohe_model = ohe.fit(indexed_data)
encoded_data = ohe_model.transform(indexed_data)
encoded_data.show()


############################ Feature Scaling #####################################
'''    
This diversity in scale could cause a lot of problems in some machine learning algorithms e.g. KMeans. 
This is because the algorithm may treat some variables as more dominant according to their value range. 
For example: consider a dataset about employees. We may have a years of experience column that ranges between 0 → 30 
and a salary column with values in thousands. But this does not mean that the salary column is more dominant!
To solve this problem we transform the values to be at the same scale. There are a lot of transformation methods, 
we will look at two of them.
Note that scalers are applied on Vector Data Types that is why we need to collect the features using a VectorAssembler first:

'''
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=data.columns[1:], outputCol="features")
data_2 = assembler.transform(data)

################### StandardScaler###################

from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(data_2)
scaled_data = scaler_model.transform(data_2)



################ MinMaxScaler #######################
from pyspark.ml.feature import MinMaxScaler
scaler = MinMaxScaler(min=0, max=1, inputCol='features', outputCol='features_minmax')
scaler_model = scaler.fit(data_2)
data_3 = scaler_model.transform(data_2)


################ Principal Component Analysis ###################
data = spark.read.csv('./datasets/digits.csv', header=True, inferSchema=True)
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=data.columns[1:], outputCol='features')
data_2 = assembler.transform(data)

from pyspark.ml.feature import PCA
pca = PCA(k=2, inputCol='features', outputCol='features_pca')
pca_model = pca.fit(data_2)
pca_data = pca_model.transform(data_2).select('features_pca')










