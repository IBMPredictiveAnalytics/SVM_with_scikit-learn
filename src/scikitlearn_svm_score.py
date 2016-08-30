
script_details = ("scikitlearn_svm.py",0.5)

from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext

from sklearn import preprocessing, svm
from pyspark.sql.types import StructField, StructType, StringType
import time
import sys
import os

import json
ascontext=None
if len(sys.argv) > 1 and sys.argv[1] == "-test":
    import os
    sc = SparkContext('local')
    sqlCtx = SQLContext(sc)
    # get an input dataframe with sample data by looking in working directory for file DRUG1N.json
    wd = os.getcwd()
    df = sqlCtx.load("file://"+wd+"/DRUG1N.json","json").repartition(4)
    # specify predictors and target
    modelpath = "/tmp/svm.model"
    modelmetadata_path = "/tmp/svm.metadata"
    target = json.loads(open(modelmetadata_path,"r").read())["target"]
    schema = df.schema
else:
    import spss.pyspark.runtime
    ascontext = spss.pyspark.runtime.getContext()
    sc = ascontext.getSparkContext()
    sqlCtx = ascontext.getSparkSQLContext()
    df = ascontext.getSparkInputData()
    target = "%%target%%"
    schema = ascontext.getSparkInputSchema()

prediction_field = "$T-" + target
prediction_type = StringType()
output_schema = StructType(schema.fields + [StructField(prediction_field, prediction_type, nullable=True)])

if ascontext:
    if ascontext.isComputeDataModelOnly():
        ascontext.setSparkOutputSchema(output_schema)
        sys.exit(0)

if ascontext:
    s_model = ascontext.getModelContentToString("model")
    s_metadata = ascontext.getModelContentToString("model.metadata")
else:
    s_model = open(modelpath,"r").read()
    s_metadata = open(modelmetadata_path,"r").read()

df = df.toPandas()
model_metadata = json.loads(s_metadata)
import pickle
clf = pickle.loads(s_model)


df[prediction_field] = clf.predict(df[model_metadata['predictors']])

df = sqlCtx.createDataFrame(df)

if ascontext:
    ascontext.setSparkOutputData(df)
