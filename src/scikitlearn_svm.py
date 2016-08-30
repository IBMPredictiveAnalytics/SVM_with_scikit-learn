
script_details = ("scikitlearn_svm.py",0.5)

from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark import AccumulatorParam
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import DenseVector
from sklearn import preprocessing, svm
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
    fields = ["Age","K","Na"]
    target = "BP"
    modelpath = "/tmp/svm.model"
    modelmetadata_path = "/tmp/svm.metadata"
    kernel = "linear"
    tol=0.001
    coef0=0.0
    gamma = 'auto'
    degree=3
    shrinking=True
else:
    import spss.pyspark.runtime
    ascontext = spss.pyspark.runtime.getContext()
    sc = ascontext.getSparkContext()
    df = ascontext.getSparkInputData()
    fields = map(lambda x: x.strip(),"%%fields%%".split(","))
    target = '%%target%%'
    kernel = '%%kernel%%'
    tol=float('%%tol%%')
    coef0=float('%%coef0%%')
    gamma_enabled = ('%%gamma_enabled%%' == 'Y')
    if gamma_enabled:
        gamma=float('%%gamma%%')
    else:
        gamma='auto'
    degree=int('%%degree%%')
    shrinking=('%%shrinking%%'=='Y')

df = df.toPandas()
params = {"kernel":kernel,"coef0":coef0,"tol":tol,"degree":degree, "shrinking":shrinking}
if gamma != 'auto':
    params["gamma"] = gamma

clf = svm.SVC(**params)
model = clf.fit(df[fields],df[target])
print(str(model))

model_metadata = { "predictors": fields, "target":target }

import pickle
s_model = pickle.dumps(clf)
s_metadata = json.dumps(model_metadata)

if ascontext:
    ascontext.setModelContentFromString("model",s_model)
    ascontext.setModelContentFromString("model.metadata",s_metadata)
else:
    open(modelpath,"w").write(s_model)
    open(modelmetadata_path,"w").write(s_metadata)