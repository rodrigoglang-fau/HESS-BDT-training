import os
import xgboost
from importlib import reload
import xgboost2tmva as xgboost2tmva
import uproot
import numpy as np

xgboost2tmva = reload(xgboost2tmva)

def convertWeightsToTMVA(model, variablesNameForTMVA, outputName, scaledEffFilename, histogramTitle, splitData):

    X, X_test, Y, Y_test = splitData

    featureNames = []
    for i in range(len(variablesNameForTMVA)):
        featureNames.append(variablesNameForTMVA[i][0])

    model.get_booster().feature_names = featureNames
    xgboost2tmva.convert_model(model.get_booster().get_dump(),variablesNameForTMVA,outputName)

    # For some reason, when HAP uses the xml weights, it returns value between -1 and 1, instead of 0 and 1. For that reason, I need to expand the zetaBDT profile for -1 to 1 here.

    superbins = np.linspace(-1,1,10000)
    values, bins = np.histogram((model.predict(X[np.where(Y == 1)])-0.5)*2, bins=superbins, density=False)
    eff = np.flip(np.cumsum(np.flip(values))/values.sum())

    scaledFile = uproot.recreate(scaledEffFilename)
    scaledFile[histogramTitle] = eff, bins
    scaledFile.close()

    return featureNames
