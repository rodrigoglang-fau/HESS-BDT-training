import yaml
import sys
import os
from readTrees import readInputTrees
from plotInputDistributions import plotInputDistributions
from doTraining import train
from plotOutputInfo import plotOutputInfo
from convertWeights import convertWeightsToTMVA
import utils as utils

energyRange = []
sizeRange = []

config = sys.argv[2]
muonPhase = sys.argv[3]
zenith = int(sys.argv[4])
azimuth = int(sys.argv[5])
offset = float(sys.argv[6])
energyRange.append(float(sys.argv[7]))
energyRange.append(float(sys.argv[8]))
sizeRange.append(float(sys.argv[9]))
sizeRange.append(float(sys.argv[10]))

configFile = open(sys.argv[1],'r')
configOptions = yaml.safe_load(configFile)

plotInputDebug = False
if ("PlotInputDebug" in configOptions):
    plotInputDebug = configOptions["PlotInputDebug"]
doTraining = False
if ("DoTraining" in configOptions):
    doTraining = configOptions["DoTraining"]
plotOutputDebug = False
if ("PlotOutputDebug" in configOptions):
    plotOutputDebug = configOptions["PlotOutputDebug"]

if "MainDirectory" not in configOptions:
    sys.exit("ERROR! Main directory not defined in the config file! Please use the option 'MainDirectory'.")
mainDirectory = configOptions["MainDirectory"]
if (not os.path.exists(mainDirectory)):
    sys.exit("ERROR! Main directory does not exist!")
if "TrainingName" not in configOptions:
    sys.exit("ERROR! Training name not defined in the config file! Please use the option 'TrainingName'.")
trainingName = configOptions["TrainingName"]
workDirectory = mainDirectory + "/" + trainingName
if (not os.path.exists(workDirectory)):
    os.mkdir(workDirectory)

if "Signal" not in configOptions:
    sys.exit("ERROR! Signal data not defined in the config file! Please use the option 'Signal'.")
signal = configOptions["Signal"]
if (signal != "Gamma" and signal != "Gamma-diffuse"):
    sys.exit("ERROR! The only options for the signal data are: 'Gamma' and 'Gamma-diffuse'.")
if "Background" not in configOptions:
    sys.exit("ERROR! Background data not defined in the config file! Please use the option 'Background'.")
background = configOptions["Background"]
if (background != "Offruns" and background != "Proton" and background != "Selmuon"):
    sys.exit("ERROR! The only options for the background data are: 'Offruns', 'Proton' and 'Selmuon'.")
if "VariablesForInputTrees" not in configOptions and generateInputTrees:
    sys.exit("ERROR! Variables for input trees not defined in the config file! Please use the option 'VariablesForInputTrees'.")
variablesForInputTrees = configOptions["VariablesForInputTrees"]
if "VariablesNameForTMVA" not in configOptions:
    sys.exit("ERROR! Name of the variables in the TMVA format not defined in the config file! Please use the option 'VariablesNameForTMVA'.")
variablesNameForTMVA = configOptions["VariablesNameForTMVA"]

if plotInputDebug:
    if ("InputVariablesToPlot" not in configOptions):
        sys.exit("ERROR! Please define the input variables to be plotted with the option 'InputVariablesToPlot'.")
    if ("RangeToPlotInputVariables" not in configOptions or len(configOptions["InputVariablesToPlot"]) != len(configOptions["RangeToPlotInputVariables"])):
        sys.exit("ERROR! Different number of variables and ranges defined! Please check that the array in 'InputVariablesToPlot' and 'RangeToPlotInputVariables' have the same size!")
    inputVariablesToPlot = configOptions["InputVariablesToPlot"]
    rangeToPlotInputVariables = configOptions["RangeToPlotInputVariables"]

if ("MaxEvents" not in configOptions):
    sys.exit("ERROR! Maximum number of events expected but not provided in the config file! Please use the option 'MaxEvents'.")
maxEvents = int(configOptions["MaxEvents"])
if ("BackgroundToSignalFraction" not in configOptions):
    sys.exit("ERROR! Fraction of background to signal events to be considered expected but not provided in the config file! Please use the option 'BackgroundToSignalFraction'.")
optimizationArray_backgroundToSignalFraction = configOptions["BackgroundToSignalFraction"]
if ("Classifier" not in configOptions):
    sys.exit("ERROR! Classifier expected but not provided in the config file! Please use the option 'Classifier'.")
else:
    if (configOptions["Classifier"] != "XGBRegressor" and configOptions["Classifier"] != "scikit"):
        sys.exit("ERROR! Unkwown classifier! The possible options are: 'XGBRegressor' and 'scikit'.")
    classifier = configOptions["Classifier"]
if ("TestToTrainFraction" not in configOptions):
    sys.exit("ERROR! Fraction of test/train events expected but not provided in the config file! Please use the option 'TestToTrainFraction'.")
optimizationArray_testToTrainFraction = configOptions["TestToTrainFraction"]

if ("PreCuts" not in configOptions):
    preCuts = []
else:
    preCuts = configOptions["PreCuts"]

xgboosterParameters = {"objective":"binary:logistic"}
xgboosterParameters["eval_metric"] = 'logloss' #otherwise will give this warning: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
if ("n_estimators" in configOptions):
    optimizationArray_n_estimators = configOptions["n_estimators"]
else:
    optimizationArray_n_estimators = [100]
if ("max_depth" in configOptions):
    optimizationArray_max_depth = configOptions["max_depth"]
else:
    optimizationArray_max_depth = [6]
if ("learning_rate" in configOptions):
    optimizationArray_learning_rate = configOptions["learning_rate"]
else:
    optimizationArray_learning_rate = [0.3]
if ("gamma" in configOptions):
    optimizationArray_gamma = configOptions["gamma"]
else:
    optimizationArray_gamma = [0]
if ("reg_alpha" in configOptions):
    optimizationArray_reg_alpha = configOptions["reg_alpha"]
else:
    optimizationArray_reg_alpha = [0]
if ("reg_lambda" in configOptions):
    optimizationArray_reg_lambda = configOptions["reg_lambda"]
else:
    optimizationArray_reg_lambda = [0]
if ("scale_pos_weight" in configOptions):
    optimizationArray_scale_pos_weight = configOptions["scale_pos_weight"]
else:
    optimizationArray_scale_pos_weight = [1]
if ("min_child_weight" in configOptions):
    optimizationArray_min_child_weight = configOptions["min_child_weight"]
else:
    optimizationArray_min_child_weight = [1]
if ("max_delta_step" in configOptions):
    optimizationArray_max_delta_step = configOptions["max_delta_step"]
else:
    optimizationArray_max_delta_step = [0]
if ("subsample" in configOptions):
    optimizationArray_subsample = configOptions["subsample"]
else:
    optimizationArray_subsample = [1]

if (sizeRange[0] == 0 and sizeRange[1] == 0):
    hasSizeRange = False
else:
    hasSizeRange = True
if (energyRange[0] == 0 and energyRange[1] == 0):
    hasEnergyRange = False
else:
    hasEnergyRange = True

print("\nZenith: " + str(zenith) + "deg - Azimuth: " + str(azimuth) + "deg - Offset: " + str(offset) + "deg")
print(" -> Reading input files")
signalInput, backgroundInput = readInputTrees(workDirectory, signal, background, zenith, azimuth, offset, config, muonPhase, variablesNameForTMVA, preCuts, energyRange, sizeRange, 100, maxEvents)
if (len(signalInput[0]) > 0 and len(backgroundInput[0]) > 0):
    if (plotInputDebug):
        print(" -> Plotting input distributions")
        if (not hasEnergyRange and not hasSizeRange):
            outputName = workDirectory + "/plots/InputDistribution-" + config + "-" + muonPhase + "-" + str(zenith) + "deg-" + str(azimuth) + "deg-" + str(offset) + "deg"
            plotTitle = config + " | " + signal + " vs " + background + "\nZenith = " + str(zenith) + "deg | Azimuth = " + str(azimuth) + "deg | Offset = " + str(offset) + "deg"
        elif (hasEnergyRange):
            outputName = workDirectory + "/plots/InputDistribution-" + config + "-" + muonPhase + "-" + str(zenith) + "deg-" + str(azimuth) + "deg-" + str(offset) + "deg-" + str(energyRange[0]) + 'to' + str(energyRange[1]) + 'TeV'
            plotTitle = config + " | " + signal + " vs " + background + "\nZenith = " + str(zenith) + "deg | Azimuth = " + str(azimuth) + "deg | Offset = " + str(offset) + "deg | E = [" + str(energyRange[0]) + ',' + str(energyRange[1]) + '] TeV'
        elif (hasSizeRange):
            outputName = workDirectory + "/plots/InputDistribution-" + config + "-" + muonPhase + "-" + str(zenith) + "deg-" + str(azimuth) + "deg-" + str(offset) + "deg-" + str(sizeRange[0]) + 'to' + str(sizeRange[1]) + 'pe'
            plotTitle = config + " | " + signal + " vs " + background + "\nZenith = " + str(zenith) + "deg | Azimuth = " + str(azimuth) + "deg | Offset = " + str(offset) + "deg | Size = [" + str(sizeRange[0]) + ',' + str(sizeRange[1]) + '] pe'
        plotInputDistributions(inputVariablesToPlot, rangeToPlotInputVariables, variablesNameForTMVA, signalInput, backgroundInput, outputName,plotTitle)
    if (doTraining):
        print(" -> Optimizing the training")
        bestScore = 0.
        for backgroundToSignalFraction in optimizationArray_backgroundToSignalFraction:
            signalInput, backgroundInput = readInputTrees(workDirectory, signal, background, zenith, azimuth, offset, config, muonPhase, variablesNameForTMVA, preCuts, energyRange, sizeRange, backgroundToSignalFraction, maxEvents)
            for testToTrainFraction in optimizationArray_testToTrainFraction:
                for n_estimators in optimizationArray_n_estimators:
                     for max_depth in optimizationArray_max_depth:
                         for learning_rate in optimizationArray_learning_rate:
                             for gamma in optimizationArray_gamma:
                                 for reg_alpha in optimizationArray_reg_alpha:
                                     for reg_lambda in optimizationArray_reg_lambda:
                                         for scale_pos_weight in optimizationArray_scale_pos_weight:
                                             for min_child_weight in optimizationArray_min_child_weight:
                                                 for max_delta_step in optimizationArray_max_delta_step:
                                                     for subsample in optimizationArray_subsample:
                                                         xgboosterParameters["n_estimators"] = n_estimators
                                                         xgboosterParameters["max_depth"] = max_depth
                                                         xgboosterParameters["learning_rate"] = learning_rate
                                                         xgboosterParameters["gamma"] = gamma
                                                         xgboosterParameters["reg_alpha"] = reg_alpha
                                                         xgboosterParameters["scale_pos_weight"] = scale_pos_weight
                                                         xgboosterParameters["min_child_weight"] = min_child_weight
                                                         xgboosterParameters["max_delta_step"] = max_delta_step
                                                         xgboosterParameters["subsample"] = subsample
                                                         model, splitData, featureImportances = train(signalInput, backgroundInput, classifier, testToTrainFraction, xgboosterParameters)
                                                         X, X_test, Y, Y_test = splitData
                                                         currentModelScore = model.score(X_test,Y_test)
                                                         if (currentModelScore > bestScore):
                                                             bestScore = currentModelScore
                                                             best_backgroundToSignalFraction = backgroundToSignalFraction
                                                             best_testToTrainFraction = testToTrainFraction
                                                             best_n_estimators = n_estimators
                                                             best_max_depth = max_depth
                                                             best_learning_rate = learning_rate
                                                             best_gamma = gamma
                                                             best_reg_alpha = reg_alpha
                                                             best_scale_pos_weight = scale_pos_weight
                                                             best_min_child_weight = min_child_weight
                                                             best_max_delta_step = max_delta_step
                                                             best_subsample = subsample
        backgroundToSignalFraction = best_backgroundToSignalFraction
        signalInput, backgroundInput = readInputTrees(workDirectory, signal, background, zenith, azimuth, offset, config, muonPhase, variablesNameForTMVA, preCuts, energyRange, sizeRange, backgroundToSignalFraction, maxEvents)
        testToTrainFraction = best_testToTrainFraction
        xgboosterParameters["n_estimators"] = best_n_estimators
        xgboosterParameters["max_depth"] = best_max_depth
        xgboosterParameters["learning_rate"] = best_learning_rate
        xgboosterParameters["gamma"] = best_gamma
        xgboosterParameters["reg_alpha"] = best_reg_alpha
        xgboosterParameters["scale_pos_weight"] = best_scale_pos_weight
        xgboosterParameters["min_child_weight"] = best_min_child_weight
        xgboosterParameters["max_delta_step"] = best_max_delta_step
        xgboosterParameters["subsample"] = best_subsample
        print(" -> Best parameters found at: (" + str(backgroundToSignalFraction) + ", " + str(testToTrainFraction) + ", " + str(xgboosterParameters["n_estimators"]) + ", " + str(xgboosterParameters["max_depth"]) + ", " + str(xgboosterParameters["learning_rate"]) + ", " + str(xgboosterParameters["gamma"]) + ", " + str(xgboosterParameters["reg_alpha"]) + ", " + str(xgboosterParameters["scale_pos_weight"]) + ", " + str(xgboosterParameters["min_child_weight"]) + ", " + str(xgboosterParameters["max_delta_step"]) + ", " + str(xgboosterParameters["subsample"]) + ")")
        model, splitData, featureImportances = train(signalInput, backgroundInput, classifier, testToTrainFraction, xgboosterParameters)
        print(" -> Converting weigths to TMVA")
        if ('hybrid' in config) or ('mono' in config):
            if not os.path.exists(workDirectory + "/config/" + config + "/" + muonPhase + "/v4.2/weights/"):
                os.makedirs(workDirectory + "/config/" + config + "/" + muonPhase + "/v4.2/weights/")
        else:
            if not os.path.exists(workDirectory + "/config/" + config + "/" + muonPhase + "/v3.6.1/weights/"):
                os.makedirs(workDirectory + "/config/" + config + "/" + muonPhase + "/v3.6.1/weights/")
        if ('hybrid' in config) or ('mono' in config):
            if (not hasEnergyRange and not hasSizeRange):
                outputName = workDirectory + "/config/" + config + "/" + muonPhase + "/v4.2/weights/" + config + '_' + str(zenith) + 'degzenith_' + str(offset) + 'degoffset_BDT.weights.xml'
                scaledEffFileName = workDirectory + "/config/" + config + "/" + muonPhase + "/v4.2/weights/ScaledEff2_" + config + '_' + str(zenith) + 'degzenith_' + str(offset) + 'degoffset.root'
                histogramTitle = str(zenith) + 'deg_zenith_' + str(offset) + 'deg_offset'
            if (hasEnergyRange):
                outputName = workDirectory + "/config/" + config + "/" + muonPhase + "/v4.2/weights/" + config + '_' + str(zenith) + 'degzenith_' + str(offset) + 'degoffset_' + str(energyRange[0]) + 'to' + str(energyRange[1]) + 'TeV_BDT.weights.xml'
                scaledEffFileName = workDirectory + "/config/" + config + "/" + muonPhase + "/v4.2/weights/ScaledEff2_" + config + '_' + str(zenith) + 'degzenith_' + str(offset) + 'degoffset_' + str(energyRange[0]) + 'to' + str(energyRange[1]) + 'TeV.root'
                histogramTitle = str(zenith) + 'deg_zenith_' + str(offset) + 'deg_offset_' + str(energyRange[0]) + 'to' + str(energyRange[1]) + 'TeV'
            if (hasSizeRange):
                outputName = workDirectory + "/config/" + config + "/" + muonPhase + "/v4.2/weights/" + config + '_' + str(zenith) + 'degzenith_' + str(offset) + 'degoffset_' + str(sizeRange[0]) + 'to' + str(sizeRange[1]) + 'pe_BDT.weights.xml'
                scaledEffFileName = workDirectory + "/config/" + config + "/" + muonPhase + "/v4.2/weights/ScaledEff2_" + config + '_' + str(zenith) + 'degzenith_' + str(offset) + 'degoffset_' + str(sizeRange[0]) + 'to' + str(sizeRange[1]) + 'pe.root'
                histogramTitle = str(zenith) + 'deg_zenith_' + str(offset) + 'deg_offset_' + str(sizeRange[0]) + 'to' + str(sizeRange[1]) + 'pe'
        else:  # HAP expect .txt files for stereo
            if (not hasEnergyRange and not hasSizeRange):
                outputName = workDirectory + "/config/" + config + "/" + muonPhase + "/v3.6.1/weights/" + config + '_' + str(zenith) + 'degzenith_' + str(offset) + 'degoffset_BDT.weights.txt'
                scaledEffFileName = workDirectory + "/config/" + config + "/" + muonPhase + "/v3.6.1/weights/ScaledEff2_" + config + '_' + str(zenith) + 'degzenith_' + str(offset) + 'degoffset.root'
                histogramTitle = str(zenith) + 'deg_zenith_' + str(offset) + 'deg_offset'
            if (hasEnergyRange):
                outputName = workDirectory + "/config/" + config + "/" + muonPhase + "/v3.6.1/weights/" + config + '_' + str(zenith) + 'degzenith_' + str(offset) + 'degoffset_' + str(energyRange[0]) + 'to' + str(energyRange[1]) + 'TeV_BDT.weights.txt'
                scaledEffFileName = workDirectory + "/config/" + config + "/" + muonPhase + "/v3.6.1/weights/ScaledEff2_" + config + '_' + str(zenith) + 'degzenith_' + str(offset) + 'degoffset_' + str(energyRange[0]) + 'to' + str(energyRange[1]) + 'TeV.root'
                histogramTitle = str(zenith) + 'deg_zenith_' + str(offset) + 'deg_offset_' + str(energyRange[0]) + 'to' + str(energyRange[1]) + 'TeV'
            if (hasSizeRange):
                outputName = workDirectory + "/config/" + config + "/" + muonPhase + "/v3.6.1/weights/" + config + '_' + str(zenith) + 'degzenith_' + str(offset) + 'degoffset_' + str(sizeRange[0]) + 'to' + str(sizeRange[1]) + 'pe_BDT.weights.txt'
                scaledEffFileName = workDirectory + "/config/" + config + "/" + muonPhase + "/v3.6.1/weights/ScaledEff2_" + config + '_' + str(zenith) + 'degzenith_' + str(offset) + 'degoffset_' + str(sizeRange[0]) + 'to' + str(sizeRange[1]) + 'pe.root'
                histogramTitle = str(zenith) + 'deg_zenith_' + str(offset) + 'deg_offset_' + str(sizeRange[0]) + 'to' + str(sizeRange[1]) + 'pe'
        featureNames = convertWeightsToTMVA(model, variablesNameForTMVA, outputName, scaledEffFileName,histogramTitle, splitData)
        if (plotOutputDebug):
            print(" -> Plotting output information")
            if (not hasEnergyRange and not hasSizeRange):
                plotTitle = config + " | " + signal + " vs " + background + "\nZenith = " + str(zenith) + "deg | Azimuth = " + str(azimuth) + "deg | Offset = " + str(offset) + "deg"
                outputName = workDirectory + "/plots/OutputPlots-" + config + "-" + muonPhase + "-" + str(zenith) + "deg-" + str(azimuth) + "deg-" + str(offset) + "deg"
            elif (hasEnergyRange):
                plotTitle = config + " | " + signal + " vs " + background + "\nZenith = " + str(zenith) + "deg | Azimuth = " + str(azimuth) + "deg | Offset = " + str(offset) + "deg | E = [" + str(energyRange[0]) + ',' + str(energyRange[1]) + '] TeV'
                outputName = workDirectory + "/plots/OutputPlots-" + config + "-" + muonPhase + "-" + str(zenith) + "deg-" + str(azimuth) + "deg-" + str(offset) + "deg-" + str(energyRange[0]) + 'to' + str(energyRange[1]) + 'TeV'
            elif (hasSizeRange):
                plotTitle = config + " | " + signal + " vs " + background + "\nZenith = " + str(zenith) + "deg | Azimuth = " + str(azimuth) + "deg | Offset = " + str(offset) + "deg | Size = [" + str(sizeRange[0]) + ',' + str(sizeRange[1]) + '] p.e.'
                outputName = workDirectory + "/plots/OutputPlots-" + config + "-" + muonPhase + "-" + str(zenith) + "deg-" + str(azimuth) + "deg-" + str(offset) + "deg-" + str(sizeRange[0]) + 'to' + str(sizeRange[1]) + 'pe'
            plotOutputInfo(model, splitData, featureNames, outputName, plotTitle, featureImportances)
    print(" -> Done")
