# Script to train the gamma/hadron separation using xgboost
#
# Author: Rodrigo Guedes Lang, based on previous scripts by Laura Olivera-Nieto and Simon Steinmassl
#
# Versions succesfully tested in the HD cluster:
#   Python: 3.9.7
#   uproot: 4.1.2
#   numpy: 1.21.1
#   pandas: 1.3.3
#   xgboost: 1.4.2
#   sklearn: 0.24.2

import yaml
import sys
import os
from scripts.readTrees import readInputTrees
from scripts.plotInputDistributions import plotInputDistributions
from scripts.doTraining import train
from scripts.plotOutputInfo import plotOutputInfo
from scripts.generateLookups import makeLookups_EnergyShape, makeLookups_OffEnergyShape, makeLookups
from scripts.generateTrees import generateTrees
from scripts.generateTreesForOptimization import generateInputTreesForOptimization
from scripts.optimizeCuts import runOptimization
from scripts.plotOptimization import plotOptimization, GetOptimizedCuts
from scripts.plotIRFs import getIRFPlots
import scripts.utils as utils
import shutil

if (len(sys.argv) != 2):
    sys.exit("Run this script with: python runSeparationTraining.py <configfile>")

configFile = open(sys.argv[1],'r')
configOptions = yaml.safe_load(configFile)

# Reading and checking config file

print("Welcome! Reading input config file.")

makeEnergyShape = False
if ("MakeEnergyShape" in configOptions):
    makeEnergyShape = configOptions["MakeEnergyShape"]
makeOffEnergyShape = False
if ("MakeOffEnergyShape" in configOptions):
    makeOffEnergyShape = configOptions["MakeOffEnergyShape"]
generateInputTrees = False
if ("GenerateInputTrees" in configOptions):
    generateInputTrees = configOptions["GenerateInputTrees"]
plotInputDebug = False
if ("PlotInputDebug" in configOptions):
    plotInputDebug = configOptions["PlotInputDebug"]
doTraining = False
if ("DoTraining" in configOptions):
    doTraining = configOptions["DoTraining"]
plotOutputDebug = False
if ("PlotOutputDebug" in configOptions):
    plotOutputDebug = configOptions["PlotOutputDebug"]
generateTreesForOptimization = False
if ("GenerateTreesForOptimization" in configOptions):
    generateTreesForOptimization = configOptions["GenerateTreesForOptimization"]
plotOptimizationResults = False
if ("PlotOptimizationResults" in configOptions):
    plotOptimizationResults = configOptions["PlotOptimizationResults"]
optimizeCuts = False
if ("OptimizeCuts" in configOptions):
    optimizeCuts = configOptions["OptimizeCuts"]
finishLookupTables = False
if ("FinishLookupTables" in configOptions):
    finishLookupTables = configOptions["FinishLookupTables"]
plotIRFs = False
if ("PlotIRFs" in configOptions):
    plotIRFs = configOptions["PlotIRFs"]

if (plotOutputDebug and not doTraining):
    sys.exit("ERROR! Impossible to run output debugging without doing the training!")

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

if (not generateInputTrees and doTraining and not os.path.exists(workDirectory + "/hap/")):
    sys.exit("ERROR! No 'hap' folder located in " + workDirectory + ". Please first generate the inputs trees with the option 'GenerateInputTrees'.")
if (not generateInputTrees and doTraining and not os.listdir(workDirectory + "/hap/")):
    sys.exit("ERROR! Empty 'hap' folder in " + workDirectory + ". Please first generate the inputs trees with the option 'GenerateInputTrees'.")

if "Configs" not in configOptions:
    sys.exit("ERROR! Configs to be used not defined in the config file! Please use the option 'Configs'.")
configs = configOptions["Configs"]
if "MuonPhases" not in configOptions:
    sys.exit("ERROR! Muon phases to be used not defined in the config file! Please use the option 'MuonPhases'.")
muonPhases = configOptions["MuonPhases"]
if "ZenithAngles" not in configOptions:
    sys.exit("ERROR! Zenith angles not defined in the config file! Please use the option 'ZenithAngles'.")
zenithAngles = configOptions["ZenithAngles"]
if "AzimuthAngles" not in configOptions:
    sys.exit("ERROR! Azimuth angles not defined in the config file! Please use the option 'AzimuthAngles'.")
azimuthAngles = configOptions["AzimuthAngles"]
if "OffsetAngles" not in configOptions:
    sys.exit("ERROR! Offset angles not defined in the config file! Please use the option 'OffsetAngles'.")
offsetAngles = configOptions["OffsetAngles"]

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
maxOffrunsEventsForInputTrees=0
if "MaxOffrunsEventsForInputTrees" in configOptions:
    maxOffrunsEventsForInputTrees = configOptions["MaxOffrunsEventsForInputTrees"]
if "VariablesNameForTMVA" not in configOptions:
    sys.exit("ERROR! Name of the variables in the TMVA format not defined in the config file! Please use the option 'VariablesNameForTMVA'.")
variablesNameForTMVA = configOptions["VariablesNameForTMVA"]

if ("UseEnergyRanges" not in configOptions):
    useEnergyRanges = False
else:
    useEnergyRanges = configOptions["UseEnergyRanges"]
if (configOptions["UseEnergyRanges"] and "EnergyRanges" not in configOptions):
    sys.exit("ERROR! Energy ranges expected but not provided in the config file! Please use the option 'EnergyRanges'.")
energyRanges = configOptions["EnergyRanges"]

if ("UseSizeRanges" not in configOptions):
    useSizeRanges = False
else:
    useSizeRanges = configOptions["UseSizeRanges"]
if (configOptions["UseSizeRanges"] and "SizeRanges" not in configOptions):
    sys.exit("ERROR! Size ranges expected but not provided in the config file! Please use the option 'SizeRanges'.")
sizeRanges = configOptions["SizeRanges"]

if ("PreCuts" not in configOptions):
    preCuts = []
else:
    preCuts = configOptions["PreCuts"]

if plotInputDebug:
    if ("InputVariablesToPlot" not in configOptions):
        sys.exit("ERROR! Please define the input variables to be plotted with the option 'InputVariablesToPlot'.")
    if ("RangeToPlotInputVariables" not in configOptions or len(configOptions["InputVariablesToPlot"]) != len(configOptions["RangeToPlotInputVariables"])):
        sys.exit("ERROR! Different number of variables and ranges defined! Please check that the array in 'InputVariablesToPlot' and 'RangeToPlotInputVariables' have the same size!")
    inputVariablesToPlot = configOptions["InputVariablesToPlot"]
    rangeToPlotInputVariables = configOptions["RangeToPlotInputVariables"]

if generateTreesForOptimization or optimizeCuts or plotIRFs:
    if ("ZenithAngleForOptimization" not in configOptions):
        print("Warning! Zenith angle for optimization expected but not provided in the config file! Using standard value of 20.")
        zenithAngleForOptimization = 20
    zenithAngleForOptimization = int(configOptions["ZenithAngleForOptimization"])

if optimizeCuts or plotOptimizationResults or finishLookupTables or plotIRFs:
    if ("SpectralIndex" not in configOptions):
        sys.exit("ERROR! Spectral index used in optimization expected but not provided in the config file! Please use the option 'SpectralIndex'.")
    spectralIndex = configOptions["SpectralIndex"]
    if ("NormalizationFactor" not in configOptions):
        sys.exit("ERROR! Normalization factor used in optimization expected but not provided in the config file! Please use the option 'NormalizationFactor'.")
    normalizationFactor = configOptions["NormalizationFactor"]
    if ("MinimumSignalEfficiency" not in configOptions):
        sys.exit("ERROR! Minimum signal efficiency considered in optimization expected but not provided in the config file! Please use the option 'MinimumSignalEfficiency'.")
    minimumSignalEfficiency = configOptions["MinimumSignalEfficiency"]
    verbose = False
    if ("Verbose" in configOptions):
        verbose = configOptions["Verbose"]
if generateTreesForOptimization or generateTrees:
    maxOffrunsEvents = 0
    if ("MaxOffrunsEvents" in configOptions):
        maxOffrunsEvents = configOptions["MaxOffrunsEvents"]

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

xgboosterParameters = {"objective":"binary:logistic"}
xgboosterParameters["eval_metric"] = 'logloss' #otherwise will give this warning: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
if ("n_estimators" in configOptions):
    xgboosterParameters["n_estimators"] = configOptions["n_estimators"]
if ("max_depth" in configOptions):
    xgboosterParameters["max_depth"] = configOptions["max_depth"]
if ("learning_rate" in configOptions):
    xgboosterParameters["learning_rate"] = configOptions["learning_rate"]
if ("gamma" in configOptions):
    xgboosterParameters["gamma"] = configOptions["gamma"]
if ("reg_alpha" in configOptions):
    xgboosterParameters["reg_alpha"] = configOptions["reg_alpha"]
if ("reg_lambda" in configOptions):
    xgboosterParameters["reg_lambda"] = configOptions["reg_lambda"]
if ("scale_pos_weight" in configOptions):
    xgboosterParameters["scale_pos_weight"] = configOptions["scale_pos_weight"]

environmentVariables = {}
environmentVariables["HESSROOT"] = os.getenv('HESSROOT')
environmentVariables["HESSDATA"] = os.getenv('HESSDATA')
environmentVariables["HESSDST"] = os.getenv('HESSDST')

# Start with the initial lookups

print("\nEnvironment Variables:")
print("-> $HESSROOT = " + environmentVariables["HESSROOT"])
print("-> $HESSDATA = " + environmentVariables["HESSDATA"])
print("-> $HESSDST = " + environmentVariables["HESSDST"])
print("-> WorkDir = " + workDirectory)

if (makeEnergyShape or makeOffEnergyShape):
    print("\n-> Making energy shape lookups. This may take a while, go grab a coffee.")
    print("-> Configs:", configs)
    print("-> MuonPhases:", muonPhases)
    for config in configs:
        muonPhase = muonPhases[0] # TO BE CORRECTED
        jobs = []
        print(" -> Config: " + config)
        if (makeEnergyShape):
            newjobs = makeLookups_EnergyShape(workDirectory, config, environmentVariables)
            jobs = jobs + newjobs
            utils.wait_for_jobs_to_finish(jobs)
            jobs = []
            jobs = jobs + makeLookups('MergeEnergyShape', workDirectory, config, environmentVariables)
            print(" -> Merging energy shape lookups")
            utils.wait_for_jobs_to_finish(jobs)
            jobs = []
        if (makeOffEnergyShape):
            newjobs = makeLookups_OffEnergyShape(workDirectory, config, zenithAngles, muonPhase, environmentVariables)
            jobs = jobs + newjobs
            utils.wait_for_jobs_to_finish(jobs)
            jobs = []
            jobs = jobs + makeLookups('MergeOffShape', workDirectory, config, environmentVariables)
            print(" -> Merging off energy shape lookups")
            utils.wait_for_jobs_to_finish(jobs)
        if (makeEnergyShape):
            shutil.move(workDirectory + "/config/" + config + "/result/ScaleInfo.root",workDirectory + "/config/" + config + "/ScaleInfo.root")
            shutil.move(workDirectory + "/config/" + config + "/result/EnergyInfo.root",workDirectory + "/config/" + config + "/EnergyInfo.root")
        if (makeOffEnergyShape):
            shutil.move(workDirectory + "/config/" + config + "/result/ScaleInfoOff.root",workDirectory + "/config/" + config + "/ScaleInfoOff.root")

if (generateInputTrees):
    print("\n-> Generating input trees with hap")
    print("-> Configs:", configs)
    print("-> MuonPhases:", muonPhases)
    jobs = []
    for config in configs:
        print(" -> Config: " + config)
        for muonPhase in muonPhases:
            print("  -> Muon phase: " + muonPhase)
            newjobs = generateTrees(workDirectory, config, signal, variablesForInputTrees, muonPhase, zenithAngles, azimuthAngles, offsetAngles, environmentVariables, maxOffrunsEventsForInputTrees)
            jobs = jobs + newjobs
            newjobs = generateTrees(workDirectory, config, background, variablesForInputTrees, muonPhase, zenithAngles, azimuthAngles, offsetAngles, environmentVariables, maxOffrunsEventsForInputTrees)
            jobs = jobs + newjobs
    utils.wait_for_jobs_to_finish(jobs)
    jobs = []
    for config in configs:
        if 'hybrid' in config:
            os.system('rm ' + workDirectory + '/config/' + config + '/analysis.conf')
        for muonPhase in muonPhases:
            for zenith in zenithAngles:
                for azimuth in azimuthAngles:
                    haddScriptName = 'hadd_offruns_' + config + '_' + muonPhase + '_' + str(zenith) + 'deg_' + str(azimuth) + 'deg'
                    haddLogFile = workDirectory + '/logs/OffrunsHadd_' + config + '_' + muonPhase + '_' + str(zenith) + 'deg_' + str(azimuth) + 'deg'
                    haddCommand = 'hadd -f ' + workDirectory + '/hap/Offruns_' + config + '_' + muonPhase + '_' + str(zenith) + 'deg_' + str(azimuth) + 'deg_events.root ' + workDirectory + '/hap/Offruns_' + config + '_' + muonPhase + '_' + str(zenith) + 'deg_' + str(azimuth) + 'deg/events*.root'
                    hadd_job_id = utils.submit_job_ws(haddCommand, haddScriptName, False, haddLogFile, workDirectory + "/temp_scripts/")
                    jobs.append(hadd_job_id)
    utils.wait_for_jobs_to_finish(jobs) # this part had to be added since hap_split.pl doesn't generate _events.root as a standard. For some people this might be the case and then this step will be done twice. But better safe than sorry.


if (plotInputDebug or doTraining or plotOutputDebug):

    if ((plotInputDebug or plotOutputDebug) and not os.path.exists(workDirectory + '/plots/')):
        os.mkdir(workDirectory + '/plots/')

    if (not os.path.exists(workDirectory + '/temp_scripts/')):
        os.mkdir(workDirectory + '/temp_scripts/')


    print("\nCombination of input variables to plot/train:")

    print("-> Configs:",configs)
    print("-> Muon phases:",muonPhases)
    print("-> Zenith angles:", zenithAngles)
    print("-> Azimuth angles:", azimuthAngles)
    print("-> Offset angles:", offsetAngles)
    if (useEnergyRanges):
        print("-> Energy ranges:", energyRanges)
    if (useSizeRanges):
        print("-> Size ranges:", sizeRanges)

    jobs = []

    for config in configs:
        print(" -> Config: " + config)
        for muonPhase in muonPhases:
            print(" -> Muon phases: " + muonPhase)
            if ("hybrid" in config) or ("mono" in config):
                subFolderName = "v4.2"
            else:
                subFolderName = "v3.6.1"
            os.system("rm " + workDirectory + "/config/" + config + "/" + muonPhase + "/" + subFolderName + "/weights/*")
            for zenith in zenithAngles:
                for azimuth in azimuthAngles:
                    for offset in offsetAngles:
                        if (not useEnergyRanges and not useSizeRanges):
                            energyRange = [0,0]
                            sizeRange = [0,0] # ignore ranges
                            print("\nZenith: " + str(zenith) + "deg - Azimuth: " + str(azimuth) + "deg - Offset: " + str(offset) + "deg")
                            print(" -> Submitting job")

                            f = open(workDirectory + '/temp_scripts/Training_' + config + '_' + muonPhase + '_' + str(zenith) + 'deg_' + str(azimuth) + 'deg_' + str(offset) + 'deg.sh','w')
                            f.write("#!/bin/bash")
                            f.write("\n")
                            f.write('python scripts/submitTraining.py ' + sys.argv[1] + ' ' + config + ' ' + muonPhase + ' ' + str(zenith) + ' ' + str(azimuth) + ' ' + str(offset) + ' ' + str(energyRange[0]) + ' ' + str(energyRange[1]) + ' ' + str(sizeRange[0]) + ' ' + str(sizeRange[1]))
                            f.close()
                            command = workDirectory + '/temp_scripts/Training_' + config + '_' + muonPhase + '_' + str(zenith) + 'deg_' + str(azimuth) + 'deg_' + str(offset) + 'deg.sh'
                            logFile = workDirectory + '/logs/Training_' + config + '_' + muonPhase + '_' + str(zenith) + 'deg_' + str(azimuth) + 'deg_' + str(offset) + 'deg.log'
                            jobs.append(utils.submit_job_python(command, logFile))

                        if (useEnergyRanges):
                            sizeRange = [0,0]
                            for energyRange in energyRanges:
                                print("\nZenith: " + str(zenith) + "deg - Azimuth: " + str(azimuth) + "deg - Offset: " + str(offset) + "deg - Energy range: [" + str(energyRange[0]) + "," + str(energyRange[1]) + "] TeV")
                                print(" -> Submitting job")

                                f = open(workDirectory + '/temp_scripts/Training_' + config + '_' + muonPhase + '_' + str(zenith) + 'deg_' + str(azimuth) + 'deg_' + str(offset) + 'deg_' + str(energyRange[0]) + 'to' + str(energyRange[1]) + 'TeV.sh','w')
                                f.write("#!/bin/bash")
                                f.write("\n")
                                f.write('python scripts/submitTraining.py ' + sys.argv[1] + ' ' + config + ' ' + muonPhase + ' ' + str(zenith) + ' ' + str(azimuth) + ' ' + str(offset) + ' ' + str(energyRange[0]) + ' ' + str(energyRange[1]) + ' ' + str(sizeRange[0]) + ' ' + str(sizeRange[1]))
                                f.close()
                                command = workDirectory + '/temp_scripts/Training_' + config + '_' + muonPhase + '_' + str(zenith) + 'deg_' + str(azimuth) + 'deg_' + str(offset) + 'deg_' + str(energyRange[0]) + 'to' + str(energyRange[1]) + 'TeV.sh'
                                logFile = workDirectory + '/logs/Training_' + config + '_' + muonPhase + '_' + str(zenith) + 'deg_' + str(azimuth) + 'deg_' + str(offset) + 'deg_' + str(energyRange[0]) + 'to' + str(energyRange[1]) + 'TeV.log'
                                jobs.append(utils.submit_job_python(command, logFile))

                        if (useSizeRanges):
                            energyRange = [0,0]
                            for sizeRange in sizeRanges:
                                print("\nZenith: " + str(zenith) + "deg - Azimuth: " + str(azimuth) + "deg - Offset: " + str(offset) + "deg - Size range: [" + str(sizeRange[0]) + "," + str(sizeRange[1]) + "] p.e.")
                                print(" -> Submitting job")

                                f = open(workDirectory + '/temp_scripts/Training_' + config + '_' + muonPhase + '_' + str(zenith) + 'deg_' + str(azimuth) + 'deg_' + str(offset) + 'deg_' + str(sizeRange[0]) + 'to' + str(sizeRange[1]) + 'pe.sh','w')
                                f.write("#!/bin/bash")
                                f.write("\n")
                                f.write('python scripts/submitTraining.py ' + sys.argv[1] + ' ' + config + ' ' + muonPhase + ' ' + str(zenith) + ' ' + str(azimuth) + ' ' + str(offset) + ' ' + str(energyRange[0]) + ' ' + str(energyRange[1]) + ' ' + str(sizeRange[0]) + ' ' + str(sizeRange[1]))
                                f.close()
                                command = workDirectory + '/temp_scripts/Training_' + config + '_' + muonPhase + '_' + str(zenith) + 'deg_' + str(azimuth) + 'deg_' + str(offset) + 'deg_' + str(sizeRange[0]) + 'to' + str(sizeRange[1]) + 'pe.sh'
                                logFile = workDirectory + '/logs/Training_' + config + '_' + muonPhase + '_' + str(zenith) + 'deg_' + str(azimuth) + 'deg_' + str(offset) + 'deg_' + str(sizeRange[0]) + 'to' + str(sizeRange[1]) + 'pe.log'
                                jobs.append(utils.submit_job_python(command, logFile))

    utils.wait_for_jobs_to_finish(jobs)

    if (doTraining):

        for config in configs:
            for muonPhase in muonPhases:
                if ('hybrid' in config) or ('mono' in config):
                    subFolderName = 'v4.2'
                    weightFileType = 'xml'
                    scaledEfficiencyROOTFile = workDirectory + "/config/" + config + "/" + muonPhase + "/v4.2/weights/ScaledEff2_" + config + "_0.5degoffset.root" # apparently HAP can just deal with 0.5 deg?
                else: # can't export the files to txt as hap expect the old versions. Going around needed.
                    subFolderName = 'v4.2'
                    weightFileType = 'xml'
                    scaledEfficiencyROOTFile = workDirectory + "/config/" + config + "/" + muonPhase + "/v3.6.1/weights/ScaledEff2_" + config + ".root" # hard coded in HAP
                if ('hybrid' in config) or ('stereo' in config):                
                    for zenith in zenithAngles:
                        os.system("scripts/checkMissingWeights.sh " + workDirectory + "/config/" + config + "/" + muonPhase + "/" + subFolderName + "/weights/ " + config + " " + str(zenith) + " " + weightFileType)
                if os.path.exists(scaledEfficiencyROOTFile):
                    os.system("rm " + scaledEfficiencyROOTFile)
                if (not useSizeRanges and not useEnergyRanges):
                    fileToBeJoined = workDirectory + "/config/" + config + "/" + muonPhase + "/" + subFolderName + "/weights/ScaledEff2_" + config + "*zenith*offset.root"
                    os.system("hadd " + scaledEfficiencyROOTFile + " " + fileToBeJoined)
                if (useEnergyRanges):
                    fileToBeJoined = workDirectory + "/config/" + config + "/" + muonPhase + "/" + subFolderName + "/weights/ScaledEff2_" + config + "*zenith*TeV.root"
                    os.system("hadd " + scaledEfficiencyROOTFile + " " + fileToBeJoined)
                if (useSizeRanges):
                    fileToBeJoined = workDirectory + "/config/" + config + "/" + muonPhase + "/" + subFolderName + "/weights/ScaledEff2_" + config + "*zenith*pe.root"
                    os.system("hadd " + scaledEfficiencyROOTFile + " " + fileToBeJoined)
                os.system("rm " + fileToBeJoined)


    print("\n-> Finished the training for all the combinations of input variables.")

if generateTreesForOptimization:
    print("\n-> Generating the input trees for the optimization process")
    jobs = []
    for config in configs:
        print(" -> Config: " + config)
        for muonPhase in muonPhases:
            print(" -> MuonPhase: " + muonPhase)
            newjobs = generateInputTreesForOptimization(workDirectory, config, muonPhase, zenithAngleForOptimization, 0, 0.5, environmentVariables, maxOffrunsEvents)
            jobs = jobs + newjobs
    utils.wait_for_jobs_to_finish(jobs)

if optimizeCuts:
    print("\n-> Optimizing cuts")
    for config in configs:
        print(" -> Config: " + config)
        for muonPhase in muonPhases:
            print(" -> MuonPhase: " + muonPhase)
            runOptimization(workDirectory, config, muonPhase, zenithAngleForOptimization, 0, 0.5, spectralIndex, normalizationFactor, minimumSignalEfficiency, verbose, environmentVariables)
            if 'mono' not in config:
                if not os.path.exists(workDirectory + '/config/' + config + '/analysis_prelookups.conf'):
                    sys.exit("ERROR! You have not defined the basic configuration file: analysis_prelookups.conf! Please do so before running!")

                os.system('cp ' + workDirectory + '/config/' + config + '/analysis_prelookups.conf ' + workDirectory + '/config/' + config + '/analysis.conf')

                if 'zeta' in config:

                    os.system('echo " " >> ' + workDirectory + '/config/' + config + '/analysis.conf')
                    os.system('echo "[Preselect]" >> ' + workDirectory + '/config/' + config + '/analysis.conf')
                    os.system('echo "  HillasReco::ScaledParameters.LookupName = ScaleInfoOff.root" >> ' + workDirectory + '/config/' + config + '/analysis.conf')
                    os.system('echo " " >> ' + workDirectory + '/config/' + config + '/analysis.conf')
                    os.system('echo "[TMVA]" >> ' + workDirectory + '/config/' + config + '/analysis.conf')
                    os.system('echo "  WorkDir = ' + workDirectory + '/config/' + config + '/' + muonPhase + '" >> ' + workDirectory + '/config/' + config + '/analysis.conf')

                thetaSqr, zetaBDT = GetOptimizedCuts(workDirectory, config, muonPhase, normalizationFactor, spectralIndex, minimumSignalEfficiency)

                os.system('echo "[Postselect]" >> ' + workDirectory + '/config/' + config + '/analysis.conf')
                os.system('echo "  HillasReco::TMVAParameters.ChainShower.ZetaBDT = (0.,' + str(zetaBDT) + ')" >> ' + workDirectory + '/config/' + config + '/analysis.conf')
                os.system('echo "" >> ' + workDirectory + '/config/' + config + '/analysis.conf')
                os.system('echo "[Background]" >> ' + workDirectory + '/config/' + config + '/analysis.conf')
                os.system('echo "  ThetaSqr = ' + str(thetaSqr) + '" >> ' + workDirectory + '/config/' + config + '/analysis.conf')


if plotOptimizationResults:
    print("\n-> Printing optimization results")
    for config in configs:
        print(" -> Config: " + config)
        for muonPhase in muonPhases:
            print(" -> MuonPhase: " + muonPhase)
            plotOptimization(workDirectory, config, muonPhase, normalizationFactor, spectralIndex, minimumSignalEfficiency)

if finishLookupTables:
    print("\n-> Generating the final lookup tables")
    for config in configs:
        print(" -> Config: " + config)
        for muonPhase in muonPhases:
            print(" -> MuonPhase: " + muonPhase)
            thetaSqr, zetaBDT = GetOptimizedCuts(workDirectory, config, muonPhase, normalizationFactor, spectralIndex, minimumSignalEfficiency)
            print("  -> Using thetasqr = " + str(thetaSqr))
            jobs = []
            print("  -> Effective Area")
            jobs = jobs + makeLookups('EffectiveArea', workDirectory, config, environmentVariables, muonPhase, thetaSqr, zetaBDT)
            utils.wait_for_jobs_to_finish(jobs)
            jobs = []
            jobs = jobs + makeLookups('MergeEffectiveArea', workDirectory, config, environmentVariables, muonPhase, thetaSqr, zetaBDT)
            utils.wait_for_jobs_to_finish(jobs)
            shutil.move(workDirectory + "/config/" + config + "/result/EffectiveArea.root",workDirectory + "/config/" + config + "/EffectiveArea.root")
            shutil.move(workDirectory + "/config/" + config + "/result/EnergyResolution.root",workDirectory + "/config/" + config + "/EnergyResolution.root")
            shutil.move(workDirectory + "/config/" + config + "/result/EnergyResolution2.root",workDirectory + "/config/" + config + "/EnergyResolution2.root")
            shutil.move(workDirectory + "/config/" + config + "/result/PSF.root",workDirectory + "/config/" + config + "/PSF.root")
            #jobs = []
            #print("  -> Radial Acceptance")
            #jobs = jobs + makeLookups('RadAcc', workDirectory, config, environmentVariables, thetaSqr, zetaBDT)
            #utils.wait_for_jobs_to_finish(jobs)
            #jobs = []
            #jobs = jobs + makeLookups('MergeRadAcc', workDirectory, config, environmentVariables, thetaSqr, zetaBDT)
            #utils.wait_for_jobs_to_finish(jobs)
            #shutil.move(workDirectory + "/config/" + config + "/result/RadialAcceptance.root",workDirectory + "/config/" + config + "/RadialAcceptance.root")
            #jobs = []
            #print("  -> Radial Acceptance - off")
            #jobs = jobs + makeLookups('RadAccOff', workDirectory, config, environmentVariables, thetaSqr, zetaBDT)
            #utils.wait_for_jobs_to_finish(jobs)
            #jobs = []
            #jobs = jobs + makeLookups('MergeRadAccOff', workDirectory, config, environmentVariables, thetaSqr, zetaBDT)
            #utils.wait_for_jobs_to_finish(jobs)
            #shutil.move(workDirectory + "/config/" + config + "/result/RadialAcceptanceOff.root",workDirectory + "/config/" + config + "/RadialAcceptanceOff.root")

if plotIRFs:
    print("-> Plotting IRFs")
    for config in configs:
        print(" -> Config: " + config)
        for muonPhase in muonPhases:
            print(" -> MuonPhase: " + muonPhase)
            thetaSqr, zetaBDT = GetOptimizedCuts(workDirectory, config, muonPhase, normalizationFactor, spectralIndex, minimumSignalEfficiency)
            print(" -> zetaBDT = " + str(zetaBDT))
            print(" -> thetaSqr = " + str(thetaSqr))
            outputName = workDirectory + "/plots/IRFs-" + config + "_" + str(zenithAngleForOptimization) + "deg"
            getIRFPlots(workDirectory, config, muonPhase, zenithAngleForOptimization, 0, 0.5, thetaSqr, zetaBDT, outputName)

print("\nMy job here is finished. Thank you!")
