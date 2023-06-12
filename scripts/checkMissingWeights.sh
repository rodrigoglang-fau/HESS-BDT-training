#!/bin/bash

dir=$1
config=$2
zen=$3
ftype=$4

if test ! -f ${dir}/${config}_${zen}degzenith_0.5degoffset_2.0to5.0TeV_BDT.weights.$ftype; then
  echo "Missing file ${dir}/${config}_${zen}degzenith_0.5degoffset_2.0to5.0TeV_BDT.weights.$ftype, copying from upper energy band"
  cp ${dir}/${config}_${zen}degzenith_0.5degoffset_5.0to100.0TeV_BDT.weights.$ftype ${dir}/${config}_${zen}degzenith_0.5degoffset_2.0to5.0TeV_BDT.weights.$ftype
  root -l 'scripts/copyHistogram.C("'${dir}'/ScaledEff2_'${config}'_'${zen}'degzenith_0.5degoffset_5.0to100.0TeV.root", "'${zen}'deg_zenith_0.5deg_offset_5.0to100.0TeV", "'${dir}'/ScaledEff2_'${config}'_'${zen}'degzenith_0.5degoffset_2.0to5.0TeV.root", "'${zen}'deg_zenith_0.5deg_offset_2.0to5.0TeV")'
fi

if test ! -f ${dir}/${config}_${zen}degzenith_0.5degoffset_1.0to2.0TeV_BDT.weights.$ftype; then
  echo "Missing file ${dir}/${config}_${zen}degzenith_0.5degoffset_1.0to2.0TeV_BDT.weights.$ftype, copying from upper energy band"
  cp ${dir}/${config}_${zen}degzenith_0.5degoffset_2.0to5.0TeV_BDT.weights.$ftype ${dir}/${config}_${zen}degzenith_0.5degoffset_1.0to2.0TeV_BDT.weights.$ftype
  root -l 'scripts/copyHistogram.C("'${dir}'/ScaledEff2_'${config}'_'${zen}'degzenith_0.5degoffset_2.0to5.0TeV.root", "'${zen}'deg_zenith_0.5deg_offset_2.0to5.0TeV", "'${dir}'/ScaledEff2_'${config}'_'${zen}'degzenith_0.5degoffset_1.0to2.0TeV.root", "'${zen}'deg_zenith_0.5deg_offset_1.0to2.0TeV")'
fi

if test ! -f ${dir}/${config}_${zen}degzenith_0.5degoffset_0.5to1.0TeV_BDT.weights.$ftype; then
  echo "Missing file ${dir}/${config}_${zen}degzenith_0.5degoffset_0.5to1.0TeV_BDT.weights.$ftype, copying from upper energy band"
  cp ${dir}/${config}_${zen}degzenith_0.5degoffset_1.0to2.0TeV_BDT.weights.$ftype ${dir}/${config}_${zen}degzenith_0.5degoffset_0.5to1.0TeV_BDT.weights.$ftype
  root -l 'scripts/copyHistogram.C("'${dir}'/ScaledEff2_'${config}'_'${zen}'degzenith_0.5degoffset_1.0to2.0TeV.root", "'${zen}'deg_zenith_0.5deg_offset_1.0to2.0TeV", "'${dir}'/ScaledEff2_'${config}'_'${zen}'degzenith_0.5degoffset_0.5to1.0TeV.root", "'${zen}'deg_zenith_0.5deg_offset_0.5to1.0TeV")'
fi
    
if test ! -f ${dir}/${config}_${zen}degzenith_0.5degoffset_0.3to0.5TeV_BDT.weights.$ftype; then
  echo "Missing file ${dir}/${config}_${zen}degzenith_0.5degoffset_0.3to0.5TeV_BDT.weights.$ftype, copying from upper energy band"
  cp ${dir}/${config}_${zen}degzenith_0.5degoffset_0.5to1.0TeV_BDT.weights.$ftype ${dir}/${config}_${zen}degzenith_0.5degoffset_0.3to0.5TeV_BDT.weights.$ftype
  root -l 'scripts/copyHistogram.C("'${dir}'/ScaledEff2_'${config}'_'${zen}'degzenith_0.5degoffset_0.5to1.0TeV.root", "'${zen}'deg_zenith_0.5deg_offset_0.5to1.0TeV", "'${dir}'/ScaledEff2_'${config}'_'${zen}'degzenith_0.5degoffset_0.3to0.5TeV.root", "'${zen}'deg_zenith_0.5deg_offset_0.3to0.5TeV")'
fi
       
if test ! -f ${dir}/${config}_${zen}degzenith_0.5degoffset_0.1to0.3TeV_BDT.weights.$ftype; then
  echo "Missing file ${dir}/${config}_${zen}degzenith_0.5degoffset_0.1to0.3TeV_BDT.weights.$ftype, copying from upper energy band"
  cp ${dir}/${config}_${zen}degzenith_0.5degoffset_0.3to0.5TeV_BDT.weights.$ftype ${dir}/${config}_${zen}degzenith_0.5degoffset_0.1to0.3TeV_BDT.weights.$ftype
  root -l 'scripts/copyHistogram.C("'${dir}'/ScaledEff2_'${config}'_'${zen}'degzenith_0.5degoffset_0.3to0.5TeV.root", "'${zen}'deg_zenith_0.5deg_offset_0.3to0.5TeV", "'${dir}'/ScaledEff2_'${config}'_'${zen}'degzenith_0.5degoffset_0.1to0.3TeV.root", "'${zen}'deg_zenith_0.5deg_offset_0.1to0.3TeV")'
fi

if test ! -f ${dir}/${config}_${zen}degzenith_0.5degoffset_0.05to0.1TeV_BDT.weights.$ftype; then
  echo "Missing file ${dir}/${config}_${zen}degzenith_0.5degoffset_0.05to0.1TeV_BDT.weights.$ftype, copying from upper energy band"
  cp ${dir}/${config}_${zen}degzenith_0.5degoffset_0.1to0.3TeV_BDT.weights.$ftype ${dir}/${config}_${zen}degzenith_0.5degoffset_0.05to0.1TeV_BDT.weights.$ftype
  root -l 'scripts/copyHistogram.C("'${dir}'/ScaledEff2_'${config}'_'${zen}'degzenith_0.5degoffset_0.1to0.3TeV.root", "'${zen}'deg_zenith_0.5deg_offset_0.1to0.3TeV", "'${dir}'/ScaledEff2_'${config}'_'${zen}'degzenith_0.5degoffset_0.05to0.1TeV.root", "'${zen}'deg_zenith_0.5deg_offset_0.05to0.1TeV")'
fi
