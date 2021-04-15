#!/bin/bash

export NR=${1}

# setup environment
echo "setting up environment..."
cd /afs/desy.de/user/j/jabuschh/
source .bashrc
source activate env_python2

# go to desired directory
# cd /nfs/dust/cms/user/jabuschh/anaconda2/envs/env_python2/
cd /nfs/dust/cms/user/jabuschh/NonResonantTTbar/DNNClassifier_py2 # DNNClassifier_py3

# setup root environment
echo "setting up root..."
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.22.00/x86_64-centos7-gcc48-opt/bin/thisroot.sh

echo "starting job..."
echo ""

./steer_array_wSystems_DNN.py
