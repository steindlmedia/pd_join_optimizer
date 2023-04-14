#!/bin/bash

# Define colors for formatting console output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
BLUE='\033[0;34m'

print_step() {
    echo -e "${BLUE}>>> ${1}${NC}"
}

# Build the distribution package
print_step "Building the distribution package"
cd implementation
python setup.py sdist

# Create a new conda environment with Python 3.11.2
print_step "Creating a new conda environment with Python 3.11.2"
conda create -n test-packaging python=3.11.2 -y

# Activate the new conda environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate test-packaging

# Install the package in the dist folder using pip3
print_step "Installing the package in the dist folder using pip3"
cd dist
pip3 install *.tar.gz

# Run Python script for verification of the package
print_step "Running a Python script for verification of the package functionality"
PYTHON_SCRIPT="import pandas as pd
import pd_join_optimizer

r_1 = pd.DataFrame({'x_3': [('b_1'), ('b_2'), ('b_3')]})
r_2 = pd.DataFrame({'x_2': [('c_1'), ('c_1'), ('c_1')], 'x_4': [('a_1'), ('a_1'), ('a_2')], 'x_3': [('b_1'), ('b_2'), ('b_2')]})
r_3 = pd.DataFrame({'x_1': [('s_1'), ('s_1'), ('s_3'), ('s_3'), ('s_2')], 'x_2': [('c_1'), ('c_1'), ('c_3'), ('c_1'), ('c_2')], 'x_3': [('b_1'), ('b_2'), ('b_1'), ('b_4'), ('b_3')]})
r_4 = pd.DataFrame({'x_2': [('c_1'), ('c_1'), ('c_4')], 'x_3': [('b_2'), ('b_1'), ('b_6')]})
r_5 = pd.DataFrame({'x_5': [('c_1'), ('c_1'), ('c_4')], 'x_6': [('b_2'), ('b_1'), ('b_6')]})

dataframes = [r_1, r_2, r_3, r_4, r_5]
print(pd_join_optimizer.join_optimized(dataframes)[0])"

var=$(python -c "$PYTHON_SCRIPT")
EXIT_CODE=$?

echo $var

# Check the exit code of the Python script
print_step "Checking the exit code of the Python script"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Script completed successfully 🎉"
else
    echo -e "${RED}Script failed with exit code $?${NC} ❌"
fi

# Deactivate the conda environment
print_step "Deactivating the conda environment"
conda deactivate

# Remove the conda environment
print_step "Removing the conda environment"
conda remove -n test-packaging --all -y

if [ $EXIT_CODE -eq 0 ]; then
    exit 0
else
    exit 1
fi