#!/bin/bash --login

set -e

conda activate multirtc

cd /home/conda/multirtc

python src/multirtc/multirtc.py "$@"
