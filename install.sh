#!/bin/bash

pip install -r requirements.txt
cd ./packages_from_deepunrollnet
cd ./package_correlation
python setup.py install
cd ../package_forward_warp
python setup.py install
cd ../package_core
python setup.py install
cd ..
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
