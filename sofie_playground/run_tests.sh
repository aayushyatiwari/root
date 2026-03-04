#!/bin/bash
cd models
python3 ../python/elu_test.py
python3 ../python/maxpool2d_test.py
root -l -q ../cpp/generate/elu_pt_test.cxx
root -l -q ../cpp/generate/maxpool2d_pt_test.cxx
root -l -q ../cpp/infer/elu_test.C
root -l -q ../cpp/infer/maxpool_test.C
