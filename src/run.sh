# cd /src
MODEL_NAME=NRMS CUDA=0 python train.py
MODEL_NAME=NAML CUDA=0 python train.py
MODEL_NAME=LSTUR CUDA=2 python train.py

MODEL_NAME=DKN CUDA=0 python train.py
MODEL_NAME=HiFiArk CUDA=0 python train.py
MODEL_NAME=TANR CUDA=0 python train.py
MODEL_NAME=Exp1 CUDA=0 python train.py