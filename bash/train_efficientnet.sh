#!/bin/bash
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

#########################################################################################################

# Bash script for running QAT EfficientNet-Lite training configuration.
# IMAGES_DIR: path to local raw imagenet dataset and
# MODEL_DIR: path to model's pretrained weights
# should be at least specified.
#
# Example of using this script:
# $ ./bash/train_efficientnet.sh --IMAGES_DIR path_to_imagenet_raw --MODEL_DIR path_to_weights
#
# For getting usage info:
# $ ./bash/train_efficientnet.sh
#
# Other configurable params:
# N_BITS: 3(default), 4 are currently supported
# METHOD: freeze (default; iterative weight freezing), damp (oscillations dampening)
#
# The script may be extended with further input parameters (please refer to "/utils/click_options.py")

#########################################################################################################

source bash/set_env.sh

MODEL='efficientnet'
N_BITS=3
METHOD='freeze'

for ARG in "$@"
do
    key=$(echo $ARG | cut -f1 -d=)
    value=$(echo $ARG | cut -f2 -d=)

    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare $v="${value}"
    fi
done

if [[ -z $IMAGES_DIR ]] || [[ -z $MODEL_DIR ]]; then
  echo "Usage: $(basename "$0")
  --IMAGES_DIR=[path to imagenet_raw]
  --MODEL_DIR=[path to model's pretrained weights]
  --N_BITS=[3(default), 4]
  --METHOD=[freeze(default), damp]"
  exit 1
fi

if [ $N_BITS -ne 3 ] && [ $N_BITS -ne 4 ]; then
  echo "Only 3,4 bits configuration currently supported"
  exit 1
fi

if [ "$METHOD" != 'freeze' ] && [ "$METHOD" != 'damp' ]; then
  echo "Only methods 'damp' and 'freeze' are currently supported."
  exit 1
fi

CMD_ARGS='--architecture efficientnet_lite0_quantized
--act-quant-method MSE
--weight-quant-method MSE
--optimizer SGD
--max-epochs 50
--learning-rate-schedule cosine:0
--sep-quant-optimizer
--quant-optimizer Adam
--quant-learning-rate 1e-5
--quant-weight-decay 0.0'

# QAT methods
if [ $METHOD == 'freeze' ]; then
  CMD_QAT='--oscillations-dampen-weight 0 --oscillations-dampen-weight-final 0.1'
  if [ $N_BITS == 3 ]; then
    CMD_BITS='--n-bits 3 --learning-rate 0.01 --weight-decay 5e-05 --oscillations-freeze-threshold-final 0.005'
  else
    CMD_BITS='--n-bits 4 --learning-rate 0.0033 --weight-decay 1e-04 --oscillations-freeze-threshold-final 0.015'
  fi
else
  CMD_QAT='--oscillations-dampen-weight 0 --oscillations-dampen-weight-final 0.1'
  if [ $N_BITS == 3 ]; then
    CMD_BITS='--n-bits 3 --learning-rate 0.01  --weight-decay 5e-5'
  else
    CMD_BITS='--n-bits 4 --learning-rate 0.0033 --weight-decay 1e-4'
  fi
fi

CMD_ARGS="$CMD_ARGS $CMD_QAT $CMD_BITS"

python main.py train-quantized \
--images-dir $IMAGES_DIR \
$CMD_ARGS
