#!/bin/bash
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Setting up the environment
source env/bin/activate
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONPATH=${PYTHONPATH}:$(realpath "$PWD")
