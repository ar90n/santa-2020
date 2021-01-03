#!/usr/bin/env bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
CONFIG_DIR=${SCRIPT_DIR}/../config
MY_DATA_DIR=/kaggle/input/my-santa-2020-data


for YAML_PATH in `ls ${CONFIG_DIR}/*.yaml`;
do
    ln -s $YAML_PATH $MY_DATA_DIR
done;
