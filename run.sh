#!/bin/bash

CASE_NAMES=(
    "kimoju_cloud"
    "cat2_monster_toy"
    "duck_toy_robot_toy"
)

for CASE_NAME in ${CASE_NAMES[@]}; do
    python direct_merge.py --config ./configs/${CASE_NAME}.toml
    python dropout_merge.py --config ./configs/${CASE_NAME}.toml
    python dropout_merge.py --config ./configs/${CASE_NAME}.toml --orthogonal
    python concatenate.py --case_name ${CASE_NAME}
done
