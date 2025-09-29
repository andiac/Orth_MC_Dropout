#!/bin/bash

MERGE_CASE_NAMES=(
    "kimoju_cloud"
    "cat2_monster_toy"
    "duck_toy_robot_toy"
)

SINGLE_CASE_NAMES=(
    "keqing"
    "kimoju"
    "dog6"
)

for CASE_NAME in ${MERGE_CASE_NAMES[@]}; do
    python direct_merge.py --config ./configs/merge/${CASE_NAME}.toml
    python dropout_merge.py --config ./configs/merge/${CASE_NAME}.toml
    python dropout_merge.py --config ./configs/merge/${CASE_NAME}.toml --orthogonal
    python concatenate.py --case_name ${CASE_NAME}
done

for CASE_NAME in ${SINGLE_CASE_NAMES[@]}; do
    for dropout in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python single_lora.py --config ./configs/single/${CASE_NAME}.toml --dropout ${dropout}
    done
done
