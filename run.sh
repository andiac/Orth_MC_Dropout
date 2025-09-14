#!/bin/bash

MERGE_CASE_NAMES=(
    "kimoju_cloud"
    "kimoju_dog6"
    "cat2_monster_toy"
    "duck_toy_robot_toy"
    "dog6_cloud"
    "duck_toy_yarn"
)

SINGLE_CASE_NAMES=(
    "keqing"
    "kimoju"
    "cloud"
    "dog6"
    "duck_toy"
    "backpack_dog"
)

for CASE_NAME in ${MERGE_CASE_NAMES[@]}; do
    python direct_merge.py --config ./configs/merge/${CASE_NAME}.toml
    python dropout_merge.py --config ./configs/merge/${CASE_NAME}.toml
    python dropout_merge.py --config ./configs/merge/${CASE_NAME}.toml --orthogonal
    python concatenate.py --case_name ${CASE_NAME}
done

for CASE_NAME in ${SINGLE_CASE_NAMES[@]}; do
    for dropout in 0.1 0.3 0.5 0.7 0.9; do
    python single_lora.py --config ./configs/single/${CASE_NAME}.toml --dropout ${dropout}
    done
done
