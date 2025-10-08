# Orth_MC_Dropout

This is the official GitHub repository for the paper **"Rethinking Inter-LoRA Orthogonality in Adapter Merging: Insights from Orthogonal Monte Carlo Dropout."** [arXiv](https://arxiv.org/abs/2510.03262)

## Quick Start

Run the following command directly:

```bash
bash run.sh
```

After execution:

* Generated images of direct merge, dropout merge, and orthogonal merge will be saved in `./output`
* Concatenated preview images of the three methods will be available in `./concat_img`
* The results of evaluating redundancy of the dropout rate will be saved in `./output_single_lora`

## Adding a New Case for Merging

1. Add a configuration file in `.toml` format to the `./configs/merge` folder.
2. Add the case name to the `MERGE_CASE_NAMES` list in `run.sh`.
3. Make sure the `save_folder_name` property inside the `.toml` file matches the filename (excluding the extension).

Refer to the sample configs in this repository for examples.

## Adding a New Case for Single LoRA with Different Dropout Rates

1. Add a configuration file in `.toml` format to the `./configs/single` folder.
2. Add the case name to the `SINGLE_CASE_NAMES` list in `run.sh`.
3. Make sure the `save_folder_name` property inside the `.toml` file matches the filename (excluding the extension).

Refer to the sample configs in this repository for examples.

## Showcase

Below are some comparison results from the paper:

![concat\_img1](./concat_img/kimoju_cloud.png)
![concat\_img2](./concat_img/cat2_monster_toy.png)
![concat\_img3](./concat_img/duck_toy_robot_toy.png)

---

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

