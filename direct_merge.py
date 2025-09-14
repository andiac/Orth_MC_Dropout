import argparse
import tomllib
from diffusers import DiffusionPipeline, KDPM2DiscreteScheduler
import torch
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/duck_toy_robot_toy.toml")
    parser.add_argument("--output_path", type=str, default="./output")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    print(config)

    output_path = os.path.join(args.output_path, "direct_merge", config["save_folder_name"])
    os.makedirs(output_path, exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(config["pipe_id"], torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)

    for lora in config["loras"]:
        pipe.load_lora_weights(lora["file_path"], weight_name=lora["file_name"], adapter_name=lora["adapter_name"])
    pipe.set_adapters([lora["adapter_name"] for lora in config["loras"]], adapter_weights=[lora["adapter_weight"] for lora in config["loras"]])

    for i in range(config["num_batches"]):
        images = pipe(config["prompt"], num_images_per_prompt=config["batch_size"], cross_attention_kwargs={"scale": 1.0}, negative_prompt=config["neg_prompt"], num_inference_steps=40, clip_skip=2, generator=torch.manual_seed(i), guidance_scale=7, width=1024, height=1024).images
        for j, image in enumerate(images):
            image.save(f"{output_path}/{i*config["batch_size"]+j}.png")
    
if __name__ == "__main__":
    main()