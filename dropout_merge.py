import argparse
import tomllib
from diffusers import DiffusionPipeline, KDPM2DiscreteScheduler
import torch
import torch.nn as nn
import os
import peft
from collections import defaultdict
from dropouts import OrthogonalDropout, NormalDropout

class OrthLorasInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, linear_state_dict, lora_list, orthogonal=True):
        super(OrthLorasInjectedLinear, self).__init__()
        self.orthogonal = orthogonal
        self.origial_linear = nn.Linear(in_features, out_features, bias)
        self.origial_linear.load_state_dict(linear_state_dict)
        self.loras = nn.ModuleList()
        self.ps = []
        self.alphas = []
        self.rs = []
        self.scales = []

        for lora in lora_list:
            loraA = nn.Linear(lora["loraA"]["weight"].shape[1], lora["loraA"]["weight"].shape[0], "bias" in lora["loraA"])
            loraA.load_state_dict(lora["loraA"])
            loraB = nn.Linear(lora["loraB"]["weight"].shape[1], lora["loraB"]["weight"].shape[0], "bias" in lora["loraB"])
            loraB.load_state_dict(lora["loraB"])

            self.loras.append(nn.Sequential(loraA, loraB))
            self.ps.append(lora["p"])
            self.alphas.append(lora["alpha"])
            self.rs.append(lora["loraA"]["weight"].shape[0])
            self.scales.append(lora["scale"])

        if self.orthogonal:
            self.orth_dropout = OrthogonalDropout(self.ps)
        else:
            self.orth_dropout = NormalDropout(self.ps)

    def forward(self, x):
        x_list = []
        for lora in self.loras:
            x_list.append(lora(x))
        # outs = [scale * alpha * out / r for scale, alpha, r, out in zip(self.scales, self.alphas, self.rs, self.orth_dropout(x_list))]
        # assert torch.sum(outs[0] * outs[1]) == 0
        return self.origial_linear(x) + sum([scale * alpha * out / r for scale, alpha, r, out in zip(self.scales, self.alphas, self.rs, self.orth_dropout(x_list))])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/duck_toy_robot_toy.toml")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--orthogonal", action="store_true")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    print(config)

    if args.orthogonal:
        output_path = os.path.join(args.output_path, "orthogonal_merge", config["save_folder_name"])
    else:
        output_path = os.path.join(args.output_path, "dropout_merge", config["save_folder_name"])
    os.makedirs(output_path, exist_ok=True)

    loras_dict_unet = defaultdict(list)
    loras_dict_te = defaultdict(list)
    loras_dict_te2 = defaultdict(list)
    loras_dicts = [loras_dict_unet, loras_dict_te, loras_dict_te2]

    for lora in config["loras"]:
        pipe = DiffusionPipeline.from_pretrained(config["pipe_id"], torch_dtype=torch.float16).to("cuda")
        pipe.load_lora_weights(lora["file_path"], weight_name=lora["file_name"], adapter_name=lora["adapter_name"])
        named_modules = [pipe.unet.named_modules(), pipe.text_encoder.named_modules(), pipe.text_encoder_2.named_modules()]
        currentp=lora["dropout_rate"]
        currentscale = 1. / (1. - currentp) * lora["adapter_weight"]

        for loras_dict, nms in zip(loras_dicts, named_modules):
            for fullname, module in nms:
                if isinstance(module, peft.tuners.lora.layer.Linear):
                    loras_dict[fullname].append({"loraA": list(module.lora_A.values())[0].state_dict(), "loraB": list(module.lora_B.values())[0].state_dict(), "alpha": list(module.lora_alpha.values())[0], "p": currentp, "scale": currentscale})

    pipe = DiffusionPipeline.from_pretrained(config["pipe_id"], torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)

    ancestors = [pipe.unet, pipe.text_encoder, pipe.text_encoder_2]
    named_modules = [pipe.unet.named_modules(), pipe.text_encoder.named_modules(), pipe.text_encoder_2.named_modules()]

    for loras_dict, ancestor, nms in zip(loras_dicts, ancestors, named_modules):
        for fullname, module in nms:
            if fullname in loras_dict:
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                tmp = OrthLorasInjectedLinear(parent._modules[name].in_features, parent._modules[name].out_features, not parent._modules[name].bias == None, parent._modules[name].state_dict(), loras_dict[fullname], args.orthogonal).to(parent._modules[name].weight.device).to(parent._modules[name].weight.dtype)
                parent._modules[name] = tmp

    for i in range(config["num_batches"]):
        images = pipe(config["prompt"], num_images_per_prompt=config["batch_size"], cross_attention_kwargs={"scale": 1.0}, negative_prompt=config["neg_prompt"], num_inference_steps=40, clip_skip=2, generator=torch.manual_seed(i), guidance_scale=7, width=1024, height=1024).images
        for j, image in enumerate(images):
            image.save(f"{output_path}/{i*config["batch_size"]+j}.png")
    
if __name__ == "__main__":
    main()