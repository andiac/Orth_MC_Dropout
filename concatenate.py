# concatenate the output images
import argparse
from PIL import Image
import os

parser = argparse.ArgumentParser()
parser.add_argument("--case_name", type=str, default="")
args = parser.parse_args()

if args.case_name == "":
    case_names = [
        "keqing_kimoju09",
        "keqing_kimoju",
        "keqing_dog6",
        "miku_dog6",
    ]
else:
    case_names = [args.case_name]

WIDTH = 1024
HEIGHT = 1024
DOWNSCALE_FACTOR = 4

for case_name in case_names:
    save_path = os.path.join(".", "concat_img", case_name + ".png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    normal_image_list = [Image.open(os.path.join(".", "output", "direct_merge", case_name, f)) for f in os.listdir(os.path.join(".", "output", "direct_merge", case_name)) if f.endswith(".png")]
    dropout_image_list = [Image.open(os.path.join(".", "output", "dropout_merge", case_name, f)) for f in os.listdir(os.path.join(".", "output", "dropout_merge", case_name)) if f.endswith(".png")]
    orthogonal_image_list = [Image.open(os.path.join(".", "output", "orthogonal_merge", case_name, f)) for f in os.listdir(os.path.join(".", "output", "orthogonal_merge", case_name)) if f.endswith(".png")]

    assert len(normal_image_list) == len(dropout_image_list) == len(orthogonal_image_list)

    new_image = Image.new("RGB", (WIDTH * 3, HEIGHT * len(normal_image_list)))
    for i in range(len(normal_image_list)):
        new_image.paste(normal_image_list[i], (0, i * HEIGHT))
        new_image.paste(dropout_image_list[i], (WIDTH, i * HEIGHT))
        new_image.paste(orthogonal_image_list[i], (WIDTH * 2, i * HEIGHT))

    # downscale the image by DOWNSCALE_FACTOR
    new_image = new_image.resize((new_image.width // DOWNSCALE_FACTOR, new_image.height // DOWNSCALE_FACTOR))

    new_image.save(save_path)
    