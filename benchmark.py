import wandb
import argparse, random
from PIL import Image
from types import SimpleNamespace

from tensorflow import keras
from stable_diffusion_tf.stable_diffusion import Text2Image


PROJECT = "stable_diffusions"
JOB_TYPE = "benchmark"
GROUP = "tensorflow"


defaults = SimpleNamespace(H=512, W=512, steps=20, scale=7.5, temp=1, seed=42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The city of Santiago in Chile by Makoto Shinkai")
    parser.add_argument("--H",type=int,default=defaults.H,help="image height, in pixel space")
    parser.add_argument("--W",type=int,default=defaults.W,help="image width, in pixel space")
    parser.add_argument("--scale",type=float,default=defaults.scale,help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    parser.add_argument("--steps",type=int,default=defaults.steps,help="number of ddim sampling steps")
    parser.add_argument("--temp",type=float,default=defaults.temp,help="temperature")
    parser.add_argument("--seed",type=int,default=defaults.seed,help="random seed")
    parser.add_argument("--jit_compile", action="store_true", help="jit_compile")
    parser.add_argument("--mp", action="store_true", help="mp")
    parser.add_argument("--log", action="store_true", help="log result to wandb")
    parser.add_argument("--n", type=int, default=1, help="number of runs")
    args = parser.parse_args()
    return args

def main(args):
    if args.mp:
        print("Using mixed precision.")
        keras.mixed_precision.set_global_policy("mixed_float16")

    generator = Text2Image(img_height=args.H, img_width=args.W, jit_compile=args.jit_compile)
    results = []
    for _ in range(args.n):
        img = generator.generate(
                args.prompt,
                num_steps=args.steps,
                unconditional_guidance_scale=args.scale,
                temperature=1,
                batch_size=1,
                seed=args.seed if args.n == 1 else random.randint(0,1e15),
        )
        results.append(img)

    if args.log:
        table = wandb.Table(columns=["prompt", "image"])
        for img in results:
            pil_img = Image.fromarray(img[0])
            table.add_data(args.prompt, wandb.Image(pil_img))
        wandb.log({"Inference_results":table})

if __name__ == "__main__":
    args = parse_args()

    with wandb.init(project=PROJECT, job_type=JOB_TYPE, group=GROUP, config=args):
        main(args)
