from PIL import Image
import argparse
import random
import tensorflow as tf
from types import SimpleNamespace
from stable_diffusion_tf.stable_diffusion import get_model, text2image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--H",
            type=int,
            default=512,
            help="image height, in pixel space",
        )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="number of ddim sampling steps"
    )
    parser.add_argument(
        "--extras",
        type=str,
        default="HQ, 4K",
        help="to append to prompt"
    )
    
    parser.add_argument(
        "--n",
        type=int,
        default=-1,
        help="How many prompts?"
    )
    args = parser.parse_args()
    return args


def main(args):    
    ## Cities

    artists = ("Makoto Shinkai, Joseph Mallord William Turner, Guillermo del Toro,"
               "Salvador Dali, DOOM, Hasui Kawase, Hidetaka Miyazaki, Hans Ruedi Giger,"
               "John Martin, Dawid Jurek, Claude Monet, Vincent Van Gogh, Alan Lee, Beksiński").split(",")

    cities = "Santigo in Chile, Valparaiso in Chile".split(",")

    extras = "HQ, 4K"

    
    results = []
    for artist in artists[:args.n]:
        for city in cities:
            prompt = f"The city of {city} in the style of {artist} {extras}"
            pil_img = run_inference(prompt)
            results.append([prompt, pil_img, artist, city])
            
            
    ## Wandb
    import wandb

    table = wandb.Table(columns=["prompt", "image", "artist", "city"])

    for prompt, pil_img, artist, city in results:
        table.add_data(prompt, wandb.Image(pil_img), artist, city)

    with wandb.init(project="stable_diffusions", config=args):
        wandb.log({"Results":table})
    

def run_inference(prompt):
    tf.random.set_seed(random.randint(0,1e10))
    print(f"Current prompt: {prompt}")
    img = text2image(prompt, 
                     img_height=args.H, 
                     img_width=args.W,  
                     text_encoder=text_encoder, 
                     diffusion_model=diffusion_model, 
                     decoder=decoder,  
                     batch_size=1, 
                     n_steps=args.steps, 
                     unconditional_guidance_scale =args.scale , 
                     temperature = 1
                    )
    pil_img = Image.fromarray(img[0])
    return pil_img
        
if __name__ == "__main__":
    args = parse_args()

    ## loads models globally
    text_encoder, diffusion_model, decoder = get_model(args.H, args.W, download_weights=True)
    main(args)