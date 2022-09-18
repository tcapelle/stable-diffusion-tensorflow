import numpy as np
from tqdm import tqdm
import math

import tensorflow as tf
from tensorflow import keras

from .autoencoder_kl import Decoder
from .diffusion_model import UNetModel
from .clip_encoder import CLIPTextTransformer
from .clip_tokenizer import SimpleTokenizer
from .constants import _TOKENS_UNCONDITIONAL, _ALPHAS_CUMPROD

MAX_TEXT_LEN = 77


class Text2Image:
    def __init__(
        self, img_height=1000, img_width=1000, batch_size=1, jit_compile=False
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.tokenizer = SimpleTokenizer()

        text_encoder, diffusion_model, decoder = get_models(img_height, img_width)
        self.text_encoder = text_encoder
        self.diffusion_model = diffusion_model
        self.decoder = decoder
        if jit_compile:
            self.text_encoder.compile(jit_compile=True)
            self.diffusion_model.compile(jit_compile=True)
            self.decoder.compile(jit_compile=True)

        tokens_unconditional = np.array(_TOKENS_UNCONDITIONAL)[None].astype("int32")
        tokens_unconditional = np.repeat(tokens_unconditional, batch_size, axis=0)
        self.tokens_unconditional = tf.convert_to_tensor(tokens_unconditional)

    def timestep_embedding(self, timesteps, dim=320, max_period=10000):
        half = dim // 2
        freqs = np.exp(
            -math.log(max_period) * np.arange(0, half, dtype="float32") / half
        )
        args = np.array(timesteps) * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)])
        return tf.convert_to_tensor(embedding.reshape(1, -1))

    def get_model_output(
        self, latent, t, context, unconditional_context, unconditional_guidance_scale
    ):
        timesteps = np.array([t])
        t_emb = self.timestep_embedding(timesteps)
        t_emb = np.repeat(t_emb, self.batch_size, axis=0)
        unconditional_latent = self.diffusion_model.predict_on_batch(
            [latent, t_emb, unconditional_context]
        )
        latent = self.diffusion_model.predict_on_batch([latent, t_emb, context])
        return unconditional_latent + unconditional_guidance_scale * (
            latent - unconditional_latent
        )

    def get_x_prev_and_pred_x0(self, x, e_t, index, a_t, a_prev, temperature):
        sigma_t = 0
        sqrt_one_minus_at = math.sqrt(1 - a_t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)

        # Direction pointing to x_t
        dir_xt = math.sqrt(1.0 - a_prev - sigma_t**2) * e_t
        noise = sigma_t * tf.random.normal(x.shape) * temperature
        x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev, pred_x0

    def generate(
        self, prompt, n_steps=25, unconditional_guidance_scale=7.5, temperature=1
    ):
        n_h = self.img_height // 8
        n_w = self.img_width // 8

        inputs = self.tokenizer.encode(prompt)
        assert len(inputs) < 77, "Prompt is too long (should be < 77 tokens)"
        phrase = inputs + [49407] * (77 - len(inputs))

        pos_ids = tf.convert_to_tensor(np.array(list(range(77)))[None].astype("int32"))
        pos_ids = np.repeat(pos_ids, self.batch_size, axis=0)

        # Get context
        phrase = np.array(phrase)[None].astype("int32")
        phrase = np.repeat(phrase, self.batch_size, axis=0)
        phrase = tf.convert_to_tensor(phrase)
        context = self.text_encoder.predict_on_batch([phrase, pos_ids])

        unconditional_context = self.text_encoder.predict_on_batch(
            [self.tokens_unconditional, pos_ids]
        )

        timesteps = list(np.arange(1, 1000, 1000 // n_steps))
        print(f"Running for {timesteps} timesteps")

        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]

        latent = tf.random.normal((self.batch_size, n_h, n_w, 4))

        progbar = tqdm(list(enumerate(timesteps))[::-1])
        for index, timestep in progbar:
            progbar.set_description(f"{index:3d} {timestep:3d}")
            e_t = self.get_model_output(
                latent,
                timestep,
                context,
                unconditional_context,
                unconditional_guidance_scale,
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            x_prev, pred_x0 = self.get_x_prev_and_pred_x0(
                latent, e_t, index, a_t, a_prev, temperature
            )
            latent = x_prev

        decoded = self.decoder.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255
        return np.clip(decoded, 0, 255).astype("uint8")


def get_models(img_height, img_width, download_weights=True):
    n_h = img_height // 8
    n_w = img_width // 8

    # Create text encoder
    input_word_ids = keras.layers.Input(shape=(MAX_TEXT_LEN,), dtype=tf.int32)
    input_pos_ids = keras.layers.Input(shape=(MAX_TEXT_LEN,), dtype=tf.int32)
    embeds = CLIPTextTransformer()([input_word_ids, input_pos_ids])
    text_encoder = keras.models.Model([input_word_ids, input_pos_ids], embeds)

    # Creation diffusion UNet
    context = keras.layers.Input((MAX_TEXT_LEN, 768))
    t_emb = keras.layers.Input((320,))
    latent = keras.layers.Input((n_h, n_w, 4))
    unet = UNetModel()
    diffusion_model = keras.models.Model(
        [latent, t_emb, context], unet([latent, t_emb, context])
    )

    # Create decoder
    latent = keras.layers.Input((n_h, n_w, 4))
    decoder = Decoder()
    decoder = keras.models.Model(latent, decoder(latent))

    text_encoder_weights_fpath = keras.utils.get_file(
        origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/text_encoder.h5",
        file_hash="d7805118aeb156fc1d39e38a9a082b05501e2af8c8fbdc1753c9cb85212d6619",
    )
    diffusion_model_weights_fpath = keras.utils.get_file(
        origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/diffusion_model.h5",
        file_hash="a5b2eea58365b18b40caee689a2e5d00f4c31dbcb4e1d58a9cf1071f55bbbd3a",
    )
    decoder_weights_fpath = keras.utils.get_file(
        origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/decoder.h5",
        file_hash="0e2c7e4bbf32962daa37ff3b744cf2503f000ead2c8ffaa412e6cf5bec066b6b",
    )

    text_encoder.load_weights(text_encoder_weights_fpath)
    diffusion_model.load_weights(diffusion_model_weights_fpath)
    decoder.load_weights(decoder_weights_fpath)

    return text_encoder, diffusion_model, decoder

