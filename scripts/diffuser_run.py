import torch
import matplotlib.pyplot as plt

from daam import trace, set_seed
from diffusers import DiffusionPipeline

seed = set_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

with torch.no_grad():
  with trace(base) as tc:
    image = base(prompt=prompt, generator=seed)
    exp = tc.to_experiment('experiment-dir')
    heat_map = tc.compute_global_heat_map()
    heat_map = heat_map.compute_word_heat_map("jungle")
    heat_map.plot_overlay(image.images[0])
    exp.save()