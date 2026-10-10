import torch
from diffusers import StableDiffusionPipeline

# 1. Maintain hardware optimization configurations
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# 2. Memory optimization rules for 16GB shared limits
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

prompt = input("Enter your prompt for image generation: ")

print("Running GPU diffusion loop steps...")
output = pipe(
    prompt, width=384, height=384, num_inference_steps=15, output_type="latent"
)
latents = output.images

print("Diffusion loop complete. Removing accelerator hooks to unlock VAE weights...")
# Break the automatic CPU offload manager hooks on the VAE component
if hasattr(pipe, "remove_all_hooks"):
    pipe.remove_all_hooks()
elif hasattr(pipe.vae, "_hf_hook"):
    del pipe.vae._hf_hook

print("Moving VAE to CPU...")
# Isolate the VAE to the system host memory architecture
pipe.vae.to("cpu")
pipe.vae.float()

# Cast and scale the raw latency arrays to full float32 math blocks on the host CPU
latents = latents.to("cpu").float()
latents = latents / pipe.vae.config.scaling_factor

print("Decoding final image matrices safely on the CPU processor...")
with torch.no_grad():
    image_tensor = pipe.vae.decode(latents).sample

# Process raw floating arrays back into standard image maps
images = pipe.image_processor.postprocess(image_tensor, output_type="pil")

for i, image in enumerate(images):
    image.save(f"{i}.png")

print("Success! Image generated successfully and saved to cyberpunk_city.png")
