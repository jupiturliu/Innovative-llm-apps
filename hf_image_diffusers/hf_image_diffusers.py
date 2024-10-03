import torch
import streamlit as st
from diffusers import DiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

st.title("Create Image with prompt 🌐")
st.caption("This app allows you to create a image using Diffusers")

# Get the webpage URL from the user
prompt = st.text_input("Enter Image Prompt", type="default")

if prompt:
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
# prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images
    st.image(image, caption="Generated Image")