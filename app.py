import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


# Model Definition
class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim, num_classes, image_dim):
        super(ConditionalVAE, self).__init__()
        self.label_embed = nn.Embedding(num_classes, 50)
        self.fc1 = nn.Linear(image_dim + 50, 512)
        self.fc2_mu = nn.Linear(512, latent_dim)
        self.fc2_logvar = nn.Linear(512, latent_dim)
        self.fc3 = nn.Linear(latent_dim + 50, 512)
        self.fc4 = nn.Linear(512, image_dim)

    def encode(self, x, labels):
        labels_embed = self.label_embed(labels)
        x = torch.cat((x, labels_embed), dim=1)
        h = F.relu(self.fc1(x))
        return self.fc2_mu(h), self.fc2_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        labels_embed = self.label_embed(labels)
        z = torch.cat((z, labels_embed), dim=1)
        h = F.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar


# Load trained model
@st.cache_resource
def load_model():
    model = ConditionalVAE(latent_dim=20, num_classes=10, image_dim=28 * 28)
    model.load_state_dict(torch.load("model/cvae_mnist.pth", map_location=torch.device("cpu")))
    model.eval()
    return model


model = load_model()

# Streamlit UI
st.title("Handwritten Digit Image Generator")
st.markdown("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))

if st.button("Generate Images"):
    st.write(f"Generated images of digit {digit}")

    with torch.no_grad():
        z = torch.randn(5, 20)
        labels = torch.tensor([digit] * 5)
        samples = model.decode(z, labels)
        samples = torch.sigmoid(samples).view(-1, 1, 28, 28)

        # Display in 1 row
        cols = st.columns(5)
        for i in range(5):
            img = transforms.ToPILImage()(samples[i])
            cols[i].image(img, caption=f"Sample {i + 1}", width=100)
