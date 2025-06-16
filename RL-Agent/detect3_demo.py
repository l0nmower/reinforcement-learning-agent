# -----------------------------
# STEP 1 – Imports
# -----------------------------
import torch
import matplotlib.pyplot as plt
import random
import time
from torchvision.datasets import MNIST
from torchvision import transforms
from IPython.display import clear_output

# -----------------------------
# STEP 2 – Define model structure (must match training)
# -----------------------------
def build_model():
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(36864, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2)
    )

# -----------------------------
# STEP 3 – Load trained model from .pth file
# -----------------------------
checkpoint = torch.load("detect3_rl_checkpoint.pth", map_location=torch.device('cpu'))

model = build_model()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("✅ Model loaded successfully from checkpoint.\n")

# -----------------------------
# STEP 4 – Load MNIST test data
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

# -----------------------------
# STEP 5 – Predict on 100 random images with live display
# -----------------------------
print("🔍 Starting live prediction on 100 random MNIST images...\n")

for i in range(100):
    # clear_output(wait=True)

    # Random image from MNIST test set
    idx = random.randint(0, len(test_dataset) - 1)
    image, label = test_dataset[idx]
    image_input = image.unsqueeze(0)  # [1, 1, 28, 28]

    # Predict using the loaded model
    with torch.no_grad():
        q_vals = model(image_input)
        action = torch.argmax(q_vals).item()

    # Interpret prediction
    prediction = "YES (3)" if action == 1 else "NO (not 3)"
    true_label = "3" if label == 3 else "not 3"

    # Display the image and result
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Prediction: {prediction} | True Label: {true_label}", fontsize=15)
    plt.axis("off")
    plt.show()

    time.sleep(1.5)  # delay between images