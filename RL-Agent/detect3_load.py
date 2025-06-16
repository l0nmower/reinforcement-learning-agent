import torch

# Must define the architecture again
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

# Load the checkpoint
checkpoint = torch.load("detect3_rl_checkpoint.pth", map_location=torch.device('cpu'))

# Rebuild and load model
model = build_model()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Get config and epsilon if needed
config = checkpoint["config"]
epsilon = checkpoint["epsilon"]

print("✅ Agent and model successfully restored.")