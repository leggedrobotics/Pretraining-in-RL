import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import wandb
import argparse
from torch.utils.data import DataLoader, TensorDataset, random_split
from einops import rearrange
from tqdm import tqdm

# Load Data from HDF5
def load_h5_dataset(h5_file):
    with h5py.File(h5_file, "r") as f:
        X = torch.tensor(f["X"][:], dtype=torch.float32)
        Y = torch.tensor(f["Y"][:], dtype=torch.float32)
        input_joint_names = f.attrs["input_joint_names"].split(",")
        output_body_names = f.attrs["output_body_names"].split(",")
        output_feature_dimensions = f.attrs["output_feature_dimensions"].split(",")
        print(f"Loaded dataset with {len(X)} samples. Input joint names: {input_joint_names}, Output body names: {output_body_names}, Output feature dimensions: {output_feature_dimensions}")
    return X, Y


# Define MLP Model
class KinematicMLP(nn.Module):
    def __init__(self, input_dim, num_bodies, num_output_features_per_body, hidden_dim = 128):
        super(KinematicMLP, self).__init__()
        self.num_bodies = num_bodies
        self.num_output_features_per_body = num_output_features_per_body
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.output_layer = nn.Linear(hidden_dim, num_bodies*num_output_features_per_body)

    def forward(self, x):
        x = x.squeeze()
        x = self.backbone(x)
        x = self.output_layer(x)
        return rearrange(x, 'b (n d) -> b n d', n=self.num_bodies, d=self.num_output_features_per_body)


# Training Function
def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    wandb.init(project="kinematic-mlp")

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader):
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader):
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Log training and validation loss
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss, "val_loss": avg_val_loss})

    wandb.finish()

# Main Function with Argument Parsing
def main():
    parser = argparse.ArgumentParser(description="Train an MLP on an HDF5 dataset")
    parser.add_argument("--data", type=str, default="./logs/datasets/kinematic_dataset_100k.h5", help="Path to HDF5 dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save_model", type=str, default="./logs/pretrain/kinematic_model.pth", help="Path to save trained model")
    args = parser.parse_args()

    # Load dataset
    X, Y = load_h5_dataset(args.data)

    # exlude the dimension of contact forces for now
    # Y = Y[..., :-1]
    # only focus on the translation part of the pose
    Y = Y[..., :3]

    # Split dataset into training and validation sets (90% train, 10% validation)
    dataset = TensorDataset(X, Y)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Define model, loss, and optimizer
    model = KinematicMLP(input_dim=X.shape[-1], 
                            num_bodies=Y.shape[-2], 
                            num_output_features_per_body=Y.shape[-1])
    criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss(beta=0.02)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    train(model, train_loader, val_loader, criterion, optimizer, args.epochs)

    # Save trained model
    torch.save(model.state_dict(), args.save_model)
    print(f"Model saved to '{args.save_model}'")

# Run script
if __name__ == "__main__":
    main()