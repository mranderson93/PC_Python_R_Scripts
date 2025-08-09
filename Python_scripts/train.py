import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from Transformer import TabTransformer
from engine import train
from Transformer_dataloader import pass_dataloader
# # Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set up the device Agnostic Code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set number of epochs
NUM_EPOCHS = 50

# Set Up the DataLoader
train_loader, test_loader = pass_dataloader()

# Transformer Model
model_EF = TabTransformer(input_dim=100,
                         d_model = 128,
                         nhead = 4,
                         num_layers=4,
                         dropout = 0.4).to(device)

# Setup loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model_EF.parameters(),
                             lr=0.01,
                             weight_decay=1e-4)

# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Train model_EF_results
model_EF_results = train(model=model_EF,
                        train_dataloader=train_loader,
                        test_dataloader=test_loader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS,
                        device = device)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")