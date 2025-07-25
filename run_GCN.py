# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv
import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_knn_edge_index(node_time_series, k=5):
    """
    Compute the k-Nearest Neighbors (kNN) graph for a set of node time series.
    """
    if isinstance(node_time_series, torch.Tensor):
        node_time_series = node_time_series.cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(node_time_series)
    distances, indices = nbrs.kneighbors(node_time_series)

    # Remove self-loops (the first neighbor is always the node itself)
    indices = indices[:, 1:]  # Shape: (num_nodes, k)

    num_nodes = node_time_series.shape[0]
    source_nodes = np.repeat(np.arange(num_nodes), k)
    target_nodes = indices.flatten()

    edge_index = np.vstack((source_nodes, target_nodes))  # Shape: (2, num_edges)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return edge_index


class TemporalGraphNetwork(nn.Module):
    def __init__(self, kernels, dropout=0.5):
        super().__init__()
        self.kernels = kernels
        self.dropout = nn.Dropout(dropout)

        # Create a sequence of EdgeConv layers
        self.edge_convs = nn.ModuleList([
            EdgeConv(
                nn.Sequential(
                    nn.Linear(2 if i == 0 else 2*kernels[i - 1], kernels[i]),
                    nn.ReLU(),
                    nn.BatchNorm1d(kernels[i])
                )
            )
            for i in range(len(kernels))
        ])

        # Calculate the total number of features after the last EdgeConv layer
        self.fc_input_dim = sum(kernels)

        # Final fully connected layer for classification (2 output classes)
        self.fc = nn.Linear(self.fc_input_dim, 2)

    def forward(self, x, edge_index):
        # x: shape (num_frames, num_rois, 1) for a single subject
        num_frames, num_rois, num_features = x.shape
        #print(f"Input shape: {x.shape}")  # Debug print

        features_tensor = []  # List to store features for all time frames

        for i in range(num_frames):
            out = x[i, :, :]  # Select the i-th time frame (Shape: [num_rois, num_features])
            features = []  # Temporary list for each convolution layer output
            #print(f"shape of out={out.shape}")

            for edge_conv in self.edge_convs:
                out = edge_conv(out, edge_index)  # EdgeConv expects (num_nodes, num_features)
                #print(f"Shape after edge_conv {i}: {out.shape}")  # Debug print
                features.append(out)

            # Concatenate features from all layers
            out = torch.cat(features, dim=-1)  # Concatenate along the feature dimension
            out = self.dropout(out)  # Apply dropout
            features_tensor.append(out)  # Store the features for each time frame

        # Stack the features for all time frames (Shape: [num_frames, num_nodes, feature_dim])
        out = torch.stack(features_tensor, dim=0)  # Stack along the time dimension
        #print(f"Shape after stacking time frames: {out.shape}")  # Debug print

        # Apply pooling across time frames (mean over time)
        out = out.mean(dim=0)  # Shape: [num_nodes, feature_dim]
        #print(f"Shape after temporal pooling: {out.shape}")  # Debug print

        # Apply pooling across nodes (mean over nodes)
        out = out.mean(dim=0)  # Shape: [feature_dim]
        #print(f"Shape after spatial pooling: {out.shape}")  # Debug print

        # Fully connected layer for classification
        out = self.fc(out)  # Shape: [num_classes]
        #print(f"Output shape: {out.shape}")  # Debug print

        return out

# %%
import pickle
with open("abide_data.pkl","rb") as file:
    abide_dataset=pickle.load(file)


abide_fmri=abide_dataset["rois_cc200"]
abide_labels=abide_dataset["phenotypic"]["DX_GROUP"].to_numpy()-1

# %%


data=[]
num_rois=200
for i in range(len(abide_fmri)):
    x=abide_fmri[i].T
    edge_index=get_knn_edge_index(x, k=5)
    x=x.T.reshape(-1,num_rois,1)
    data.append((torch.tensor(x,dtype=torch.float,requires_grad=True), torch.tensor(edge_index), torch.tensor(abide_labels[i])))


from sklearn.model_selection import train_test_split
train_data, test_data=train_test_split(data, test_size=0.3, random_state=42, stratify=abide_labels )

del data


# %%
import random
import torch_cluster
# Model parameters
n_epochs = 60
num_rois = 200
#batch_size = 1
kernels = [8, 16,32,64,128]

# Initialize the model, optimizer, and scheduler
model = TemporalGraphNetwork(kernels).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

loss= nn.CrossEntropyLoss()

pocket_model=TemporalGraphNetwork(kernels).to("cuda")
min_loss=1000

# Training loop
model.train()
for epoch in range(n_epochs):
    loss_list = []
    random.shuffle(train_data)

    for x, edge_index, y in train_data:
        # Prepare the target labels as floats for BCEWithLogitsLoss

        # Forward pass
        output = model(x.to("cuda"), edge_index.to("cuda"))

        # Compute the loss
        l = loss(output, y.to("cuda"))  # Use .squeeze() if output has an extra dimension

        # Backward pass and optimization
        l.backward()
        optimizer.step()
        
        optimizer.zero_grad()

        # Log the loss
        loss_list.append(l.detach().cpu().item())

    # Epoch summary
    loss_list = np.array(loss_list)
    #print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss_list.mean():.3f}")


    if (loss_list.mean()<min_loss):
        min_epoch=epoch
        min_loss=loss_list.mean()
        pocket_model.load_state_dict(model.state_dict())
        #print(f"Pocket model updated with loss: {min_loss:.3f}")


# Save the best model (pocket model)
torch.save(pocket_model.state_dict(), "pocket_model.pth")
np.savetxt("min_loss.txt",np.array([min_loss,min_epoch]))
#print(f"Training complete. Best model saved with mean loss: {min_loss:.3f}")





