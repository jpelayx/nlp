from dataset import WN18RR
from model import NodeEmbedder, LinkPredictor, Model

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import structured_negative_sampling, remove_self_loops


def train(model, optimizer, g, targets, batch_size=None, scheduler=None):
    model.train()
    optimizer.zero_grad()

    batch_size = batch_size if batch_size else 1
    link_loader = DataLoader(targets, batch_size=batch_size, shuffle=True)

    total_loss = 0
    for edges, attrs, labels in link_loader:
        preds = model(g.x, g.edge_index, g.edge_attr, edges.T, attrs)
        loss = F.binary_cross_entropy_with_logits(preds.flatten(), labels)
        loss.backward()
        optimizer.step()
        total_loss += torch.sum(loss)
    if scheduler:
        scheduler.step()
    return total_loss / len(targets)


def get_train_targets(g, device, load=None, save=None):
    target_edges, target_attrs = remove_self_loops(g.edge_index, g.edge_attr)
    target_labels = torch.ones(target_edges.shape[1], dtype=torch.float)

    if load:
        negative_edges, negative_attrs = torch.load(load)
        target_edges = torch.cat([target_edges, negative_edges], dim=1)
        target_attrs = torch.cat([target_attrs, negative_attrs], dim=0)
        target_labels = torch.cat([target_labels, torch.zeros_like(target_labels)])
        targets = TensorDataset(
            target_edges.T.to(device), target_attrs.to(device), target_labels.to(device)
        )
        return targets

    relation_mask = torch.argmax(target_attrs, dim=1)
    for i in relation_mask.unique():
        mask = relation_mask == i
        target_edges_i = target_edges[:, mask]
        target_attrs_i = target_attrs[mask]
        src, _, obj = structured_negative_sampling(target_edges_i, g.num_nodes)
        negative_edges = torch.stack([src, obj])

        target_edges = torch.cat([target_edges, negative_edges], dim=1)
        target_attrs = torch.cat([target_attrs, target_attrs_i.repeat(2, 1)], dim=0)
        target_labels = torch.cat([target_labels, torch.zeros_like(target_labels)])

    targets = TensorDataset(
        target_edges.T.to(device), target_attrs.to(device), target_labels.to(device)
    )
    if save:
        torch.save((target_edges, target_attrs), save)
    return targets


if __name__ == "__main__":
    import time
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_embedder = NodeEmbedder(
        input_dim=768,
        hidden_dim=256,
        output_dim=256,
        edge_dim=12,
        num_heads=1,
        num_layers=2,
    ).to(device)

    link_pred = LinkPredictor(
        input_dim=256, hidden_dim=128, output_dim=1, edge_dim=12, num_layers=3
    ).to(device)

    model = Model(node_embedder, link_pred)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    g = WN18RR(root="data/wn18rr", split="train")[0].to(device)
    g.edge_attr = F.one_hot(g.edge_attr, 12).float().squeeze(1)
    train_targets = get_train_targets(g, device, save="data/wn18rr/train_targets.pt")

    results = {
        "epoch": [],
        "loss": [],
        "time": [],
    }
    best_loss = 1
    for epoch in range(100):
        t0 = time.time()
        loss = train(model, optimizer, g, train_targets, batch_size=1024)
        tf = time.time()

        if loss < best_loss:
            torch.save(model.state_dict(), "best_model_weights.pt")
            best_loss = loss

        results["epoch"].append(epoch)
        results["loss"].append(loss.item())
        results["time"].append(tf - t0)
        print(f"Epoch {epoch} - Loss: {loss} ({tf - t0:.2f}s)")

    pd.DataFrame(results).to_csv("results.csv", index=False)
    print("Done")
