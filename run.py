from dataset import WN18RR
from model import NodeEmbedder, LinkPredictor, Model

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import structured_negative_sampling, remove_self_loops

import time
import pandas as pd

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


def train_loop(
    model, optimizer, g, targets, epochs, batch_size=None, scheduler=None, save=None
):
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

        if loss < best_loss and save:
            torch.save(model.state_dict(), save)
            best_loss = loss

        results["epoch"].append(epoch)
        results["loss"].append(loss.item())
        results["time"].append(tf - t0)
        print(f"Epoch {epoch} - Loss: {loss} ({tf - t0:.2f}s)")

    pd.DataFrame(results).to_csv("train_results.csv", index=False)
    return torch.load(save)


def get_train_targets(g, device, load=None, save=None):
    target_edges, target_attrs = remove_self_loops(g.edge_index, g.edge_attr)
    target_labels = torch.ones(target_edges.shape[1], dtype=torch.float)

    if load:
        targets = torch.load(load, map_location=device)
        return targets

    relation_mask = torch.argmax(target_attrs, dim=1)
    negative_edges = []
    negative_attrs = []
    for i in relation_mask.unique():
        mask = relation_mask == i
        target_edges_i = target_edges[:, mask]
        target_attrs_i = target_attrs[mask]

        src, _, obj = structured_negative_sampling(target_edges_i, g.num_nodes)
        negative_edges.append(torch.stack([src, obj], dim=0))
        negative_attrs.append(target_attrs_i)

    negative_edges = torch.cat(negative_edges, dim=1)
    negative_attrs = torch.cat(negative_attrs, dim=0)

    target_edges = torch.cat([target_edges, negative_edges], dim=1)
    target_attrs = torch.cat([target_attrs, negative_attrs], dim=0)
    target_labels = torch.cat([target_labels, torch.zeros_like(target_labels)])

    targets = TensorDataset(
        target_edges.T.to(device), target_attrs.to(device), target_labels.to(device)
    )

    if save:
        torch.save((target_edges, target_attrs), save)
    return targets


def get_validation_data(g, filter_data=None, save=None, load=None):
    if load:
        validation_data = torch.load(load)
        return validation_data

    validation_data = TensorDataset(g.edge_index.T, g.edge_attr)
    if filter_data:
        validation_data = filter_edges(validation_data, filter_data)

    if save:
        torch.save(validation_data, save)
    return validation_data


def filter_edges(validation_data, train_data):
    train_edges, train_attrs  = train_data
    mask = torch.ones(len(validation_data), dtype=torch.bool)
    for idx, (edge, attr) in enumerate(validation_data):
        repeated_edges = torch.isin(train_edges, edge)
        repeated_edges = torch.logical_and(repeated_edges[:, 0], repeated_edges[:, 1])
        if repeated_edges.any():
            indices = torch.argwhere(repeated_edges).squeeze(1)
            mask[idx] = ~torch.isin(
                torch.argmax(train_attrs[indices], dim=1), torch.argmax(attr)
            ).any()
    return TensorDataset(*validation_data[mask])


def evaluate(model, g, validation_data, batch_size=None):
    model.eval()

    with torch.no_grad():
        node_embeddings = model.node_embedder(g.x, g.edge_index, g.edge_attr)

    batch_size = batch_size if batch_size else 1
    ranks = []
    eval_time = 0
    for edge, attr in validation_data:
        preds = []
        t0 = time.time()
        with torch.no_grad():
            entity_loader = DataLoader(
                node_embeddings, batch_size=batch_size, shuffle=False
            )
            for xo in entity_loader:
                current_batch_size = xo.shape[0]
                xs = node_embeddings[edge[0]].repeat(current_batch_size, 1)
                xr = attr.repeat(current_batch_size, 1)
                preds.append(model.link_predictor(xs, xr, xo).to("cpu"))
        tf = time.time()
        preds = torch.cat(preds)
        ranks.append(torch.sum(preds > preds[edge[1]]).item())
        eval_time += tf - t0 
    print(f"Total evaluation time: {eval_time:.2f}s ({eval_time/len(ranks):.2f} s/item)")    
    return ranks


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", action="store_true")
    argparser.add_argument("--validate", action="store_true")
    argparser.add_argument("--load", type=str, default=None)
    argparser.add_argument("--save", type=str, default=None)
    argparser.add_argument("--epochs", type=int, default=100)
    args = argparser.parse_args()

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

    if args.load:
        model.load_state_dict(torch.load(args.load))

    g = WN18RR(root="data/wn18rr", split="train", verbose_processing=True)[0].to(device)
    g.edge_attr = F.one_hot(g.edge_attr, 12).float().squeeze(1)
    train_targets = get_train_targets(g, device, load="data/wn18rr/train_targets.pt")

    if args.train:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        model_weights = train_loop(
            model,
            optimizer,
            g,
            train_targets,
            epochs=args.epochs,
            save=args.save,
        )
        model.load_state_dict(model_weights)

    if args.validate:
        g_val = WN18RR(root="data/wn18rr", split="validation", verbose_processing=True)[0].to(device)
        g_val.edge_attr = F.one_hot(g_val.edge_attr, 12).float().squeeze(1)
        validation_data = get_validation_data(
            g_val, filter_data=train_targets, save="data/wn18rr/validation_data.pt"
        )
        ranks = evaluate(model, g, validation_data, batch_size=2048)
        mean_rank = sum(ranks) / len(ranks)
        hits_at_1 = sum(rank <= 1 for rank in ranks) / len(ranks)
        hits_at_3 = sum(rank <= 3 for rank in ranks) / len(ranks)
        hits_at_10 = sum(rank <= 10 for rank in ranks) / len(ranks)
        print(
            f"Mean Rank: {mean_rank}\nHits@1: {hits_at_1}\nHits@3: {hits_at_3}\nHits@10: {hits_at_10}"
        )

    print("Done")

