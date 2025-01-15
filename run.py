from dataset import WN18RR
from model import NodeEmbedder, LinkPredictor, Model

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import structured_negative_sampling, remove_self_loops

import time
import pandas as pd
import numpy as np
import csv
import os


def train(model, optimizer, g, targets, batch_size=None, scheduler=None):
    model.train()
    optimizer.zero_grad()

    batch_size = batch_size if batch_size else 1
    link_loader = DataLoader(targets, batch_size=batch_size, shuffle=True)
    num_batches = len(link_loader)

    total_loss = 0
    for edges, attrs in link_loader:
        current_batch_size = edges.shape[0]
        preds_golden = model(
            g.x, g.edge_index, g.edge_attr, edges.T[[0, 1]], attrs
        ).flatten()
        preds_corrupted_heads = model(
            g.x, g.edge_index, g.edge_attr, edges.T[[2, 1]], attrs
        ).flatten()
        preds_corrupted_tails = model(
            g.x, g.edge_index, g.edge_attr, edges.T[[0, 3]], attrs
        ).flatten()

        preds_golden = preds_golden.repeat(2)
        preds_corrupted = torch.cat([preds_corrupted_heads, preds_corrupted_tails])

        loss = F.margin_ranking_loss(
            preds_golden, preds_corrupted, torch.ones_like(preds_golden), margin=1
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if scheduler:
        scheduler.step()
    return total_loss / num_batches


def validation(model, targets, g, batch_size=10000):
    model.eval()
    link_loader = DataLoader(targets, batch_size=batch_size, shuffle=False)
    num_batches = len(link_loader)
    with torch.no_grad():
        x = model.node_embedder(g.x, g.edge_index, g.edge_attr)
        total_loss = 0
        for edges, attrs in link_loader:
            heads = x[edges[:, 0]]
            tails = x[edges[:, 1]]
            corrupted_heads = x[edges[:, 2]]
            corrupted_tails = x[edges[:, 3]]
            xr = attrs
            preds_golden = model.link_predictor(heads, xr, tails).flatten()
            preds_corrupted_heads = model.link_predictor(
                corrupted_heads, xr, tails
            ).flatten()
            preds_corrupted_tails = model.link_predictor(
                heads, xr, corrupted_tails
            ).flatten()

            preds_golden = preds_golden.repeat(2)
            preds_corrupted = torch.cat([preds_corrupted_heads, preds_corrupted_tails])
            loss = F.margin_ranking_loss(
                preds_golden, preds_corrupted, torch.ones_like(preds_golden), margin=1
            )
            total_loss += loss.item()

    return total_loss / num_batches


def train_loop(
    model,
    optimizer,
    g,
    targets,
    validation_targets,
    epochs,
    batch_size=None,
    scheduler=None,
    save=None,
):
    results = {
        "epoch": None,
        "loss": None,
        "time": None,
        "validation loss": None,
    }
    result_file = f"{save}_train.csv"
    if not os.path.exists(result_file):
        with open(result_file, "w") as f:
            csv_writer = csv.DictWriter(f, results.keys())
            csv_writer.writeheader()

    best_val_loss = 10000
    for epoch in range(epochs):
        t0 = time.time()
        loss = train(model, optimizer, g, targets, batch_size=batch_size)
        tf = time.time()

        val_loss = validation(model, validation_targets, g)

        if val_loss < best_val_loss and save:
            torch.save(model.state_dict(), f"{save}.pt")
            best_val_loss = val_loss

        results["epoch"] = epoch
        results["loss"] = loss
        results["validation loss"] = val_loss
        results["time"] = tf - t0
        with open(result_file, "a") as f:
            csv_writer = csv.DictWriter(f, results.keys())
            csv_writer.writerow(results)
        print(
            f"Epoch {epoch} - Loss: {loss} - Validation loss: {val_loss} ({tf - t0:.2f}s)"
        )
    if save:
        return torch.load(save)
    else:
        return model.state_dict()


def get_train_targets(g, save=None):
    edge_index, edge_attr = remove_self_loops(g.edge_index, g.edge_attr)
    h, t, corrupted_t = structured_negative_sampling(edge_index, g.num_nodes)
    t, h, corrupted_h = structured_negative_sampling(edge_index[[1, 0]], g.num_nodes)
    target_edges = torch.stack([h, t, corrupted_h, corrupted_t], dim=0)
    target_attrs = edge_attr

    targets = TensorDataset(target_edges.T, target_attrs)
    if save:
        torch.save(targets, save)
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
    train_edges, train_attrs, _ = train_data.tensors
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
    link_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    num_entities = node_embeddings.shape[0]
    for i, (edges, attrs) in enumerate(link_loader):
        current_batch_size = edges.shape[0]
        t0 = time.time()
        with torch.no_grad():
            xs = node_embeddings[edges[:, 0]]
            xs = xs.repeat_interleave(num_entities, dim=0)
            xr = attrs.repeat_interleave(num_entities, dim=0)
            xo = node_embeddings.repeat(current_batch_size, 1)
            preds = model.link_predictor(xs, xr, xo).to("cpu").numpy()
        tf = time.time()

        preds = preds.reshape((current_batch_size, -1))
        targets = [
            preds[p, node]
            for p, node in zip(range(current_batch_size), edges[:, 1].to("cpu"))
        ]
        rank = np.apply_along_axis(np.greater, 0, preds, targets)
        rank = np.sum(rank, axis=1)

        ranks.extend(rank)
        eval_time += tf - t0
        if i % 100 == 0:
            print(
                f"Evaluated {(i+1)*batch_size}/{len(validation_data)} - Mean rank: {np.mean(ranks)} ({tf - t0:.2f}s/batch)"
            )
    print(
        f"Total evaluation time: {eval_time:.2f}s ({eval_time/len(ranks):.2f} s/item)"
    )
    return ranks


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", action="store_true")
    argparser.add_argument("--validate", action="store_true")
    argparser.add_argument("--load", type=str, default=None)
    argparser.add_argument("--save", type=str, default=None)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--eb", type=int, default=15)
    argparser.add_argument("--tb", type=int, default=2048)
    argparser.add_argument("--lr", type=float, default=0.0001)

    args = argparser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_embedder = NodeEmbedder(
        input_dim=768,
        hidden_dim=[128, 64],
        output_dim=256,
        edge_dim=5,
        num_heads=[4, 4],
        num_layers=2,
    ).to(device)

    link_pred = LinkPredictor(
        input_dim=256,
        hidden_dim=[256, 124],
        output_dim=1,
        edge_dim=5,
        num_layers=2,
    ).to(device)

    model = Model(node_embedder, link_pred)

    if args.load:
        model.load_state_dict(torch.load(f"{args.load}.pt"))
        print(f"Loaded model {args.load}")

    g = WN18RR(root="data/wn18rr", split="train", verbose_processing=True)[0].to(device)
    train_targets = get_train_targets(g, save="data/wn18rr/train_targets.pt")

    g_val = WN18RR(root="data/wn18rr", split="validation", verbose_processing=True)[
        0
    ].to(device)
    validation_targets = get_train_targets(g_val, save="data/wn18rr/validation_data.pt")

    if args.train:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        model_weights = train_loop(
            model,
            optimizer,
            g,
            train_targets,
            validation_targets=validation_targets,
            epochs=args.epochs,
            save=args.save,
            batch_size=args.tb,
        )
        model.load_state_dict(model_weights)

    if args.validate:

        g_val = WN18RR(root="data/wn18rr", split="test", verbose_processing=True)[0].to(
            device
        )
        validation_data = get_validation_data(g_val, save="data/wn18rr/test_data.pt")
        ranks = evaluate(model, g_val, validation_data, batch_size=args.eb)
        mean_rank = sum(ranks) / len(ranks)
        hits_at_1 = sum(rank <= 1 for rank in ranks) / len(ranks)
        hits_at_3 = sum(rank <= 3 for rank in ranks) / len(ranks)
        hits_at_10 = sum(rank <= 10 for rank in ranks) / len(ranks)
        results = f"Mean Rank: {mean_rank}\nHits@1: {hits_at_1}\nHits@3: {hits_at_3}\nHits@10: {hits_at_10}"
        print(results)
        if args.save:
            with open(f"{args.save}_eval.txt", "a") as f:
                f.write(results)

    print("Done")
