from dataset import RelationsDS, select_by_edge_type, add_self_loops, split_data_stratified
from model import NodeEmbedder, LinkPredictor, Model

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import binarize

import os.path

LABEL_TRESHOLD = 0.5


def train(model, optimizer, data, batch_size=None, scheduler=None):
    model.train()
    optimizer.zero_grad()
    if batch_size is None:
        pred, labels = model(data.x, 
                             data.edge_index, 
                             data.edge_attr,
                             data.pos_samples[data.train_mask], 
                             data.neg_samples[data.train_mask]) 
        loss = F.binary_cross_entropy_with_logits(pred, labels)
        loss.backward()    
        optimizer.step()
        return loss.item()
    else:
        losses = None
        link_loader = DataLoader(data.train_mask,
                                 batch_size=batch_size,
                                 shuffle=True)
        for link_idxs in link_loader:
            pred, labels = model(data.x, 
                                 data.edge_index,
                                 data.edge_attr,
                                 data.pos_samples[:,link_idxs], 
                                 data.neg_samples[:,link_idxs])
            loss = F.binary_cross_entropy_with_logits(pred, labels)
            loss.backward()
            optimizer.step()
            if losses is None:
                losses = torch.tensor([loss.item()])
            else:
                losses = torch.concat([losses, torch.tensor([loss.item()])])
        if not scheduler is None:
            scheduler.step()
        return losses.mean().item()

def val(model, data, batch_size=None):
    model.eval()
    with torch.no_grad():
        if batch_size is None:
            pred, labels = model(data.x, 
                                data.edge_index,
                                data.edge_attr, 
                                data.pos_samples[data.val_mask], 
                                data.neg_samples[data.val_mask])
            all_preds, all_labels = pred.to('cpu'), labels.to('cpu')
            mean_loss = F.binary_cross_entropy_with_logits(pred, labels).item()
        else:
            losses = None
            all_preds, all_labels = None, None
            link_loader = DataLoader(data.val_mask,
                                     batch_size=batch_size,
                                     shuffle=False)
            for link_idxs in link_loader:
                pred, labels = model(data.x, 
                                     data.edge_index,
                                     data.edge_attr,
                                     data.pos_samples[:,link_idxs], 
                                     data.neg_samples[:,link_idxs])
                loss = F.binary_cross_entropy_with_logits(pred, labels)
                if losses is None:
                    losses = torch.tensor([loss.item()])
                    all_preds = pred.to('cpu')
                    all_labels = labels.to('cpu')
                else:
                    losses = torch.concat([losses, torch.tensor([loss.item()])])
                    all_preds = torch.concat([all_preds, pred.to('cpu')])
                    all_labels = torch.concat([all_labels, labels.to('cpu')])
            mean_loss = losses.mean().item()

        all_preds = binarize(all_preds, threshold=LABEL_TRESHOLD)
        return (mean_loss, 
                precision_score(all_labels, all_preds, zero_division=0.0), 
                recall_score(all_labels, all_preds, zero_division=0.0),
                f1_score(all_labels, all_preds, zero_division=0.0))

def test(model, data, batch_size=None):
    model.eval()
    with torch.no_grad():
        if batch_size is None:
            pred, labels = model(data.x, 
                                data.edge_index,
                                data.edge_attr, 
                                data.pos_samples[data.test_mask], 
                                data.neg_samples[data.test_mask])
            all_preds, all_labels = pred.to('cpu'), labels.to('cpu')
        else:
            all_preds, all_labels = None, None
            link_loader = DataLoader(data.test_mask,
                                    batch_size=batch_size,
                                    shuffle=False)
            for link_idxs in link_loader:
                pred, labels = model(data.x, 
                                     data.edge_index,
                                     data.edge_attr,
                                     data.pos_samples[:,link_idxs], 
                                     data.neg_samples[:,link_idxs])
                if all_preds is None:
                    all_preds = pred.to('cpu')
                    all_labels = labels.to('cpu')
                else:
                    all_preds = torch.concat([all_preds, pred.to('cpu')])
                    all_labels = torch.concat([all_labels, labels.to('cpu')])
        
        return all_labels.numpy().flatten(), all_preds.numpy().flatten()

if __name__ == '__main__':
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=None)
    parser.add_argument('--epochs', '-e', type=int, default=500)
    parser.add_argument('--name', '-n', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = RelationsDS(root='./data').to(device)
    g = data[0]

    node_embedder = NodeEmbedder(input_dim=768,
                                 hidden_dim=256,
                                 output_dim=256,
                                 num_heads=1,
                                 num_layers=2).to(device)
    link_pred = LinkPredictor(input_dim=256, 
                              hidden_dim=128,
                              output_dim=1,
                              num_layers=3).to(device)
    model = Model(node_embedder, link_pred)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1) 
    # scheduler = None

    encoding_path = './data/input_encoding.pt'
    if not os.path.exists(encoding_path):
        print("Encoding input...")
        tokens = torch.stack([g.token_ids, g.token_mask, g.token_type_ids], dim=1)
        with torch.no_grad():
            node_encodings = node_embedder.encode_inputs(tokens, batch_size=64, verbose=True) 
        node_embedder.save_input_encodings(node_encodings)
    else:
        node_encodings = node_embedder.load_input_encodings(encoding_path).to(device)
        print('Loaded pre-computed BERT encodings.')
    g.x = node_encodings 
    
    g = split_data_stratified(g, data.num_nodes) # create neg samples and train/val/test splits 
    g.edge_attr = F.one_hot(g.edge_attr).float()

    train_epochs = args.epochs
    batch_size = args.batch_size
    eval_batch_size = 150000
    max_f1 = 0
    train_losses = {'epoch':[],'train loss':[], 'val loss':[]}
    experiment_name = args.name if not args.name is None else ''
    for epoch in range(train_epochs):
        train_loss = train(model, optimizer, g, batch_size=batch_size, scheduler=scheduler)
        val_loss, val_precision, val_recall, val_f1 = val(model, g, batch_size=eval_batch_size)
        train_losses['epoch'].append(epoch)
        train_losses['train loss'].append(train_loss)
        train_losses['val loss'].append(val_loss)
        if val_f1 > max_f1:
            max_f1 = val_f1
            torch.save(model.state_dict(), '.best_model.pth')
        print(f'{epoch+1},{train_loss},{val_loss},{val_precision},{val_recall}')

    best_model = Model(node_embedder, link_pred)
    best_model.load_state_dict(torch.load('.best_model.pth'))    
    best_model = best_model.to(device)
    y_true, y_pred = test(best_model, g, batch_size=eval_batch_size)
    results = pd.DataFrame({'y_true': y_true, 'y_pred':y_pred})
    results.to_csv(f'{experiment_name}-results.csv')
    losses = pd.DataFrame(train_losses)
    losses.to_csv(f'{experiment_name}-train_losses.csv')

    # print(f'Test loss: {test_loss}, test precision: {test_precision}, test recall: {test_recall}')



