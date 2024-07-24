from dataset import RelationsDS, select_by_edge_type
from model import NodeEmbedder, LinkPredictor, Model

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import structured_negative_sampling
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import binarize

import os.path

LABEL_TRESHOLD = 0.5

def split_data(g):
    idx = torch.randperm(g.edge_index.shape[1])
    g.edge_index = g.edge_index[:,idx]
    src, tar, neg = structured_negative_sampling(g.edge_index)
    train_size = int(g.edge_index.shape[1] * 0.8)
    val_size = int(g.edge_index.shape[1] * 0.1)
    test_size = g.edge_index.shape[1] - train_size - val_size

    g.pos_train_edge_index = torch.stack([src[:train_size], tar[:train_size]])
    g.pos_val_edge_index = torch.stack([src[train_size:train_size+val_size], tar[train_size:train_size+val_size]])
    g.pos_test_edge_index = torch.stack([src[train_size+val_size:], tar[train_size+val_size:]])
    g.neg_train_edge_index = torch.stack([src[:train_size], neg[:train_size]])
    g.neg_val_edge_index = torch.stack([src[train_size:train_size+val_size], neg[train_size:train_size+val_size]])
    g.neg_test_edge_index = torch.stack([src[train_size+val_size:], neg[train_size+val_size:]])
    return g

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    pred, labels = model(data.x, data.pos_train_edge_index, data.neg_train_edge_index)

    loss = F.binary_cross_entropy_with_logits(pred, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def val(model, data):
    model.eval()
    link_pred.eval()
    with torch.no_grad():
        pred, labels = model(data.x, data.pos_val_edge_index, data.neg_val_edge_index)
        loss = F.binary_cross_entropy_with_logits(pred, labels)

        labels = labels.to('cpu')
        pred = binarize(pred.to('cpu'), threshold=LABEL_TRESHOLD)
        return (loss.item(), 
                precision_score(labels, pred, zero_division=0.0), 
                recall_score(labels, pred, zero_division=0.0))

def test(model, data):
    model.eval()
    link_pred.eval()
    with torch.no_grad():
        pred, labels = model(data.x, data.pos_test_edge_index, data.neg_test_edge_index)
        loss = F.binary_cross_entropy_with_logits(pred, labels)

        labels = labels.to('cpu')
        pred = binarize(pred.to('cpu'), threshold=LABEL_TRESHOLD)
        return (loss.item(), 
                precision_score(labels, pred, zero_division=0.0), 
                recall_score(labels, pred, zero_division=0.0))

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = RelationsDS(root='./data').to(device)
    g = select_by_edge_type('Hypernyms', data)

    node_embedder = NodeEmbedder(input_dim=768,
                                 hidden_dim=256,
                                 output_dim=256,
                                 num_layers=2).to(device)
    link_pred = LinkPredictor(input_dim=256, 
                              hidden_dim=128,
                              output_dim=1,
                              num_layers=3).to(device)
    model = Model(node_embedder, link_pred)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)   

    encoding_path = './data/input_encoding.pt'
    if not os.path.exists(encoding_path):
        print("Encoding input...")
        tokens = torch.stack([g.token_ids, g.token_mask, g.token_type_ids], dim=1)
        with torch.no_grad():
            node_encodings = node_embedder.encode_inputs(tokens, verbose=True) 
        node_embedder.save_input_encodings(node_encodings)
    else:
        node_encodings = node_embedder.load_input_encodings(encoding_path).to(device)
        print('Loaded pre-computed BERT encodings.')
    g.x = node_encodings 
    
    g = split_data(g)

    train_epochs = 500 
    num_train_links = g.pos_train_edge_index.shape[1]
    for epoch in range(train_epochs):
        train_loss = train(model, optimizer, g)
        val_loss, val_precision, val_recall = val(model, g)
        # print(f'Epoch {epoch+1}/{train_epochs}, train loss: {train_loss}, val loss: {val_loss}, val f1: {val_f1}')
        print(f'{epoch+1},{train_loss},{val_loss},{val_precision},{val_recall}')
    test_loss, test_precision, test_recall = test(model, g)
    print(f'Test loss: {test_loss}, test precision: {test_precision}, test recall: {test_recall}')



