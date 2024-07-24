from dataset import RelationsDS, select_by_edge_type
from model import Model, LinkPredictor

import torch
import torch.nn.functional as F
from torch_geometric.utils import structured_negative_sampling
from sklearn.metrics import f1_score

import os.path

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


def train(model, link_pred, optimizer, data):
   model.train()
   link_pred.train()
   optimizer.zero_grad()
  
   embs = model(x=data.x, edge_index=data.pos_train_edge_index)

   pos_pred = link_pred(embs[data.pos_train_edge_index[0]],
                        embs[data.pos_train_edge_index[1]])
   neg_pred = link_pred(embs[data.neg_train_edge_index[0]],
                        embs[data.neg_train_edge_index[1]])
   pred = torch.concat([pos_pred, neg_pred])
   labels = torch.concat([torch.ones(pos_pred.shape), torch.zeros(neg_pred.shape)])

   loss = F.binary_cross_entropy_with_logits(pred, labels)
   loss.backward()
   optimizer.step()

   return loss.item()

def val(model, link_pred, data):
   model.eval()
   link_pred.eval()
   embs = model(x=data.x, edge_index=data.pos_train_edge_index)
   pos_pred = link_pred(embs[data.pos_train_edge_index[0]],
                        embs[data.pos_train_edge_index[1]])
   neg_pred = link_pred(embs[data.neg_train_edge_index[0]],
                        embs[data.neg_train_edge_index[1]])
   pred = torch.concat([pos_pred, neg_pred])
   labels = torch.concat([torch.ones(pos_pred.shape), torch.zeros(neg_pred.shape)])
   loss = F.binary_cross_entropy_with_logits(pred, labels)

   return loss.item()

def test(model, link_pred, data):
   model.eval()
   link_pred.eval()
   embs = model(x=data.x, edge_index=data.pos_train_edge_index)
   pos_pred = link_pred(embs[data.pos_train_edge_index[0]],
                        embs[data.pos_train_edge_index[1]])
   neg_pred = link_pred(embs[data.neg_train_edge_index[0]],
                        embs[data.neg_train_edge_index[1]])
   pred = torch.concat([pos_pred, neg_pred])
   labels = torch.concat([torch.ones(pos_pred.shape), torch.zeros(neg_pred.shape)])
   loss = F.binary_cross_entropy_with_logits(pred, labels)

   return loss.item(), f1_score(labels, pred)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = RelationsDS(root='./data').to(device)
    g = select_by_edge_type('Hypernyms', data)

    model = Model(input_dim=768,
                  hidden_dim=256,
                  output_dim=256,
                  num_layers=2).to(device)
    link_pred = LinkPredictor(input_dim=256, 
                              hidden_dim=128,
                              output_dim=1,
                              num_layers=3).to(device)

    encoding_path = './data/input_encoding.pt'
    if not os.path.exists(encoding_path):
        print("Encoding input...")
        tokens = torch.stack([g.token_ids, g.token_mask, g.token_type_ids], dim=1)
        with torch.no_grad():
            node_encodings = model.encode_inputs(tokens) 
        model.save_input_encodings(node_encodings)
    else:
        node_encodings = model.load_input_encodings()
    
    g = split_data(g)

    train_epochs = 10 
    train_losses = []
    val_losses = []
    for epoch in range(train_epochs):
        train_losses.append(train(model, link_pred, g))
        val_losses.append(val(model, link_pred, g))
        print(f'Epoch {epoch+1}/{train_epochs}, train loss: {train_losses[-1]}, val loss: {val_losses[-1]}')
    test_loss, test_f1 = test(model, link_pred, g)
    print(f'Test loss: {test_loss}, test F1: {test_f1}')



