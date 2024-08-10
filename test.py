from dataset import RelationsDS, select_by_edge_type, add_self_loops, split_data_stratified
from model import NodeEmbedder, LinkPredictor, Model

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os.path

def test(model, data, batch_size=None, return_links=False):
    model.eval()
    with torch.no_grad():
        if batch_size is None:
            target_links = torch.cat([data.pos_samples[:,data.test_mask],
                                      data.neg_samples[:,data.test_mask]], dim=1)
            labels = torch.cat([data.y[data.test_mask], 
                                torch.zeros_like(data.y[data.test_mask])])
            preds = model(data.x, 
                          data.edge_index, 
                          data.edge_attr,
                          target_links) 
            all_preds, all_labels = preds.to('cpu'), labels.to('cpu')
            if return_links:
                all_links = target_links.to('cpu')
        else:
            all_preds, all_labels = None, None
            link_loader = DataLoader(data.test_mask,
                                    batch_size=batch_size,
                                    shuffle=False)
            for link_idxs in link_loader:
                target_links = torch.cat([data.pos_samples[:,link_idxs],
                                          data.neg_samples[:,link_idxs]], dim=1)
                labels = torch.cat([data.y[link_idxs], 
                                    torch.zeros_like(data.y[link_idxs])])
                preds = model(data.x, 
                              data.edge_index, 
                              data.edge_attr,
                              target_links) 
                if all_preds is None:
                    all_preds = preds.to('cpu')
                    all_labels = labels.to('cpu')
                    if return_links:
                        all_links = target_links.to('cpu')
                else:
                    all_preds = torch.concat([all_preds, preds.to('cpu')])
                    all_labels = torch.concat([all_labels, labels.to('cpu')])
                    if return_links:
                        all_links = torch.concat([all_links, target_links.to('cpu')])
        if return_links:
            return all_labels.numpy(), all_preds.numpy(), all_links.numpy()
        else:
            return all_labels.numpy(), all_preds.numpy()

if __name__ == '__main__':
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default=None)
    parser.add_argument('--save_links', '-s', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = RelationsDS(root='./data').to(device)
    g = data[0]

    node_embedder = NodeEmbedder(input_dim=768,
                                 hidden_dim=768,
                                 output_dim=512,
                                 num_heads=1,
                                 num_layers=1, 
                                 dropout=0.5)
    link_pred = LinkPredictor(input_dim=512, 
                              hidden_dim=[256, 128, 64],
                              output_dim=3,
                              num_layers=4, 
                              dropout=0.5)
    model = Model(node_embedder, link_pred)
    model.load_state_dict(torch.load('.best_model.pth'))    
    model = model.to(device)

    encoding_path = './data/input_encoding.pt'
    node_encodings = node_embedder.load_input_encodings(encoding_path).to(device)
    print('Loaded pre-computed BERT encodings.')
    g.x = node_encodings 
    g.edge_attr = F.one_hot(g.edge_attr).float()

    eval_batch_size = 150000
    experiment_name = args.name if not args.name is None else ''

    if args.save_links:
        y_true, y_pred, links = test(model, g, batch_size=eval_batch_size, return_links=True)
        results = {'y_true': y_true, 
                   'y_pred_neg': y_pred[:,0],
                   'y_pred_hyp': y_pred[:,1],
                   'y_pred_hol': y_pred[:,2], 
                   'edge_i': links[0,:], 
                   'edge_j': links[1,:]}
    else:
        y_true, y_pred = test(model, g, batch_size=eval_batch_size)
        results = {'y_true': y_true, 
                   'y_pred_neg': y_pred[:,0],
                   'y_pred_hyp': y_pred[:,1],
                   'y_pred_hol': y_pred[:,2]}
    results = pd.DataFrame(results)
    results.to_csv(f'{experiment_name}-results.csv')


