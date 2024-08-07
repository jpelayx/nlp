import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import structured_negative_sampling
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

import pandas as pd 
import numpy as np

import os.path 
import re

class RelationsDS(InMemoryDataset):
    edge2id = {
        'Hypernyms': 1,
        'Holonyms': 2
    }

    def __init__(self, root, tokenizer=None, transform=None, pre_transform=None, pre_filter=None):
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        self.num_nodes = self[0].token_ids.shape[0]

    @property
    def raw_file_names(self):
        return ['direct_relations.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        return 

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.process_csv()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
    
    def process_csv(self):
        path = os.path.join(self.root, self.raw_file_names[0])
        df = pd.read_csv(path)

        all_ids = pd.concat([df['ID_Synset'], df['ID_Relacionada']])
        all_definitions = pd.concat([df['Definição_Synset'], df['Definição_Relacionada']])

        entities = pd.DataFrame({'ID': all_ids, 
                                 'Definição': all_definitions}).drop_duplicates()
       
        ids = entities['ID'].to_numpy()
        definitions = entities.apply(self._compose_definition, axis=1)
        mapping = {name : i for i, name in enumerate(ids)}

        edge_index_i =  df['ID_Synset'].map(mapping).to_numpy()
        edge_index_j =  df['ID_Relacionada'].map(mapping).to_numpy()
        edge_index = np.stack([edge_index_i, edge_index_j])

        edge_attr = df['Relacao'].map(self.edge2id).to_numpy()

        tokens = self.tokenizer(list(definitions),
                                padding=True, 
                                return_tensors='pt', 
                                add_special_tokens=True)
        token_ids = tokens.input_ids
        token_mask = tokens.attention_mask
        token_type_ids = tokens.token_type_ids 

        data = Data()
        data.edge_index = torch.tensor(edge_index, dtype=torch.long)
        data.edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        data.token_ids = token_ids
        data.token_mask = token_mask
        data.token_type_ids = token_type_ids

        return [data]

    def _compose_definition(self, arg):
        id, definition = arg
        clean_id =  re.sub(r'\..\.[0-9][0-9]', '', id)
        clean_id =  re.sub(r'_', ' ', clean_id)

        new_def = f'the definition of {clean_id} is {definition}'
        return new_def

def select_by_edge_type(edge_type:str, data:RelationsDS) -> Data:
    edge2id = data.edge2id
    data = data[0]

    edge_index = data.edge_index[:, data.edge_type == edge2id[edge_type]]

    # nodes_index = torch.concat([edge_index[0,:], edge_index[1,:]]).unique()
    token_ids = data.token_ids #[nodes_index]
    token_mask = data.token_mask #[nodes_index]
    token_type_ids = data.token_type_ids #[nodes_index]

    new_data = Data()
    new_data.edge_index = edge_index
    new_data.token_ids = token_ids
    new_data.token_mask = token_mask
    new_data.token_type_ids = token_type_ids

    return new_data 

def add_self_loops(data:Data, num_nodes:int) -> Data:
    device = data.edge_attr.device
    edge_index_i, edge_index_j = data.edge_index
    edge_index_i = torch.cat([edge_index_i, torch.arange(0, num_nodes, dtype=torch.int64, device=device)]) 
    edge_index_j = torch.cat([edge_index_j, torch.arange(0, num_nodes, dtype=torch.int64, device=device)]) 
    edge_index = torch.stack([edge_index_i, edge_index_j])

    edge_attr = data.edge_attr
    edge_attr = torch.cat([edge_attr, torch.zeros(num_nodes, device=device)])
    # edge2id['ID'] == 0

    data.edge_index = edge_index
    data.edge_attr = edge_attr

    return data 

def split_data_stratified(g, num_nodes, add_self_loops=True):
    device = g.edge_index.device
    src, tar, neg = structured_negative_sampling(g.edge_index,
                                                 contains_neg_self_loops=False)
    g.pos_samples = torch.stack([src, tar])
    g.neg_samples = torch.stack([src, neg])
    
    mask = torch.arange(src.shape[0])
    labels = g.edge_attr.to('cpu')
    train_mask, test_mask = train_test_split(mask, test_size=0.2, stratify=labels)
    train_mask, val_mask = train_test_split(train_mask, test_size=0.2, stratify=labels[train_mask])
    g.train_mask = train_mask.to(device)
    g.val_mask = val_mask.to(device)
    g.test_mask = test_mask.to(device)
    g.y = labels.to(device)

    if add_self_loops:    
        edge_index_i, edge_index_j = src[train_mask], tar[train_mask]
        edge_index_i = torch.cat([edge_index_i, torch.arange(num_nodes, dtype=torch.int64, device=device)]) 
        edge_index_j = torch.cat([edge_index_j, torch.arange(num_nodes, dtype=torch.int64, device=device)]) 
        edge_index = torch.stack([edge_index_i, edge_index_j])
        
        edge_attr = g.edge_attr[train_mask]
        edge_attr = torch.cat([edge_attr, torch.zeros(num_nodes, dtype=torch.long, device=device)])
    else:
        edge_index = g.edge_index[:,train_mask]
        edge_attr  = g.edge_attr[train_mask]
    
    g.edge_index = edge_index
    g.edge_attr  = edge_attr

    return g

    