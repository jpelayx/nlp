import torch
from torch_geometric.data import InMemoryDataset, Data
from transformers import BertTokenizer

import pandas as pd 
import numpy as np

import os.path 

class RelationsDS(InMemoryDataset):
    edge2id = {
        'Hypernyms': 0,
        'Holonyms': 1
    }

    def __init__(self, root, tokenizer=None, transform=None, pre_transform=None, pre_filter=None):
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
        
        super().__init__(root, transform, pre_transform, pre_filter)

        self.load(self.processed_paths[0])

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
        definitions = entities['Definição'].to_numpy()
        mapping = {name : i for i, name in enumerate(ids)}

        edge_index_i =  df['ID_Synset'].map(mapping).to_numpy()
        edge_index_j =  df['ID_Relacionada'].map(mapping).to_numpy()
        edge_index = np.stack([edge_index_i, edge_index_j])

        edge_type = df['Relacao'].map(self.edge2id).to_numpy()

        tokens = self.tokenizer(list(definitions),
                                padding=True, 
                                return_tensors='pt')
        token_ids = tokens.input_ids
        token_mask = tokens.attention_mask
        token_type_ids = tokens.token_type_ids 

        data = Data()
        data.edge_index = torch.tensor(edge_index, dtype=torch.long)
        data.edge_type = torch.tensor(edge_type)
        data.token_ids = token_ids
        data.token_mask = token_mask
        data.token_type_ids = token_type_ids

        return [data]

def select_by_edge_type(edge_type:str, data:RelationsDS) -> Data:
    edge2id = data.edge2id
    data = data[0]

    edge_index = data.edge_index[:, data.edge_type == edge2id['edge_type']]

    nodes_index = torch.concat([edge_index[0,:], edge_index[1,:]]).unique()
    token_ids = data.token_ids[nodes_index]
    token_mask = data.token_mask[nodes_index]
    token_type_ids = data.token_type_ids[nodes_index]

    new_data = Data()
    new_data.edge_index = edge_index
    new_data.token_ids = token_ids
    new_data.token_mask = token_mask
    new_data.token_type_ids = token_type_ids

    return new_data 