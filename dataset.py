import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import structured_negative_sampling
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

import pandas as pd 
import numpy as np

import os.path 
import re

class WN18RR(InMemoryDataset):
    edge2id = {
        '_hypernym': 0,
        '_derivationally_related_form': 1,
        '_instance_hypernym': 2,
        '_also_see': 3,
        '_member_meronym': 4,
        '_synset_domain_topic_of': 5,
        '_has_part': 6,
        '_member_of_domain_usage': 7,
        '_member_of_domain_region': 8,
        '_verb_group': 9,
        '_similar_to': 10
    }
    def __init__(self, root, split='train', tokenizer=None, transform=None, pre_transform=None, pre_filter=None):
        self._split = self._split_map(split)
        self.root = root
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[self._split])

    def process(self):
        if self._already_processed():
            return

        definitions_file = os.path.join(self.root, 'definitions.txt')

        definitions = pd.read_csv(
            definitions_file, sep='\t', 
            header=None, 
            index_col=0,
            names=['id', 'definition']
        )
        entity_mapping = {name : i for i, name in enumerate(definitions.index)}

        if os.path.exists(os.path.join(self.root, 'tokens.pt')):
            tokens = torch.load(os.path.join(self.root, 'tokens.pt'))
        else:
            augmented_definitions = self._compose_definitions(definitions)
            tokens = self.tokenizer(
                list(augmented_definitions),
                padding="max_length", 
                return_tensors='pt', 
                add_special_tokens=True
            )

        relations_file = os.path.join(self.root, self.raw_file_names[self._split])
        relations = pd.read_csv(
            relations_file, 
            sep="\t", 
            header=None, 
            names=["id", "relation", "related_id"]
        )
        edge_index = torch.tensor(
            relations[['id', 'related_id']].map(entity_mapping.get).to_numpy().T, dtype=torch.long
        )
        edge_attr = torch.tensor(
            relations['relation'].map(self.edge2id.get).to_numpy(), dtype=torch.long
        ).unsqueeze(1)
        
        data = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            token_ids=tokens.input_ids,
            token_mask=tokens.attention_mask,
            token_type_ids=tokens.token_type_ids
        )
        self.save([data], self.processed_paths[self._split])

    def _already_processed(self):
        path = os.path.join(self.root, self.processed_file_names[self._split])
        return os.path.exists(path)

    def _compose_definitions(self, defs):
        pos_tags = {
            'NN': 'noun',
            'VB': 'verb',
            'JJ': 'adjective',
            'RB': 'adverb'
        }

        syn_parser = re.compile(r'__(.+)_([A-Z][A-Z])_(\d+)')

        def augment_definition(synset):
            id, definition = synset.loc["id"], synset.loc["definition"]
            try:
                syn, pos, _ = syn_parser.match(id).groups()
            except:
                print(id)
                return ''
            syn = re.sub(r'_', ' ', syn)
            pos = pos_tags[pos]
            return f'the definition of the {pos} {syn} is {definition}'

        return defs.apply(augment_definition, axis=1)
    
    
    def _split_map(self, split):
        if split == 'train':
            return 0
        elif split == 'validation':
            return 1
        elif split == 'test':
            return 2
        else:
            raise ValueError('Invalid split name')

    @property
    def raw_file_names(self):
        return ['train.txt', 'valid.txt', 'test.txt']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'valid.pt', 'test.pt']
    
    def download(self):
        # Download to `self.raw_dir`.
        return

    


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

        device = 'cuda'
        data = Data().to(device)
        data.num_nodes = ids.shape[0]
        data.edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
        data.edge_attr = torch.tensor(edge_attr, dtype=torch.long, device=device)
        data.token_ids = token_ids.to(device)
        data.token_mask = token_mask.to(device)
        data.token_type_ids = token_type_ids.to(device)

        data = self._create_splits(data, device)
        data = self._add_self_loops(data, data.num_nodes, device)

        return [data.to('cpu')]
    
    def _create_splits(self, data, device):
        src, tar, neg = structured_negative_sampling(data.edge_index,
                                                     contains_neg_self_loops=False)
        data.pos_samples = torch.stack([src, tar])
        data.neg_samples = torch.stack([src, neg])
    
        mask = torch.arange(src.shape[0], device='cpu')
        labels = data.edge_attr.to('cpu')
        train_mask, test_mask = train_test_split(mask, test_size=0.2, stratify=labels)
        train_mask, val_mask = train_test_split(train_mask, test_size=0.2, stratify=labels[train_mask])
        data.train_mask = train_mask.to(device)
        data.val_mask = val_mask.to(device)
        data.test_mask = test_mask.to(device)
        data.y = labels.to(device)
        data.edge_index = data.edge_index[:,data.train_mask]
        data.edge_attr = data.edge_attr[data.train_mask]
        return data
    
    def _add_self_loops(self, data, num_nodes, device):
        edge_index_i = torch.cat([data.edge_index[0], torch.arange(num_nodes, dtype=torch.int64, device=device)]) 
        edge_index_j = torch.cat([data.edge_index[1], torch.arange(num_nodes, dtype=torch.int64, device=device)]) 
        data.edge_index = torch.stack([edge_index_i, edge_index_j])
        
        data.edge_attr = torch.cat([data.edge_attr, torch.zeros(num_nodes, dtype=torch.long, device=device)])
        
        return data

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

    