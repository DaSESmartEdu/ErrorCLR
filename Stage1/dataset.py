import abc
import contextlib
import random
import collections
import copy
import torch
from torch_geometric.data import Data
import numpy as np
import networkx as nx


class GraphDataset(object):
    def __init__(self, pair_path, graphs_path, device, task='similarity', epochs=50):
        self.device=device
        self.pair_data = np.load(pair_path, allow_pickle=True)
        self.graphs = np.load(graphs_path, allow_pickle=True).item()
        self.num = len(self.pair_data)
        self.epochs = epochs
        self.task = task


    def pairs(self, batch_size):
        """Yields batches of pair data."""
        for e in range(self.epochs):
            for start_idx in range(0, self.num, batch_size):
                batch_graphs_x = []
                batch_graphs_y = []
                batch_labels = []
                max_node1 = 0
                max_node2 = 0

                for i in range(min(batch_size, self.num-start_idx)):
                    item = self.pair_data[start_idx+i]
                    g1 = self.graphs[item[0]]
                    g2 = self.graphs[item[1]]
                    batch_graphs_x.append(g1)
                    batch_graphs_y.append(g2)

                    batch_labels.append(item[2])

                packed_graphs_x, node_line_length = self._pack_batch(batch_graphs_x)
                packed_graphs_y, _ = self._pack_batch(batch_graphs_y)

                if self.task == 'finetune':
                    label1 = torch.from_numpy(np.array(np.concatenate([i[0] for i in batch_labels]), dtype=np.int32))\
                        .long().to(self.device)
                    label2 = torch.from_numpy(np.array(np.concatenate([i[1] for i in batch_labels]), dtype=np.int32))\
                        .long().to(self.device)
                    labels = [label1, label2]
                elif self.task == 'similarity':
                    labels = torch.from_numpy(np.array(batch_labels, dtype=np.int32)).long().to(self.device)

                yield packed_graphs_x, packed_graphs_y, labels, node_line_length



    def _pack_batch(self, graphs):
        Data_graphs = []
        node_line_length = []
        node_cnt = -1
        for g in graphs:
            edge_index = torch.from_numpy(np.array([g['from_idx'], g['to_idx']])).long().to(self.device)

            node_type = torch.from_numpy(np.array(g['node_type'])).long().to(self.device)
            node_content = torch.from_numpy(np.array(g['node_content'])).float().to(self.device)

            g['line_id'] = [int(i) for i in g['line_id']]
            line_ids = []
            for i in range(0, len(g['line_id'])):
                node_cnt += 1
                if i == 0 or g['line_id'][i] != g['line_id'][i-1]:
                    line_ids.append(g['line_id'][i])
                    node_line_length.append(node_cnt)

            num_nodes = len(g['node_type'])
            Data_graphs.append(Data(node_type=node_type, node_content=node_content,
                                    edge_index=edge_index, node2lineid=g['line_id'], lineid = line_ids,
                                    num_nodes=num_nodes))

        node_line_length.append(node_cnt)
        return Data_graphs, node_line_length