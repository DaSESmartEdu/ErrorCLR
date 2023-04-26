import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_add_pool, GraphConv, GCNConv
from torch_geometric.data import Batch
from torch_geometric.nn.glob import GlobalAttention

def pairwise_euclidean_similarity(x, y):
    return torch.cdist(x,y,p=2)


def pairwise_dot_product_similarity(x, y):
    return torch.mm(x, torch.transpose(y, 1, 0))


def pairwise_cosine_similarity(x, y):
    x = torch.div(x, torch.sqrt(max(torch.sum(x ** 2), 1e-12)))
    y = torch.div(y, torch.sqrt(max(torch.sum(y ** 2), 1e-12)))
    return torch.mm(x, torch.transpose(y, 1, 0))


PAIRWISE_SIMILARITY_FUNCTION = {
    'euclidean': pairwise_euclidean_similarity,
    'dotproduct': pairwise_dot_product_similarity,
    'cosine': pairwise_cosine_similarity,
}

class NodeEmbedding(nn.Module):
    def __init__(self, node_type_num, node_type_dim, node_content_dim, node_embedding_dim, device):
        super(NodeEmbedding, self).__init__()
        self.node_type_embedding = nn.Embedding(node_type_num, node_type_dim)
        self.linear = nn.Linear(node_content_dim + node_type_dim, node_embedding_dim)
        self.device = device

    def forward(self, node_types, node_contents):
        type_info = self.node_type_embedding(node_types)
        node_info = torch.cat([type_info, node_contents], dim=1)
        return self.linear(node_info)



class GraphMatchingNet(nn.Module):
    """A graph to embedding mapping network."""

    def __init__(self, node_type_num, node_type_dim, node_content_dim, node_state_dim, graph_state_dim,similarity_name, device):
        super(GraphMatchingNet, self).__init__()
        self.node_embed = NodeEmbedding(node_type_num, node_type_dim, node_content_dim, node_state_dim, device)
        self.node_propgation = nn.ModuleList([GatedGraphConv(node_state_dim, num_layers=1) for i in range(5)])
        self.pool = global_add_pool
        self.GRU = torch.nn.GRU(node_state_dim * 3, node_state_dim)
        self.sim = similarity_name
        self.gated_layer = nn.Linear(node_state_dim, node_state_dim * 2)
        self.aggr_layer = nn.Linear(node_state_dim, graph_state_dim)
        self.graph_state_dim = graph_state_dim


    def compute_cross_attention(self, g1_node_state, g2_node_state, name='cosine'):
        sim = PAIRWISE_SIMILARITY_FUNCTION[name]

        result1 = []
        result2 = []
        attns = []
        for x, y in zip(g1_node_state, g2_node_state):
            a = sim(x, y)
            a_x = torch.softmax(a, dim=1)  # i->j
            a_y = torch.softmax(a, dim=0)  # j->i
            attention_x = torch.mm(a_x, y)
            attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
            result1.append(attention_x)
            result2.append(attention_y)

            attns.append(a_x.cpu().detach().numpy().tolist())

        result1_ = torch.cat(result1, dim=0)
        result2_ = torch.cat(result2, dim=0)

        return result1_, result2_, attns

    def update_node(self, node_state, prop_message, attention=None, node_feature=None):
        node_state = torch.unsqueeze(node_state,0)   # h_0
        if attention is None:
            node_state_inputs = torch.cat([prop_message, node_feature], dim=1)
        elif node_feature is None:
            node_state_inputs = torch.cat([prop_message, attention], dim=1)
        else:
            node_state_inputs = torch.cat([prop_message, attention, node_feature], dim=1)
        node_state_inputs = torch.unsqueeze(node_state_inputs, 0)  # new_x
        _, new_node_states = self.GRU(node_state_inputs, node_state)  # do once reset and update
        new_node_states = torch.squeeze(new_node_states)

        return new_node_states

    def aggregate_graph(self, node_states, g):
        node_states_g = self.gated_layer(node_states)
        gates = torch.sigmoid(node_states_g[:, :self.graph_state_dim])
        node_states_g = node_states_g[:, self.graph_state_dim:] * gates
        graph_states = self.pool(node_states_g, batch=g.batch)
        graph_states = self.aggr_layer(graph_states)

        return graph_states


    def forward(self, g1, g2, pretrained=False):
        g1_num_nodes = [g['num_nodes'] for g in g1]
        g1 = Batch.from_data_list(g1)

        g2_num_nodes = [g['num_nodes'] for g in g2]
        g2 = Batch.from_data_list(g2)

        # NOTE: GraphEncoder
        g1_node_state_initial = self.node_embed(g1['node_type'], g1['node_content'])
        g2_node_state_initial = self.node_embed(g2['node_type'], g2['node_content'])
        g1_node_state, g2_node_state = g1_node_state_initial, g2_node_state_initial

        visualize_attns = []
        # NOTE: GraphPropLayer
        for layer in self.node_propgation:
            # NOTE: _compute_aggregated_messages: Upgrade to GatedGraphConv instead of Linear
            g1_message = layer(g1_node_state, g1['edge_index'])
            g2_message = layer(g2_node_state, g2['edge_index'])

            # NOTE: batch_block_pair_attention
            g1_attention, g2_attention, visualize_attn = self.compute_cross_attention(
                                                torch.split(g1_node_state, g1_num_nodes, dim=0),
                                                torch.split(g2_node_state, g2_num_nodes, dim=0))
            visualize_attns.append(visualize_attn)
            # NOTE: _compute_node_update
            g1_node_state = self.update_node(g1_node_state, g1_message, g1_attention, g1_node_state_initial)
            g2_node_state = self.update_node(g2_node_state, g2_message, g2_attention, g2_node_state_initial)


        # Note: GraphAggregator
        if pretrained:
            g1_output = self.aggregate_graph(g1_node_state, g1)
            g2_output = self.aggregate_graph(g2_node_state, g2)
            
            return g1_output, g2_output, g1_node_state

        else:
            return g1_node_state, visualize_attns


class ErrorPredictionModel(nn.Module):
    """A graph to embedding mapping network."""
    def __init__(self, node_type_num, node_type_dim, node_content_dim, node_state_dim,
                 graph_state_dim,similarity_name, device,
                 error_type_num_1, error_type_num_2):
        super(ErrorPredictionModel, self).__init__()
        self.node_state_model = GraphMatchingNet(node_type_num, node_type_dim, node_content_dim, node_state_dim,
                         graph_state_dim, similarity_name, device)
        self.linear_1 = nn.Linear(node_state_dim, error_type_num_1)
        self.linear_2 = nn.Linear(node_state_dim + error_type_num_1, error_type_num_2)
        # self.linear_1 = nn.Linear(node_state_dim, node_state_dim // 2)
        # self.linear_2 = nn.Linear(node_state_dim // 2, error_type_num_1)
        self.weights = torch.nn.Parameter(torch.ones(3).float())
        self.dropout = nn.Dropout()

    def forward(self, g1, g2, nll):
        node_state, visualize_attn = self.node_state_model(g1, g2)
        # node_state = self.dropout(node_state)

        line_state = torch.cat([node_state[nll[i]:nll[i+1]].mean(dim=0).unsqueeze(0) for i in range(0, len(nll)-1)], dim=0)

        output_type1 = self.linear_1(line_state)
        mixed_line_state = torch.cat([line_state, output_type1], dim=1)
        output_type1 = torch.softmax(output_type1, dim=1)
        output_type2 = torch.softmax(self.linear_2(mixed_line_state), dim=1)

        return output_type1, output_type2, visualize_attn

    def get_last_shared_layer(self):
        return self.linear_1