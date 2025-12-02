import math
import os
import pandas as pd
from sklearn.cluster import KMeans
import torch
import numpy as np
import scipy.sparse as sp
import metis
import networkx as nx
import torch.nn.functional as F
from scipy.sparse import linalg
def normalize_adj_mx(adj_mx, adj_type, return_type='dense'):
    if adj_type == 'normlap':
        adj = [calculate_normalized_laplacian(adj_mx)]
    elif adj_type == 'scalap':
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adj_type == 'symadj':
        adj = [calculate_sym_adj(adj_mx)]
    elif adj_type == 'transition':
        adj = [calculate_asym_adj(adj_mx)]
    elif adj_type == 'doubletransition':
        adj = [calculate_asym_adj(adj_mx), calculate_asym_adj(np.transpose(adj_mx))]
    elif adj_type == 'identity':
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        return []
    
    if return_type == 'dense':
        adj = [a.astype(np.float32).todense() for a in adj]
    elif return_type == 'coo':
        adj = [a.tocoo() for a in adj]
    return adj


def calculate_normalized_laplacian(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = sp.eye(adj_mx.shape[0]) - d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt).tocoo()
    return res


def calculate_scaled_laplacian(adj_mx, lambda_max=None, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    res = (2 / lambda_max * L) - I
    return res


def calculate_sym_adj(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt)
    return res


def calculate_asym_adj(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    res = d_mat_inv.dot(adj_mx)
    return res


def calculate_cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L.copy()]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[i - 1]) - LL[i - 2])
    return np.asarray(LL)

def partition_patch(n_patches,patch_layer,graph,adj_mx):
    mask=adj_mx.shape[0]
    patch = metis_partition(g=graph, n_patches=n_patches)
    if patch_layer==1:
        return patch
    elif patch_layer>1:
        for i in range(patch_layer-1):
            results=[]

            for cluster_id in range(patch.shape[0]):

                node_indices = patch[cluster_id, patch[cluster_id] != mask]

                sub_adj_matrix = adj_mx[np.ix_(node_indices, node_indices)]
                node_indices=torch.cat((node_indices, torch.tensor([mask])))
                g=adjacency_matrix_to_dict(sub_adj_matrix)
                sub_patch=metis_partition(g,n_patches)
               
                sub_patch_global_index=node_indices[sub_patch]
                results.append(sub_patch_global_index)

            max_size = max(tensor.size(1) for tensor in results)  
            
            padded_results = []
            for tensor in results:
                if tensor.size(1) < max_size:
                    padding = (0, max_size - tensor.size(1))  
                    padded_tensor = F.pad(tensor, padding, 'constant', mask)
                    padded_results.append(padded_tensor)
                else:
                    padded_results.append(tensor)
            
            result = torch.stack(padded_results)

            patch=result.reshape(-1,result.shape[-1])  
        new_shape = [n_patches] * patch_layer  
        new_shape.append(result.shape[-1])  
        result = result.reshape(*new_shape)
        return result
    

def metis_partition(g,n_patches):
    if g['num_nodes'] < n_patches:
        membership = torch.randperm(n_patches)
    else:
        adjlist = g['edge_index'].t().tolist()  
        weights = g['edge_weights'].tolist()     
        weights = [int(value * 100) for value in weights]
        G = nx.Graph()
        G.add_nodes_from(np.arange(g['num_nodes']))

        weighted_edges = [(edge[0], edge[1], weight) for edge, weight in zip(adjlist, weights)]
        G.add_weighted_edges_from(weighted_edges)
        G.graph[ 'edge_weight_attr' ]  =  'weight'
        # metis partition
        cuts, membership = metis.part_graph(G, n_patches, recursive=True,seed=2025)
    assert len(membership) >= g['num_nodes']
    membership = torch.tensor(membership[:g['num_nodes']])


    patch = []
    max_patch_size = -1
    for i in range(n_patches):
        patch.append(list())
        patch[-1] = torch.where(membership == i)[0].tolist()
        max_patch_size = max(max_patch_size, len(patch[-1]))

    for i in range(len(patch)):
        l = len(patch[i])
        if l < max_patch_size:
            patch[i] += [g['num_nodes']] * (max_patch_size - l)

    patch = torch.tensor(patch)
    return patch


def adjacency_matrix_to_dict(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    edge_index = np.argwhere(adj_matrix > 0)  
    
    g = {
        'num_nodes': num_nodes,
        'edge_index': torch.tensor(edge_index.T, dtype=torch.long),  
        'edge_weights': torch.tensor(adj_matrix[edge_index[:, 0], edge_index[:, 1]], dtype=torch.float)  
    }
    
    return g


def calculate_eigenvector(data,seq_len):

    degrees = np.sum(data, axis=1)
    D = np.diag(degrees)


    L = D - data

    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0  
    L_sym = np.eye(data.shape[0]) - D_inv_sqrt @ data @ D_inv_sqrt


    eigenvalues, eigenvectors = np.linalg.eig(L_sym)


    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    sorted_eigenvectors= sorted_eigenvectors/np.linalg.norm(sorted_eigenvectors, axis=0)

    if np.all(np.isreal(sorted_eigenvectors)):
        node_features = sorted_eigenvectors[:, 0:seq_len] 
    else:
        node_features = np.real(sorted_eigenvectors[:, 0:seq_len])  
    return node_features