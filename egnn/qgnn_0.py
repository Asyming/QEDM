from torch import nn
import torch
import math
import pennylane as qml
import copy

class QGCL(nn.Module):
    """Quantum Graph Convolution Layer"""
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), n_qubits=7, n_layers=2):
        super(QGCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.embedding_size = 2**n_qubits
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        
        self.q_encoder = nn.Linear(hidden_nf + input_nf + nodes_att_dim, self.embedding_size)
        
        # 可学习参数
        self.coeffs_param = nn.Parameter(torch.randn(self.n_layers, 2) * 0.1)
        self.A_real_param = nn.Parameter(torch.randn(self.n_layers, self.embedding_size, self.embedding_size) * 0.01)
        self.A_imag_param = nn.Parameter(torch.randn(self.n_layers, self.embedding_size, self.embedding_size) * 0.01)
        
        self.q_decoder = nn.Linear(self.embedding_size, output_nf)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qubits = list(range(self.n_qubits))
        self.q_device = qml.device("default.qubit", wires=self.n_qubits)
        self.q_circuit = qml.QNode(self.quantum_circuit, self.q_device, interface="torch")

    def quantum_circuit(self, inputs, coeffs, q_matrices):
        """
        VQC circuit
        Args:
            inputs: quantum state vector
            coeffs: evolution coefficients
            q_matrices: unitary matrices after Cayley transformation
        """
        qml.QubitStateVector(inputs, wires=self.qubits)
        
        for j in range(self.n_layers):
            qml.QubitUnitary(q_matrices[j], wires=self.qubits)
            
            for i in range(self.n_qubits):
                qml.RX(coeffs[j, 0], wires=i)
                qml.RY(coeffs[j, 1], wires=i)
            
            for i in range(self.n_qubits-1):
                qml.CNOT(wires=[i, i+1])
            if self.n_qubits > 1:
                qml.CNOT(wires=[self.n_qubits-1, 0])
        
        return [qml.state()]

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        
        mij = self.edge_mlp(out)
        
        if edge_mask is not None:
            mij = mij * edge_mask
        return mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        # 构造量子处理的输入
        if node_attr is not None:
            input_tensor = torch.cat([x, agg, node_attr], dim=1)
        else:
            input_tensor = torch.cat([x, agg], dim=1)
        
        q_input = self.q_encoder(input_tensor)
        
        batch_size = q_input.size(0)
        q_outputs = []
        
        q_matrices = []
        for j in range(self.n_layers):
            A = self.A_real_param[j].to(self.device) + 1j * self.A_imag_param[j].to(self.device)
            I = torch.eye(A.shape[0], dtype=torch.complex128).to(self.device)
            A = A + A.conj().T
            q = (A - 1j*I) @ torch.linalg.inv(A + 1j*I)
            q_matrices.append(q)
        
        for i in range(batch_size):
            node_input = q_input[i:i+1]
            norm_factor = torch.sqrt(torch.sum(torch.abs(node_input)**2) + 1e-12)
            normalized_input = node_input / norm_factor
            
            q_state = self.q_circuit(normalized_input.flatten().detach().cpu(), 
                                     self.coeffs_param.detach().cpu(),
                                     [q.detach().cpu() for q in q_matrices])
            
            q_outputs.append(torch.tensor(q_state[0], device=self.device).reshape(1, -1))
        
        q_output = torch.cat(q_outputs, dim=0)
        out = x + self.q_decoder(q_output.real.float())
        return out

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, edge_feat


class QuantumEquivariantBlock(nn.Module):
    """量子等变块，使用QGCL替代GCL"""
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cuda', act_fn=nn.SiLU(), n_layers=2, n_qubits=7,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(QuantumEquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        # 量子图卷积层
        for i in range(0, n_layers):
            self.add_module("qgcl_%d" % i, QGCL(
                hidden_nf, hidden_nf, hidden_nf, 
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=edge_feat_nf, 
                act_fn=act_fn,
                n_qubits=n_qubits,
                n_layers=2))
        
        self.add_module("qgcl_equiv", EquivariantUpdate(
            hidden_nf, edges_in_d=edge_feat_nf, 
            act_fn=nn.SiLU(), tanh=tanh,
            coords_range=self.coords_range_layer,
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method))
        
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1) if edge_attr is not None else distances
        
        for i in range(0, self.n_layers):
            h, _ = self._modules["qgcl_%d" % i](h, edge_index, edge_attr=edge_attr, 
                                               node_mask=node_mask, edge_mask=edge_mask)
        
        x = self._modules["qgcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)

        if node_mask is not None:
            h = h * node_mask
        return h, x


class QEGNN(nn.Module):
    """Quantum Equivariant Graph Neural Network"""
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cuda', act_fn=nn.SiLU(), n_layers=3, n_qubits=7,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(QEGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        
        # 量子等变块
        for i in range(0, n_layers):
            self.add_module("qe_block_%d" % i, QuantumEquivariantBlock(
                hidden_nf, edge_feat_nf=edge_feat_nf, 
                device=device, act_fn=act_fn, 
                n_layers=inv_sublayers,
                n_qubits=n_qubits,
                norm_diff=norm_diff, tanh=tanh,
                coords_range=coords_range, 
                norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method))
        
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        
        h = self.embedding(h)
        
        for i in range(0, self.n_layers):
            h, x = self._modules["qe_block_%d" % i](h, x, edge_index, 
                                                  node_mask=node_mask, 
                                                  edge_mask=edge_mask, 
                                                  edge_attr=distances)

        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x


# 保留原有的EquivariantUpdate类（不需要量子化）
class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


# 保留原有的辅助函数
def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


# 保留原有的正弦嵌入类
class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()