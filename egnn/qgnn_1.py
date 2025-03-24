from torch import nn
import torch
import math
import pennylane as qml
import numpy as np
from egnn.egnn_new import EquivariantUpdate, unsorted_segment_sum, coord2diff, SinusoidsEmbeddingNew

class QGCL(nn.Module):
    """
    Quantum Graph Convolution Layer
    """
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False, n_qubits=None, hybrid_mode='adaptive', torch_device='cuda'):
        super(QGCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention
        self.input_nf = input_nf
        self.output_nf = output_nf
        self.hidden_nf = hidden_nf
        self.hybrid_mode = hybrid_mode
        self.act_fn = act_fn
        self.torch_device = torch_device
        
        # 确定n_qubits, 除非指定, 否则限制为最小所需数量
        # 太多qubits会严重限制训练效率(复杂度为2**n_qubits)
        self.n_qubits = n_qubits if n_qubits is not None else min(3, input_nf)
        
        # 经典边MLP用于处理边特征
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        
        # 新增：特征分析器 - 决定哪些特征更适合量子处理
        # self.feature_analyzer = nn.Sequential(
        #     nn.Linear(input_nf, hidden_nf//2),
        #     act_fn,
        #     nn.Linear(hidden_nf//2, self.n_qubits),
        #     nn.Sigmoid()
        # )
        # 量子节点更新部分
        # 创建量子设备和量子节点
        self.quantum_device = qml.device('default.qubit', wires=self.n_qubits)
        self.q_node = qml.QNode(self.quantum_circuit, self.quantum_device, interface='torch', diff_method='backprop')
        # alpha: distance offset, used for quantum circuit initialization
        self.alpha = nn.Parameter(torch.Tensor([0.1 / self.n_qubits]))
        
        # beta: kinetic strength, important for gradient flow
        self.beta = nn.Parameter(torch.Tensor([np.pi / (4 * self.n_qubits)]))
        
        # gamma: coupling strength, related to graph density
        self.gamma = nn.Parameter(torch.Tensor([0.1 / self.n_qubits]))
        
        # delta: anharmonic potential strength, controlling nonlinearity
        self.delta = nn.Parameter(torch.Tensor([0.05]))
        
        # Lambda: edge weight matrix, used for message passing
        diag_val = 0.1
        off_diag_val = -diag_val / (self.n_qubits - 1) if self.n_qubits > 1 else 0
        Lambda_init = torch.ones(self.n_qubits, self.n_qubits) * off_diag_val
        Lambda_init.fill_diagonal_(diag_val)
        self.Lambda = nn.Parameter(Lambda_init, requires_grad=True)

        # anharmonic potential parameter μ - more reasonable setting
        self.mu = 0.5  # center position
        self.omega = 1.0  # harmonic oscillator frequency
        
        # quantum output --> fully connected layer
        self.post_quantum = nn.Sequential(
            nn.Linear(hidden_nf + self.n_qubits, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
        
        # additional: quantum-classical fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_nf + self.n_qubits, hidden_nf),
            act_fn
        )
        # additional: quantum part weight controller
        self.quantum_weight = nn.Parameter(torch.Tensor([0.5]))  # initialized as equal weights
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        self.quantum_encoder = nn.Sequential(
            nn.Linear(input_nf + hidden_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, self.n_qubits)
        )
        
    
    def quantum_circuit(self, inputs):
        """
        QSGCNN：
        1. Coupling Hamiltonian H_C（Message Passing）
        2. Kinetic Hamiltonian H_K（Node Update）
        3. Anharmonic Hamiltonian H_A（Nonlinear）
        4. Kinetic Hamiltonian H_K（Final Update）
        """
        n_qubits = self.n_qubits
        
        # 1. Encode Input to Quantum State（Simulate Position Operator x）
        for i in range(n_qubits):
            qml.RY(inputs[i] * self.alpha, wires=i)
        
        # 2. Coupling Hamiltonian Evolution H_C
        # Approximate as IsingZZ interaction
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                # 使用对称化的Lambda值确保图同构性
                coupling_strength = self.gamma * (self.Lambda[i,j] + self.Lambda[j,i]) / 2
                qml.IsingZZ(coupling_strength, wires=[i,j])
                # Additional operations to better approximate square terms
                qml.CNOT(wires=[i,j])
                qml.RZ(coupling_strength * 0.5, wires=j)
                qml.CNOT(wires=[i,j])
        
        # 3. Kinetic Hamiltonian Evolution H_K
        for i in range(n_qubits):
            qml.RX(self.beta, wires=i)
        
        # 4. Anharmonic Hamiltonian Evolution H_A
        for i in range(n_qubits):
            # Approximate quartic potential function x^4 - μx^2
            # Introduce nonlinear phase through RZ gate
            qml.RZ(self.delta * (1.0 - self.mu * inputs[i]**2), wires=i)
            # Compound operations enhance nonlinearity
            qml.Hadamard(wires=i)
            qml.RZ(self.delta * inputs[i]**2, wires=i)
            qml.Hadamard(wires=i)
        
        # 5. Kinetic Hamiltonian Evolution Again
        for i in range(n_qubits):
            qml.RX(self.beta, wires=i)
        
        # Measure position expectation value
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                    normalization_factor=self.normalization_factor,
                                    aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        
        quantum_inputs = self.quantum_encoder(agg)
        batch_size = x.size(0)
        
        quantum_outputs_list = []
        for i in range(batch_size):
            # 获取量子电路的输出并转换为张量
            q_out = self.q_node(quantum_inputs[i].detach().cpu().numpy())
            q_out_tensor = torch.tensor(q_out, device=self.torch_device, dtype=x.dtype)
            quantum_outputs_list.append(q_out_tensor)
        
        quantum_outputs = torch.stack(quantum_outputs_list, dim=0)
        # 将量子输出与隐藏特征连接
        combined_quantum = torch.cat([agg, quantum_outputs], dim=1)
        out = x + self.post_quantum(combined_quantum)
        
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class QuantumEquivariantBlock(nn.Module):
    """
    经典EquivariantUpdate和量子QGCL
    """
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cuda', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum', n_qubits=None, torch_device='cuda'):
        super(QuantumEquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.n_qubits = n_qubits
        self.torch_device = torch_device

        for i in range(0, n_layers):
            self.add_module("qgcl_%d" % i, QGCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, 
                                               edges_in_d=edge_feat_nf, act_fn=act_fn, attention=attention,
                                               normalization_factor=self.normalization_factor,
                                               aggregation_method=self.aggregation_method,
                                               n_qubits=n_qubits, torch_device=self.torch_device))
                                               
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, 
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
            h, _ = self._modules["qgcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)

        if node_mask is not None:
            h = h * node_mask
        return h, x


class QEGNN(nn.Module):
    """
    Quantum Equivariant Graph Neural Network
    """
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cuda', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum', n_qubits=None):
        super(QEGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        
        self.n_qubits = n_qubits if n_qubits is not None else min(3, in_node_nf)
        print(f"Using {self.n_qubits} qubits for quantum computation")

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, QuantumEquivariantBlock(
                hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                act_fn=act_fn, n_layers=inv_sublayers,
                attention=attention, norm_diff=norm_diff, tanh=tanh,
                coords_range=coords_range, norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
                n_qubits=self.n_qubits,
                torch_device=device))
                
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        dtype = self.embedding.weight.dtype
        h = h.to(dtype)
        x = x.to(dtype)
        
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.embedding(h)
        
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h, x, edge_index, 
                node_mask=node_mask, 
                edge_mask=edge_mask, 
                edge_attr=distances
            )

        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x

# old methods
def create_hamiltonian_matrix_interaction(n_qubits, edge_index, edge_attr):
    full_matrix = np.zeros((2**n_qubits, 2**n_qubits))
    
    term_dictionary = {
        0: [np.identity(2), np.identity(2)],
        1: [qml.matrix(qml.PauliZ)(0), qml.matrix(qml.PauliZ)(0)],
    }
    
    weights = edge_attr
    fraction = 1.0 / 4.0
    number_of_terms = len(term_dictionary)
    
    for term in range(number_of_terms):
        for i, edge in enumerate(edge_index.T):
            interaction_term = 1
            for qubit in range(n_qubits):
                if qubit in edge:
                    if qubit == edge[0]:
                        interaction_term = np.kron(interaction_term, term_dictionary[term][0])
                    elif qubit == edge[1]:
                        interaction_term = np.kron(interaction_term, term_dictionary[term][1])
                else:
                    interaction_term = np.kron(interaction_term, np.identity(2))
            
            if np.all(term_dictionary[term][0] == qml.matrix(qml.PauliZ)(0)):
                full_matrix += -fraction * weights[i].numpy() * interaction_term
            else:
                full_matrix += fraction * weights[i].numpy() * interaction_term
    
    return torch.tensor(full_matrix, dtype=torch.float32)

def create_hamiltonian_matrix_transverse(n_qubits):
    full_matrix = np.zeros((2**n_qubits, 2**n_qubits))
    
    for i in range(n_qubits):
        x_term = 1
        for j in range(n_qubits):
            if j == i:
                x_term = np.kron(x_term, qml.matrix(qml.PauliX)(0))
            else:
                x_term = np.kron(x_term, np.identity(2))
        full_matrix += x_term
    
    return torch.tensor(full_matrix, dtype=torch.float32)
