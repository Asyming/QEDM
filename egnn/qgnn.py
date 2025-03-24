from torch import nn
import torch
import pennylane as qml
import numpy as np
from egnn.egnn_new import EquivariantUpdate, unsorted_segment_sum, coord2diff, SinusoidsEmbeddingNew

class QGCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False, n_qubits=None, hybrid_mode='adaptive', torch_device='cuda'):
        super(QGCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention
        self.hybrid_mode = hybrid_mode
        self.torch_device = torch_device

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        
        # self.node_mlp = nn.Sequential(
        #     nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
        #     act_fn,
        #     nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
            nn.Linear(hidden_nf, 1),
            nn.Sigmoid())

        # Variational Quantum Circuit & quantum parameter initialization
        self.n_qubits = n_qubits if n_qubits is not None else min(3, input_nf)
        self.qubits = list(range(self.n_qubits))
        self.quantum_encoder = nn.Sequential(
            nn.Linear(input_nf + hidden_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, self.n_qubits)
        )
        self.post_quantum = nn.Sequential(
            nn.Linear(hidden_nf * 2 + self.n_qubits, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)) # output_nf == hidden_nf

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
        # anharmonic potential parameter μ and ω
        self.mu = 0.5  # center position
        self.omega = 1.0  # harmonic oscillator frequency

    # edges_in_d.shape == edge_attr.shape
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
    
    def quantum_circuit(self, inputs):
        """
        QSGCNN：
        1. Coupling Hamiltonian H_C（Message Passing）
        2. Kinetic Hamiltonian H_K（Node Update）
        3. Anharmonic Hamiltonian H_A（Nonlinear）
        4. Kinetic Hamiltonian H_K（Final Update）
        """
        n_qubits = self.n_qubits
        # qml.AmplitudeEmbedding(inputs, wires=self.qubits, normalize=True)
        # 1. Encode Input to Quantum State（Simulate Position Operator x）
        for i in range(n_qubits):
            qml.RY(inputs[i] * self.alpha, wires=i)
        # 2. Coupling Hamiltonian Evolution H_C
        # Approximate as IsingZZ interaction
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                # Use the symmetricized Lambda value to ensure graph isomorphism
                coupling_strength = self.gamma * (self.Lambda[i,j] + self.Lambda[j,i]) / 2
                qml.IsingZZ(coupling_strength, wires=[i,j])
                # Additional operations to better approximate square terms
                # qml.CNOT(wires=[i,j])
                # qml.RZ(coupling_strength * 0.5, wires=j)
                # qml.CNOT(wires=[i,j])
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

    def node_model(self, x, edge_index, edge_attr, node_attr):
        # x_size: batch_size * n_nodes, hidden_nf
        # agg_size == x_size
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1) #d: h_hidden+h_hidden
        quantum_inputs = self.quantum_encoder(agg) #d: h_hidden+h_hidden+n_qubits-->n_qubits
        quantum_outputs = torch.vstack([
            torch.tensor(self.q_node(q_in.detach().cpu().numpy()), 
                        device=self.torch_device, 
                        dtype=x.dtype)
            for q_in in quantum_inputs
        ])

        agg = torch.cat([agg, quantum_outputs], dim=1) #d: h_hidden+h_hidden+n_qubits
        out = x + self.post_quantum(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


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


class QuantumEquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
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

        for i in range(0, n_layers):
            self.add_module("qgcl_%d" % i, QGCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["qgcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class QEGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
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

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("qe_block_%d" % i, QuantumEquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["qe_block_%d" % i](h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x
