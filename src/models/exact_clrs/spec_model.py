"""Spec-driven CLRS encode-process-decode model.

This module keeps the validated BFS-specific implementation untouched and adds
the multi-hint machinery needed by DFS. Every input, hint, and output is read
from the CLRS spec, so all ten DFS hints are encoded and decoded by name.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from clrs._src import specs

from .decoder import ExactCLRSPointerDecoder
from .mpnn import ExactCLRSMpnn
from .hypare import ExactCLRSHyPARE, ExactCLRSMpnnGRU


class SpecCLRSEncoder(nn.Module):
    def __init__(self, hidden_dim, spec, categorical_dims):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spec = spec
        self.encoders = nn.ModuleDict()
        for name, (stage, loc, typ) in spec.items():
            if stage not in (specs.Stage.INPUT, specs.Stage.HINT):
                continue
            if loc == specs.Location.EDGE and typ == specs.Type.POINTER:
                raise NotImplementedError('Edge pointer encoders are not supported')
            if loc == specs.Location.GRAPH and typ == specs.Type.POINTER:
                raise NotImplementedError('Graph pointer encoders are not supported')
            in_dim = categorical_dims.get(name, 1)
            self.encoders[name] = nn.Linear(in_dim, hidden_dim)

    def encode_inputs(self, inputs, num_nodes, device):
        batch_size = next(iter(inputs.values())).shape[0]
        node_fts = torch.zeros(batch_size, num_nodes, self.hidden_dim,
                               device=device)
        edge_fts = torch.zeros(batch_size, num_nodes, num_nodes,
                               self.hidden_dim, device=device)
        graph_fts = torch.zeros(batch_size, self.hidden_dim, device=device)
        adj_mat = torch.eye(num_nodes, device=device).unsqueeze(0).expand(
            batch_size, -1, -1).clone()

        for name, (stage, loc, typ) in self.spec.items():
            if stage != specs.Stage.INPUT:
                continue
            node_fts, edge_fts, graph_fts, adj_mat = self._accumulate(
                name, inputs[name].to(device), loc, typ,
                node_fts, edge_fts, graph_fts, adj_mat)
        return node_fts, edge_fts, graph_fts, adj_mat

    def add_hint(self, name, data, node_fts, edge_fts, graph_fts, adj_mat):
        _, loc, typ = self.spec[name]
        return self._accumulate(
            name, data, loc, typ, node_fts, edge_fts, graph_fts, adj_mat)

    def _accumulate(self, name, data, loc, typ,
                    node_fts, edge_fts, graph_fts, adj_mat):
        encoder = self.encoders[name]
        batch_size = data.shape[0]
        if loc == specs.Location.NODE:
            if typ == specs.Type.POINTER:
                edge_fts = edge_fts + encoder(data.unsqueeze(-1))
                adj_mat = (adj_mat + (
                    (data + data.transpose(1, 2)) > 0.5).float()) > 0
            elif typ == specs.Type.CATEGORICAL:
                node_fts = node_fts + encoder(data)
            else:
                node_fts = node_fts + encoder(data.unsqueeze(-1))
        elif loc == specs.Location.EDGE:
            edge_fts = edge_fts + encoder(data.unsqueeze(-1))
            if typ == specs.Type.MASK:
                adj_mat = (adj_mat + (
                    (data + data.transpose(1, 2)) > 0.0).float()) > 0
        elif loc == specs.Location.GRAPH:
            if typ == specs.Type.CATEGORICAL:
                graph_fts = graph_fts + encoder(data)
            else:
                graph_fts = graph_fts + encoder(
                    data.reshape(batch_size, -1))
        else:
            raise NotImplementedError(f'Unsupported location {loc}')
        return node_fts, edge_fts, graph_fts, adj_mat


class _GraphScalarDecoder(nn.Module):
    def __init__(self, decoder_input_dim, hidden_dim, num_classes=1):
        super().__init__()
        self.node_projection = nn.Linear(decoder_input_dim, num_classes)
        self.graph_projection = nn.Linear(hidden_dim, num_classes)

    def forward(self, h_t, graph_fts):
        gr_emb = h_t.max(dim=1).values
        return self.node_projection(gr_emb) + self.graph_projection(graph_fts)


class SpecCLRSDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_input_dim, spec, categorical_dims):
        super().__init__()
        self.spec = spec
        self.hidden_dim = hidden_dim
        self.pointer_decoders = nn.ModuleDict()
        self.node_decoders = nn.ModuleDict()
        self.graph_decoders = nn.ModuleDict()

        for name, (stage, loc, typ) in spec.items():
            if stage not in (specs.Stage.HINT, specs.Stage.OUTPUT):
                continue
            if loc == specs.Location.NODE:
                if typ in (specs.Type.POINTER,):
                    self.pointer_decoders[name] = ExactCLRSPointerDecoder(
                        hidden_dim, decoder_input_dim)
                elif typ == specs.Type.CATEGORICAL:
                    self.node_decoders[name] = nn.Linear(
                        decoder_input_dim, categorical_dims[name])
                else:
                    self.node_decoders[name] = nn.Linear(
                        decoder_input_dim, 1)
            elif loc == specs.Location.GRAPH:
                num_classes = (categorical_dims[name]
                               if typ == specs.Type.CATEGORICAL else 1)
                self.graph_decoders[name] = _GraphScalarDecoder(
                    decoder_input_dim, hidden_dim, num_classes)
            else:
                raise NotImplementedError(
                    f'Unsupported decoder location {loc} for {name}')

    def decode(self, name, h_t, edge_fts, graph_fts):
        _, loc, typ = self.spec[name]
        if loc == specs.Location.NODE:
            if typ == specs.Type.POINTER:
                return self.pointer_decoders[name](h_t, edge_fts, None)
            decoded = self.node_decoders[name](h_t)
            if typ == specs.Type.CATEGORICAL:
                return decoded
            return decoded.squeeze(-1)
        if loc == specs.Location.GRAPH:
            decoded = self.graph_decoders[name](h_t, graph_fts)
            if typ == specs.Type.CATEGORICAL:
                return decoded
            return decoded.squeeze(-1)
        raise NotImplementedError(f'Unsupported decoder location {loc}')

    def postprocess(self, name, prediction, hard=False):
        typ = self.spec[name][2]
        if typ == specs.Type.POINTER:
            if hard:
                return prediction.argmax(dim=-1)
            return F.softmax(prediction, dim=-1)
        if typ == specs.Type.MASK:
            return torch.sigmoid(prediction)
        if typ in (specs.Type.MASK_ONE, specs.Type.CATEGORICAL):
            if hard:
                return prediction.argmax(dim=-1)
            return F.softmax(prediction, dim=-1)
        return prediction


class SpecCLRSNet(nn.Module):
    def __init__(self, spec, categorical_dims, hidden_dim=128,
                 nb_msg_passing_steps=1, processor_type='mpnn',
                 gate_init=-2.0, use_ln=True, dropout_prob=0.0):
        super().__init__()
        self.spec = spec
        self.categorical_dims = dict(categorical_dims)
        self.hidden_dim = hidden_dim
        self.nb_msg_passing_steps = nb_msg_passing_steps
        self.processor_type = processor_type
        self.dropout_prob = dropout_prob
        self.hint_names = [
            name for name, (stage, _, _) in spec.items()
            if stage == specs.Stage.HINT
        ]
        self.output_names = [
            name for name, (stage, _, _) in spec.items()
            if stage == specs.Stage.OUTPUT
        ]

        self.encoder = SpecCLRSEncoder(
            hidden_dim, spec, categorical_dims)
        if processor_type == 'mpnn':
            self.processor = ExactCLRSMpnn(
                hidden_dim=hidden_dim,
                msg_mlp_sizes=[hidden_dim, hidden_dim],
                use_full_adjacency=True,
                use_ln=use_ln,
            )
        elif processor_type == 'mpnn_gru':
            self.processor = ExactCLRSMpnnGRU(
                hidden_dim=hidden_dim, alpha=0.5, use_ln=use_ln)
        elif processor_type == 'hypare':
            self.processor = ExactCLRSHyPARE(
                hidden_dim=hidden_dim, gate_init=gate_init, use_ln=use_ln)
        else:
            raise ValueError(f'Unknown processor_type: {processor_type}')

        self.decoder = SpecCLRSDecoder(
            hidden_dim, 3 * hidden_dim, spec, categorical_dims)

    def _target_at(self, name, hint, step, num_nodes):
        typ = hint['type']
        target = hint['target'][:, step]
        mask = hint['mask'][:, step]
        while mask.dim() < target.dim():
            mask = mask.unsqueeze(-1)
        target = torch.where(
            mask > 0, target, torch.zeros_like(target))
        if typ == specs.Type.POINTER:
            return F.one_hot(target.long(), num_nodes).float()
        if typ == specs.Type.MASK_ONE:
            return F.one_hot(target.long(), num_nodes).float()
        if typ == specs.Type.CATEGORICAL:
            return F.one_hot(
                target.long(), self.categorical_dims[name]).float()
        return target

    def forward(self, inputs, hints, lengths=None, num_steps=None,
                teacher_forcing_ratio=0.0):
        device = next(iter(inputs.values())).device
        batch_size = next(iter(inputs.values())).shape[0]
        num_nodes = None
        for name, (stage, loc, _) in self.spec.items():
            if stage == specs.Stage.INPUT and loc in (
                    specs.Location.NODE, specs.Location.EDGE):
                num_nodes = inputs[name].shape[1]
                break

        total_steps = hints[self.hint_names[0]]['target'].shape[1]
        if num_steps is None:
            num_steps = max(1, total_steps - 1)

        node_fts, edge_fts, graph_fts, adj_mat = self.encoder.encode_inputs(
            inputs, num_nodes, device)
        hidden = torch.zeros(batch_size, num_nodes, self.hidden_dim,
                             device=device)
        gru_state = None
        hint_preds = {name: [] for name in self.hint_names}
        output_preds = {name: None for name in self.output_names}
        gate_values = []

        for step in range(num_steps):
            force_mask = (
                torch.ones(batch_size, dtype=torch.bool, device=device)
                if step == 0 else
                torch.rand(batch_size, device=device) < teacher_forcing_ratio
            )

            combined_node = node_fts.clone()
            combined_edge = edge_fts.clone()
            combined_graph = graph_fts.clone()
            combined_adj = adj_mat.clone()

            for name in self.hint_names:
                hint = hints[name]
                target = self._target_at(name, hint, step, num_nodes)
                if step == 0:
                    hint_data = target
                else:
                    previous = self.decoder.postprocess(
                        name, hint_preds[name][-1])
                    expand = force_mask.view(
                        -1, *([1] * (target.dim() - 1)))
                    hint_data = torch.where(expand, target, previous)
                combined_node, combined_edge, combined_graph, combined_adj = \
                    self.encoder.add_hint(
                        name, hint_data,
                        combined_node, combined_edge, combined_graph,
                        combined_adj)

            nxt_hidden = hidden
            for _ in range(self.nb_msg_passing_steps):
                if self.processor_type == 'hypare':
                    nxt_hidden, gate, gru_state = self.processor(
                        node_fts=combined_node,
                        edge_fts=combined_edge,
                        graph_fts=combined_graph,
                        adj_mat=combined_adj,
                        hidden=nxt_hidden,
                        gru_state=gru_state,
                    )
                    gate_values.append(gate)
                elif self.processor_type == 'mpnn_gru':
                    nxt_hidden, gru_state = self.processor(
                        node_fts=combined_node,
                        edge_fts=combined_edge,
                        graph_fts=combined_graph,
                        adj_mat=combined_adj,
                        hidden=nxt_hidden,
                        gru_state=gru_state,
                    )
                else:
                    nxt_hidden = self.processor(
                        node_fts=combined_node,
                        edge_fts=combined_edge,
                        graph_fts=combined_graph,
                        adj_mat=combined_adj,
                        hidden=nxt_hidden,
                    )

            if self.training and self.dropout_prob > 0.0:
                nxt_hidden = F.dropout(
                    nxt_hidden, p=self.dropout_prob, training=True)

            h_t = torch.cat([combined_node, hidden, nxt_hidden], dim=-1)
            for name in self.hint_names:
                hint_preds[name].append(
                    self.decoder.decode(
                        name, h_t, combined_edge, combined_graph))
            for name in self.output_names:
                prediction = self.decoder.decode(
                    name, h_t, combined_edge, combined_graph)
                if output_preds[name] is None:
                    output_preds[name] = prediction
                if lengths is not None:
                    is_final = step == (lengths - 2).clamp(min=0)
                    expand = is_final.view(
                        -1, *([1] * (prediction.dim() - 1)))
                    output_preds[name] = torch.where(
                        expand, prediction, output_preds[name])
                else:
                    output_preds[name] = prediction

            if lengths is not None:
                active = step < (lengths - 1).clamp(min=1)
                hidden = torch.where(
                    active[:, None, None], nxt_hidden, hidden)
            else:
                hidden = nxt_hidden

        result = {
            'hint_preds': hint_preds,
            'output_preds': output_preds,
        }
        if gate_values:
            result['gate_values'] = gate_values
        return result
