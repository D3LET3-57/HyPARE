"""Spec-driven CLRS adapter for arbitrary CLRS algorithms.

The BFS-specific adapter in clrs_native_adapter.py is retained for the
validated BFS baseline. This adapter exposes every input, hint, and output
declared by the CLRS spec, which is required for DFS's ten hint fields.
"""

import numpy as np
import torch

from clrs._src import samplers, specs


class CLRSSpecAdapter:
    def __init__(self, algorithm: str, split: str = 'train',
                 length: int = 16, batch_size: int = 32,
                 seed: int = 42, num_samples: int = -1,
                 randomize_pos: bool = True):
        self.algorithm = algorithm
        self.split = split
        self.length = length
        self.batch_size = batch_size
        self.seed = seed
        self.randomize_pos = randomize_pos
        self.num_samples = num_samples
        self._pos_rng = np.random.RandomState(seed)

        self.sampler, _ = samplers.build_sampler(
            name=algorithm,
            num_samples=num_samples,
            length=length,
            seed=seed,
            track_max_steps=True,
        )
        self.spec = specs.SPECS[algorithm]
        self.categorical_dims = {}

    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.num_samples > 0:
            batch_size = min(batch_size, self.num_samples)
        feedback = self.sampler.next(batch_size=batch_size)
        if self.randomize_pos:
            feedback = self._randomize_pos(feedback)
        return self._feedback_to_tensors(feedback)

    def get_full_batch(self):
        feedback = self.sampler.next(batch_size=None)
        if self.randomize_pos:
            feedback = self._randomize_pos(feedback)
        return self._feedback_to_tensors(feedback)

    def _randomize_pos(self, feedback):
        pos_inputs = [x for x in feedback.features.inputs if x.name == 'pos']
        if not pos_inputs:
            return feedback
        pos = pos_inputs[0]
        batch_size, num_nodes = pos.data.shape
        unsorted = self._pos_rng.uniform(size=(batch_size, num_nodes))
        new_pos = []
        for i in range(batch_size):
            split, = np.where(pos.data[i] == 0)
            split = np.concatenate([split, [num_nodes]])
            new_pos.append(np.concatenate([
                np.sort(unsorted[i, split[j]:split[j + 1]])
                for j in range(len(split) - 1)
            ]))
        pos.data = np.array(new_pos)
        inputs = [(pos if x.name == 'pos' else x)
                  for x in feedback.features.inputs]
        features = feedback.features._replace(inputs=inputs)
        return feedback._replace(features=features)

    @staticmethod
    def _batch_first(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr[None, :]
        return np.transpose(arr, (1, 0) + tuple(range(2, arr.ndim)))

    def _feedback_to_tensors(self, feedback):
        lengths = torch.from_numpy(
            np.asarray(feedback.features.lengths)).long()
        batch_size = lengths.numel()

        inputs = {}
        num_nodes = None
        for inp in feedback.features.inputs:
            arr = np.asarray(inp.data)
            if inp.location in (specs.Location.NODE, specs.Location.EDGE):
                if num_nodes is None:
                    num_nodes = arr.shape[1]
            data = torch.from_numpy(arr).float()
            if inp.location == specs.Location.GRAPH and data.dim() == 1:
                data = data.unsqueeze(-1)
            if inp.type_ == specs.Type.POINTER and data.dim() == 2:
                data = torch.nn.functional.one_hot(
                    data.long(), num_nodes).float()
            inputs[inp.name] = data

        hints = {}
        max_steps = 0
        for hint in feedback.features.hints:
            arr = self._batch_first(hint.data)
            _, loc, typ = self.spec[hint.name]
            t = arr.shape[1]
            max_steps = max(max_steps, t)
            valid_steps = (torch.arange(t).unsqueeze(0)
                           < lengths.unsqueeze(1))

            if typ == specs.Type.POINTER:
                target = torch.from_numpy(arr).long()
                if target.dim() == 4:
                    target = target.argmax(dim=-1)
                if target.dim() == 3:
                    mask = (valid_steps.unsqueeze(-1).expand_as(target)
                            & (target != -1))
                else:
                    mask = valid_steps & (target != -1)
            elif typ == specs.Type.MASK:
                target = torch.from_numpy(arr).float()
                mask = valid_steps
                if target.dim() == 3:
                    mask = valid_steps.unsqueeze(-1) & (target != -1)
            elif typ == specs.Type.MASK_ONE:
                target = torch.from_numpy(arr).argmax(dim=-1).long()
                mask = valid_steps & (torch.from_numpy(arr).max(dim=-1).values > 0)
            elif typ == specs.Type.CATEGORICAL:
                target = torch.from_numpy(arr).argmax(dim=-1).long()
                self.categorical_dims[hint.name] = arr.shape[-1]
                mask = (valid_steps.unsqueeze(-1)
                        & (torch.from_numpy(arr).max(dim=-1).values > 0))
            elif typ == specs.Type.SCALAR:
                target = torch.from_numpy(arr).float()
                mask = valid_steps
                if target.dim() == 3:
                    mask = valid_steps.unsqueeze(-1).expand_as(target)
            else:
                raise NotImplementedError(
                    f'Unsupported hint type {typ} for {hint.name}')

            hints[hint.name] = {
                'type': typ,
                'location': loc,
                'target': target,
                'mask': mask.float(),
            }

        outputs = {}
        for out in feedback.outputs:
            arr = np.asarray(out.data)
            _, loc, typ = self.spec[out.name]
            if typ == specs.Type.POINTER:
                target = torch.from_numpy(arr).long()
                if target.dim() == 3:
                    target = target.argmax(dim=-1)
                mask = (target != -1).float()
            elif typ == specs.Type.MASK_ONE:
                target = torch.from_numpy(arr).argmax(dim=-1).long()
                mask = (torch.from_numpy(arr).max(dim=-1).values > 0).float()
            elif typ == specs.Type.CATEGORICAL:
                target = torch.from_numpy(arr).argmax(dim=-1).long()
                self.categorical_dims[out.name] = arr.shape[-1]
                mask = (torch.from_numpy(arr).max(dim=-1).values > 0).float()
            elif typ == specs.Type.MASK:
                target = torch.from_numpy(arr).float()
                mask = (target != -1).float()
            elif typ == specs.Type.SCALAR:
                target = torch.from_numpy(arr).float()
                if target.dim() == 1:
                    target = target.unsqueeze(-1)
                mask = torch.ones_like(target, dtype=torch.float32)
            else:
                raise NotImplementedError(
                    f'Unsupported output type {typ} for {out.name}')
            outputs[out.name] = {
                'type': typ,
                'location': loc,
                'target': target,
                'mask': mask,
            }

        return {
            'inputs': inputs,
            'hints': hints,
            'outputs': outputs,
            'lengths': lengths,
            'num_nodes': num_nodes,
            'max_steps': max_steps,
            'spec': self.spec,
            'categorical_dims': dict(self.categorical_dims),
        }
