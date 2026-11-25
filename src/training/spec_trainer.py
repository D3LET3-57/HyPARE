"""Training utilities for the spec-driven CLRS model."""

import os
import time

import torch
import torch.nn.functional as F

from clrs._src import specs


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(v, device) for v in value)
    return value


def _weighted_mean(loss, mask):
    mask = mask.float()
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def _step_loss(prediction, target, mask, typ, num_nodes=None):
    if typ == specs.Type.POINTER:
        if prediction.dim() != 3:
            raise ValueError(f'Pointer prediction must be [B,N,N], got {prediction.shape}')
        target = target.long()
        flat_mask = mask.reshape(-1).float()
        safe_target = target.reshape(-1).clone()
        safe_target[flat_mask == 0] = 0
        loss = F.cross_entropy(
            prediction.reshape(-1, prediction.shape[-1]),
            safe_target,
            reduction='none',
        )
        return _weighted_mean(loss, flat_mask)
    if typ == specs.Type.MASK:
        safe_target = torch.where(
            mask > 0, target.float(), torch.zeros_like(target.float()))
        loss = F.binary_cross_entropy_with_logits(
            prediction, safe_target, reduction='none')
        return _weighted_mean(loss, mask)
    if typ == specs.Type.MASK_ONE:
        target = target.long()
        flat_mask = mask.reshape(-1).float()
        safe_target = target.reshape(-1).clone()
        safe_target[flat_mask == 0] = 0
        loss = F.cross_entropy(
            prediction.reshape(-1, prediction.shape[-1]),
            safe_target,
            reduction='none',
        )
        return _weighted_mean(loss, flat_mask)
    if typ == specs.Type.CATEGORICAL:
        target = target.long()
        flat_mask = mask.reshape(-1).float()
        safe_target = target.reshape(-1).clone()
        safe_target[flat_mask == 0] = 0
        loss = F.cross_entropy(
            prediction.reshape(-1, prediction.shape[-1]),
            safe_target,
            reduction='none',
        )
        return _weighted_mean(loss, flat_mask)
    if typ == specs.Type.SCALAR:
        safe_target = torch.where(
            mask > 0, target.float(), torch.zeros_like(target.float()))
        loss = F.mse_loss(
            prediction, safe_target, reduction='none')
        return _weighted_mean(loss, mask)
    raise NotImplementedError(f'Unsupported loss type: {typ}')


def compute_losses(model_output, batch):
    hints = batch['hints']
    spec = batch['spec']
    hint_losses = {}
    for name, preds in model_output['hint_preds'].items():
        typ = spec[name][2]
        step_losses = []
        for step, prediction in enumerate(preds):
            target_step = step + 1
            if target_step >= hints[name]['target'].shape[1]:
                break
            target = hints[name]['target'][:, target_step]
            mask = hints[name]['mask'][:, target_step]
            if mask.sum() == 0:
                continue
            step_losses.append(_step_loss(
                prediction, target, mask, typ))
        if step_losses:
            hint_losses[name] = torch.stack(step_losses).mean()

    output_losses = {}
    for name, prediction in model_output['output_preds'].items():
        typ = spec[name][2]
        output = batch['outputs'][name]
        output_losses[name] = _step_loss(
            prediction, output['target'], output['mask'], typ)

    total_hint = (torch.stack(list(hint_losses.values())).sum()
                  if hint_losses else
                  next(iter(model_output['output_preds'].values())).new_tensor(0.0))
    total_output = (torch.stack(list(output_losses.values())).sum()
                    if output_losses else total_hint.new_tensor(0.0))
    return {
        'total': total_hint + total_output,
        'hint': total_hint,
        'output': total_output,
        'hint_losses': hint_losses,
        'output_losses': output_losses,
    }


def _accuracy(prediction, target, mask, typ):
    if typ == specs.Type.POINTER:
        pred = prediction.argmax(dim=-1)
    elif typ == specs.Type.MASK:
        pred = (prediction > 0).float()
        target = target.float()
    elif typ in (specs.Type.MASK_ONE, specs.Type.CATEGORICAL):
        pred = prediction.argmax(dim=-1)
    elif typ == specs.Type.SCALAR:
        return None
    else:
        return None
    correct = (pred == target).float() * mask.float()
    return correct.sum(), mask.float().sum()


def compute_metrics(model_output, batch):
    hints = batch['hints']
    spec = batch['spec']
    correct = 0.0
    total = 0.0
    hint_correct = 0.0
    hint_total = 0.0

    for name, preds in model_output['hint_preds'].items():
        typ = spec[name][2]
        for step, prediction in enumerate(preds):
            target_step = step + 1
            if target_step >= hints[name]['target'].shape[1]:
                break
            result = _accuracy(
                prediction, hints[name]['target'][:, target_step],
                hints[name]['mask'][:, target_step], typ)
            if result is not None:
                c, t = result
                hint_correct += c.item()
                hint_total += t.item()

    for name, prediction in model_output['output_preds'].items():
        typ = spec[name][2]
        output = batch['outputs'][name]
        result = _accuracy(
            prediction, output['target'], output['mask'], typ)
        if result is not None:
            c, t = result
            correct += c.item()
            total += t.item()

    gate_values = model_output.get('gate_values')
    gate_mean = None
    if gate_values:
        gate_mean = torch.stack(
            [g.float().mean() for g in gate_values]).mean().item()

    return {
        'output_accuracy': correct / max(total, 1.0),
        'hint_accuracy': hint_correct / max(hint_total, 1.0),
        'gate_mean': gate_mean,
    }


def train_step(model, optimizer, batch, teacher_forcing_ratio=0.5):
    model.train()
    batch = _move_to_device(batch, next(iter(batch['inputs'].values())).device)
    optimizer.zero_grad()
    output = model(
        batch['inputs'],
        batch['hints'],
        lengths=batch['lengths'],
        teacher_forcing_ratio=teacher_forcing_ratio,
    )
    losses = compute_losses(output, batch)
    losses['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return losses


@torch.no_grad()
def evaluate(model, adapter, batch_size, device):
    model.eval()
    batch = _move_to_device(adapter.get_batch(batch_size), device)
    output = model(
        batch['inputs'],
        batch['hints'],
        lengths=batch['lengths'],
        teacher_forcing_ratio=0.0,
    )
    losses = compute_losses(output, batch)
    metrics = compute_metrics(output, batch)
    metrics.update({
        'loss': losses['total'].item(),
        'hint_loss': losses['hint'].item(),
        'output_loss': losses['output'].item(),
    })
    return metrics


def save_checkpoint(path, model, optimizer, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
    }, path)


def train_spec(model, train_adapter, valid_adapter, test_adapter,
               steps=10000, batch_size=32, learning_rate=0.001,
               teacher_forcing_ratio=0.5, eval_every=500, test_every=0,
               checkpoint_path=None, log_path=None):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_valid = -1.0
    best_state = None
    start = time.time()
    log_file = open(log_path, 'a') if log_path else None

    def log(message):
        print(message, flush=True)
        if log_file is not None:
            log_file.write(message + '\n')
            log_file.flush()

    for step in range(1, steps + 1):
        batch = train_adapter.get_batch(batch_size)
        batch = _move_to_device(batch, device)
        optimizer.zero_grad()
        output = model(
            batch['inputs'],
            batch['hints'],
            lengths=batch['lengths'],
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        losses = compute_losses(output, batch)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            log(f'step {step}/{steps} loss={losses["total"].item():.5f} '
                f'hint={losses["hint"].item():.5f} '
                f'output={losses["output"].item():.5f} '
                f'time={time.time() - start:.1f}s')

        if step % eval_every == 0 or step == steps:
            valid = evaluate(model, valid_adapter, 32, device)
            test = None
            if test_every and (step % test_every == 0 or step == steps):
                test = evaluate(model, test_adapter, 32, device)
            record = {
                'step': step,
                'train_loss': losses['total'].item(),
                'valid': valid,
                'test': test,
            }
            history.append(record)
            test_message = (f' test_out={test["output_accuracy"]:.4f}'
                            f' test_hint={test["hint_accuracy"]:.4f}'
                            if test is not None else '')
            log(f'[eval] step={step} '
                f'valid_out={valid["output_accuracy"]:.4f}{test_message}')
            if valid['output_accuracy'] > best_valid:
                best_valid = valid['output_accuracy']
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            if checkpoint_path:
                save_checkpoint(checkpoint_path, model, optimizer, step)

    if best_state is not None:
        model.load_state_dict(best_state)
    final_valid = evaluate(model, valid_adapter, 32, device)
    final_test = evaluate(model, test_adapter, 32, device)
    if log_file is not None:
        log_file.close()
    return {
        'history': history,
        'best_valid': best_valid,
        'valid': final_valid,
        'test': final_test,
    }
