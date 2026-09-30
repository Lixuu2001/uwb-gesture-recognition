"""Actual-model control identity and interrupted/restarted style training check."""
import json
import numpy as np
import torch

from runner import T, HERE, BASE, OLD, style_epoch
from style import StyleReader


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    control = HERE/'smoke_control/CONTROL/fold1/warmup'
    reference = BASE/'mild_time_augmentation/smoke/CONTROL/fold1/warmup'
    enabled = HERE/'smoke_style/STYLE/fold1/warmup'
    result = {'control_reproduction': {}, 'smoke': {}}
    with np.load(control/'predictions.npz') as new, np.load(reference/'predictions.npz') as old:
        for key in ('logits', 'val_logits', 'ids', 'val_ids', 'y', 'val_y'):
            error = float(np.max(np.abs(new[key]-old[key])))
            if error != 0: raise ValueError(f'Disabled style differs from original smoke: {key}')
            result['control_reproduction'][key+'_max_abs_error'] = error
    old_result = json.loads((reference/'result.json').read_text(encoding='utf-8'))
    for mode, directory in [('CONTROL', control), ('STYLE', enabled)]:
        record = json.loads((directory/'result.json').read_text(encoding='utf-8'))
        if record['identity']['initial_state_sha256'] != old_result['identity']['initial_state_sha256']:
            raise ValueError('Smoke initial state differs')
        if record['history'][1]['episode_hash'] != old_result['history'][1]['episode_hash']:
            raise ValueError('Smoke sampler differs')
        result['smoke'][mode] = {'initial_state_sha256': record['identity']['initial_state_sha256'],
                                'sampling_hash': record['history'][1]['episode_hash'],
                                'style_counts': record['history'][1]['style_counts']}
    checkpoint = enabled/'progress.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = saved['info']['identity']; config = identity['config']
    raw_reader, y, subjects, parts = T.protocol.open_fold(1)
    reader = StyleReader(raw_reader, y, subjects, enabled=True)
    matrix, _ = T.smoothing.prepare_matrix(1, OLD/'matrices', identity['source']['data'])
    matrix = torch.as_tensor(matrix, device='cuda')
    criterion = lambda logits, target: T.smoothing.classification_loss(logits, target, matrix)
    model = T.models.build_model(config['seed'], 'cuda')
    optimizer = T.optimizer_for(model, 'warmup', config)
    sampler = T.state.ShuffleSampler(parts['train'], config['seed']+1)
    T.state.load_progress(checkpoint, model, optimizer, sampler, identity)
    style_epoch(model, optimizer, sampler, reader, y, criterion, config)
    recovery_path = HERE/'smoke_style/recovery_checkpoint.pt'
    T.state.save_progress(recovery_path, model, optimizer, sampler, {'identity': identity})
    uninterrupted = style_epoch(model, optimizer, sampler, reader, y, criterion, config)
    hash_a = T.state.state_hash(model.state_dict())
    logits_a, _ = T.evaluate(model, reader, y, parts['val'], 'val', 32)
    del model, optimizer
    model = T.models.build_model(config['seed'], 'cuda')
    optimizer = T.optimizer_for(model, 'warmup', config)
    sampler = T.state.ShuffleSampler(parts['train'], config['seed']+1)
    reader = StyleReader(raw_reader, y, subjects, enabled=True)
    T.state.load_progress(recovery_path, model, optimizer, sampler, identity)
    resumed = style_epoch(model, optimizer, sampler, reader, y, criterion, config)
    hash_b = T.state.state_hash(model.state_dict())
    logits_b, _ = T.evaluate(model, reader, y, parts['val'], 'val', 32)
    if hash_a != hash_b or uninterrupted != resumed or not np.array_equal(logits_a, logits_b):
        raise ValueError('Style interrupted/restarted epoch differs')
    result['recovery'] = {'state_hash_equal': True, 'next_epoch_row_equal': True,
                          'validation_max_abs_error': float(np.abs(logits_a-logits_b).max()),
                          'uninterrupted_hash': hash_a, 'resumed_hash': hash_b,
                          'style_trace': resumed['style_trace'], 'style_counts': resumed['style_counts']}
    result['source_code'] = identity['source']['code']
    for path, digest in result['source_code'].items():
        if T.protocol.digest(path) != digest:
            raise ValueError('Training source changed since smoke execution')
    result['status'] = 'verified'
    T.protocol.dump(HERE/'PRETRAIN_CHECK.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
