"""Reproduce the numerical examples in the accompanying CNN blog post.

CPU-only by design. Timings describe this implementation and machine, not
universal complexity bounds. Input images are controlled pixel plots, not
Matplotlib screenshots. Both learned models receive the same quantization.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.benchmark import Timer

N = 128
LOW, HIGH = -2.0, 2.0
CHANNELS = (8, 16, 24)


def quantize(x: np.ndarray, height: int = 32):
    """Return cell-center values and row indices, with one fixed amplitude scale."""
    step = (HIGH - LOW) / height
    codes = np.floor((x - LOW) / step).clip(0, height - 1).astype(np.int64)
    values = (LOW + (codes + 0.5) * step).astype(np.float32)
    return values, codes


def pixel_plot(codes: np.ndarray, height: int = 32):
    """Shape (batch, 1, height, time); amplitude increases with the row index."""
    result = np.zeros((len(codes), 1, height, codes.shape[-1]), np.float32)
    result[np.arange(len(codes))[:, None], 0, codes,
           np.arange(codes.shape[-1])[None, :]] = 1.0
    return result


class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        layers, previous = [], 1
        for current in CHANNELS:
            layers.extend([nn.Conv1d(previous, current, 5, padding=2),
                           nn.ReLU(), nn.MaxPool1d(2)])
            previous = current
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(CHANNELS[-1], 2)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


class CNN2D(nn.Module):
    def __init__(self):
        super().__init__()
        layers, previous = [], 1
        for current in CHANNELS:
            layers.extend([nn.Conv2d(previous, current, (3, 5), padding=(1, 2)),
                           nn.ReLU(), nn.MaxPool2d(2)])
            previous = current
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(CHANNELS[-1], 2)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


class ExactImageSimulation(nn.Module):
    """Fixed pixel decoder followed by an exact height-one CNN simulation."""
    def __init__(self, original: CNN1D, height: int):
        super().__init__()
        self.decoder = nn.Conv2d(1, 1, (height, 1), bias=False)
        centers = LOW + (torch.arange(height) + 0.5) * ((HIGH - LOW) / height)
        with torch.no_grad():
            self.decoder.weight.copy_(centers.reshape(1, 1, height, 1))
        self.decoder.requires_grad_(False)
        converted = []
        for layer in original.features:
            if isinstance(layer, nn.Conv1d):
                new = nn.Conv2d(layer.in_channels, layer.out_channels,
                                (1, layer.kernel_size[0]),
                                padding=(0, layer.padding[0]))
                with torch.no_grad():
                    new.weight.copy_(layer.weight.unsqueeze(-2))
                    new.bias.copy_(layer.bias)
                converted.append(new)
            elif isinstance(layer, nn.MaxPool1d):
                converted.append(nn.MaxPool2d((1, 2)))
            else:
                converted.append(copy.deepcopy(layer))
        self.features = nn.Sequential(*converted)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = copy.deepcopy(original.head)

    def forward(self, z):
        return self.head(self.pool(self.features(self.decoder(z))).flatten(1))


def make_dataset(task: str, count: int, seed: int):
    assert count % 2 == 0
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(2), count // 2)
    rng.shuffle(labels)
    t = np.arange(N, dtype=np.float64) / N
    rows = []
    for label in labels:
        phase = rng.uniform(0, 2 * np.pi)
        if task == 'oscillations':
            frequency = rng.uniform(2, 3) if label == 0 else rng.uniform(5, 6)
            amplitude = rng.uniform(0.6, 1.2)
            curve = amplitude * np.sin(2 * np.pi * frequency * t + phase)
            curve += rng.uniform(-0.2, 0.2) + rng.normal(0, 0.08, N)
        elif task == 'transients':
            center, amplitude = rng.uniform(0.3, 0.7), rng.uniform(0.8, 1.3)
            if label == 0:
                signal = np.exp(-0.5 * ((t - center) / 0.06) ** 2)
            else:
                signal = (np.exp(-0.5 * ((t - center - 0.055) / 0.025) ** 2)
                          + np.exp(-0.5 * ((t - center + 0.055) / 0.025) ** 2))
            curve = amplitude * signal + 0.1 * np.sin(4 * np.pi * t + phase)
            curve += rng.uniform(-0.15, 0.15) + rng.normal(0, 0.06, N)
        else:
            raise ValueError(task)
        rows.append(curve)
    x = np.asarray(rows, dtype=np.float32)
    if not np.all((x >= LOW) & (x < HIGH)):
        raise ValueError('Generator exceeded fixed amplitude range; no silent clipping allowed')
    return x, labels.astype(np.int64)


def representations(x, height):
    q, codes = quantize(x, height)
    z = pixel_plot(codes, height)
    return torch.from_numpy(q[:, None]), torch.from_numpy(z)


def describe(model, example):
    records = []
    def hook(module, inputs, output):
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            per_position = module.in_channels // module.groups * int(np.prod(module.kernel_size))
            macs = output.numel() // output.shape[0] * per_position
        else:
            macs = module.in_features * module.out_features
        records.append({'layer':type(module).__name__, 'output_shape':list(output.shape[1:]),
                        'macs':int(macs),
                        'output_bytes_per_series':output.numel() // output.shape[0] * output.element_size()})
    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear))]
    with torch.inference_mode():
        model(example)
    for handle in handles:
        handle.remove()
    return {'parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'input_bytes':example[0].numel()*example.element_size(),
            'macs':sum(r['macs'] for r in records),
            'conv_macs':sum(r['macs'] for r in records if r['layer'].startswith('Conv')),
            'layers':records}


def latency(model, x, threads):
    model.eval()
    with torch.inference_mode():
        for _ in range(8):
            model(x)
        measurement = Timer(stmt='model(x)', globals={'model':model,'x':x},
                            num_threads=threads).blocked_autorange(min_run_time=0.4)
    return {'median_ms':measurement.median*1000, 'iqr_ms':measurement.iqr*1000,
            'batch_size':len(x),'amortized_ms_per_series':measurement.median*1000/len(x),
            'measurements':len(measurement.raw_times),
            'calls_per_measurement':measurement.number_per_run}


def evaluate(model, x, y):
    model.eval()
    with torch.inference_mode():
        logits = torch.cat([model(x[s:s+64]) for s in range(0,len(x),64)])
        loss = nn.functional.cross_entropy(logits, y).item()
    predictions = logits.argmax(1).numpy()
    return {'loss':loss, 'accuracy':accuracy_score(y.numpy(), predictions),
            'macro_f1':f1_score(y.numpy(), predictions, average='macro')}, predictions


def train_one(model, x, y, xv, yv, xt, yt, seed, epochs, batch):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    generator = torch.Generator().manual_seed(seed + 1000)
    history, best_state, best_loss, best_epoch = [], None, float('inf'), None
    training_seconds = 0.0
    whole_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        total_loss = 0.0
        start = time.perf_counter()
        for begin in range(0, len(x), batch):
            ids = order[begin:begin+batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[ids])
            loss = nn.functional.cross_entropy(logits, y[ids])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(ids)
        training_seconds += time.perf_counter() - start
        validation, _ = evaluate(model, xv, yv)
        history.append({'epoch':epoch,'training_loss':total_loss/len(x),
                        'validation_loss':validation['loss'],
                        'validation_accuracy':validation['accuracy']})
        if validation['loss'] < best_loss:
            best_loss, best_epoch = validation['loss'], epoch
            best_state = copy.deepcopy(model.state_dict())
    selection_seconds = time.perf_counter() - whole_start
    model.load_state_dict(best_state)
    test, predictions = evaluate(model, xt, yt)
    return {'best_epoch':best_epoch,'training_seconds':training_seconds,
            'training_and_selection_seconds':selection_seconds,
            'test':test,'history':history}, predictions


def exact_checks(height=32):
    torch.manual_seed(71)
    x, y = make_dataset('oscillations', 16, 9001)
    q, z = representations(x, height)
    original = CNN1D()
    simulated = ExactImageSimulation(original, height)
    lo, ls = original(q), simulated(z)
    decoded = simulated.decoder(z).squeeze(-2)
    nn.functional.cross_entropy(lo, torch.from_numpy(y)).backward()
    nn.functional.cross_entropy(ls, torch.from_numpy(y)).backward()
    errors = []
    for (name, p), (name2, p2) in zip(original.named_parameters(),
            [(n,p) for n,p in simulated.named_parameters() if p.requires_grad]):
        g2 = p2.grad.squeeze(-2) if p2.grad.ndim == p.grad.ndim+1 else p2.grad
        errors.append(float((p.grad-g2).abs().max()))
    result = {'max_reconstruction_error':float((decoded-q).abs().max()),
              'max_logit_error':float((lo-ls).abs().max()),
              'max_parameter_gradient_error':max(errors),
              'same_predictions':bool(torch.equal(lo.argmax(1),ls.argmax(1)))}
    assert result['max_reconstruction_error'] == 0
    assert result['max_logit_error'] < 1e-6
    assert result['max_parameter_gradient_error'] < 1e-6
    assert result['same_predictions']
    # Two equally likely constant curves in the same quantization cell.
    center = LOW + (height//2+0.5)*(HIGH-LOW)/height
    eps = (HIGH-LOW)/height/10
    raw = np.vstack([np.full(N,center-eps),np.full(N,center+eps)]).astype(np.float32)
    q2, codes = quantize(raw,height)
    result['collision'] = {'raw_separation':float(raw[1,0]-raw[0,0]),
                           'same_pixel_plot':bool(np.array_equal(pixel_plot(codes,height)[0],
                                                                  pixel_plot(codes,height)[1])),
                           'raw_bayes_error':0.0,'image_bayes_error':0.5,
                           'interpretation':'Exact two-point distribution, not an estimated CNN error'}
    assert result['collision']['same_pixel_plot']
    return result


def environment(threads):
    info = {'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
            'platform':platform.platform(),'logical_cpus_visible':os.cpu_count(),
            'torch_threads':threads,'device':'cpu','dtype':'float32','mkldnn_enabled':torch.backends.mkldnn.enabled}
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                info['cpu_model'] = line.split(':',1)[1].strip()
                break
    except OSError:
        pass
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    parser.add_argument('--epochs',type=int,default=25)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    configuration = {'length':N,'height':32,'amplitude_range':[LOW,HIGH],
                     'n_train':512,'n_validation':128,'n_test':512,'epochs':args.epochs,
                     'seeds':args.seeds,'batch_size':64,'channels':list(CHANNELS),
                     'learning_rate':0.003,'data_seed_base':20261121,
                     'checkpoint_selection':'minimum validation cross entropy',
                     'timing_scope':'training excludes validation; latency excludes encoding'}
    payload = {'environment':environment(args.threads),'configuration':configuration,
               'exact_checks':exact_checks(),'benchmarks':[],'training_runs':[],
               'preprocessing':[],'data_hashes':{}}
    (args.output/'exact_checks.json').write_text(json.dumps(payload['exact_checks'],indent=2))
    print('Exact representation and gradient checks passed',flush=True)
    # Independent untrained timing sweep; warm-up, then randomized configuration order.
    configs = [('1d',None)]+[('2d',h) for h in [16,32,64]]
    np.random.default_rng(19).shuffle(configs)
    example_raw,_ = make_dataset('oscillations',64,8080)
    for name,height in configs:
        torch.manual_seed(42)
        model = CNN1D() if name=='1d' else CNN2D()
        q,z = representations(example_raw,height or 32)
        x = q if name=='1d' else z
        row = {'model':name,'height':height,'description':describe(model,x[:1]),
               'batch1':latency(model,x[:1],args.threads),
               'batch64':latency(model,x,args.threads)}
        payload['benchmarks'].append(row)
        print('Benchmark',name,height,row['batch1']['median_ms'],flush=True)
    for task_index,task in enumerate(['oscillations','transients']):
        partitions = [make_dataset(task,count,20261121+100*task_index+offset)
                      for count,offset in [(512,0),(128,1),(512,2)]]
        for split,(raw,labels) in zip(['train','validation','test'],partitions):
            digest=hashlib.sha256(raw.tobytes()+labels.tobytes()).hexdigest()
            payload['data_hashes'][task+'/'+split]=digest
        all_q,all_z,all_y=[],[],[]
        for split,(raw,labels) in zip(['train','validation','test'],partitions):
            # Repeated independent timing calls on the same in-memory arrays.
            q_times,z_times=[],[]
            for _ in range(9):
                begin=time.perf_counter(); q,codes=quantize(raw,32)
                q_times.append(time.perf_counter()-begin)
                begin=time.perf_counter(); z=pixel_plot(codes,32)
                z_times.append(time.perf_counter()-begin)
            payload['preprocessing'].append({'task':task,'split':split,'count':len(raw),
                'quantize_ms':float(np.median(q_times)*1000),
                'additional_image_allocation_ms':float(np.median(z_times)*1000)})
            all_q.append(torch.from_numpy(q[:,None])); all_z.append(torch.from_numpy(z))
            all_y.append(torch.from_numpy(labels))
        run_order=[(seed,name) for seed in args.seeds for name in ['1d','2d']]
        np.random.default_rng(50+task_index).shuffle(run_order)
        for seed,name in run_order:
            torch.manual_seed(seed)
            model=CNN1D() if name=='1d' else CNN2D()
            data=all_q if name=='1d' else all_z
            result,predictions=train_one(model,data[0],all_y[0],data[1],all_y[1],
                data[2],all_y[2],seed,args.epochs,64)
            row={'task':task,'model':name,'seed':seed,**result}
            payload['training_runs'].append(row)
            np.savez_compressed(args.output/f'predictions_{task}_{name}_{seed}.npz',
                                y=all_y[2].numpy(),predictions=predictions)
            print(task,name,seed,'test',round(result['test']['accuracy'],4),
                  'train_s',round(result['training_seconds'],2),flush=True)
            (args.output/'results.json').write_text(json.dumps(payload,indent=2))
    rows=[]
    for run in payload['training_runs']:
        rows.append({k:run[k] for k in ['task','model','seed','best_epoch','training_seconds',
                                       'training_and_selection_seconds']} | run['test'])
    with (args.output/'training_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/'results.json').write_text(json.dumps(payload,indent=2))
    print('Finished. Results:',args.output,flush=True)


if __name__=='__main__':
    main()
