"""CNN inference (FlowNetPyramidal, cnn/FlowNet.py) on single reference/sample pairs read from .txt.

Preprocessing as in training: joint min-max normalisation of the pair to [0, 1]; output 'flow_l1' = (d_x, d_y) in
detector pixels, physical sign. CPU is enough (about 1-5 s per 256 x 256 pair).

Usage (environment with torch, see environment.yml):
    python run_cnn.py --dataset sphere                  # every pair -> results/sphere/cnn_CNN/pair_KK_{dx,dy}.txt
    python run_cnn.py --dataset shepp_logan --model finetuned_best_model
    python run_cnn.py --dataset sphere --diagnostic     # synthetic inputs of make_diagnostic_inputs.py
"""
import argparse, glob, os, sys, time
import numpy as np
import torch
from common import CNN_DIR, GEOM, RESULTS, load_pair, save_map

sys.path.insert(0, CNN_DIR)
from FlowNet import FlowNetPyramidal  # noqa: E402


def load_model(name):
    m = FlowNetPyramidal(max_disp=2, use_cbam=False)
    ck = torch.load(os.path.join(CNN_DIR, 'checkpoints', name + '.pt'), map_location='cpu')
    if isinstance(ck, dict):
        for key in ('model_state', 'state_dict', 'model_state_dict'):
            if key in ck:
                ck = ck[key]; break
    m.load_state_dict({k.replace('module.', ''): v for k, v in ck.items()})
    return m.eval()


def predict(model, ref, sam):
    lo, hi = min(ref.min(), sam.min()), max(ref.max(), sam.max())
    x = torch.from_numpy(np.stack([(ref - lo), (sam - lo)])[None].astype(np.float32) / np.float32(hi - lo + 1e-8))
    with torch.no_grad():
        f = model(x[:, 0:1], x[:, 1:2])['flow_l1'][0].numpy()
    return f[0].astype(float), f[1].astype(float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(GEOM), required=True)
    p.add_argument('--model', default='CNN', help='checkpoint name in cnn/checkpoints (CNN or finetuned_best_model)')
    p.add_argument('--diagnostic', action='store_true')
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    model = load_model(a.model)
    out = os.path.join(RESULTS, a.dataset, f'cnn_{a.model}')
    if a.diagnostic:
        d = os.path.join(RESULTS, a.dataset, 'diagnostic')
        ref = np.loadtxt(os.path.join(d, 'inputs', 'reference.txt'))
        for f in sorted(glob.glob(os.path.join(d, 'inputs', 'sample_*.txt'))):
            case = os.path.basename(f)[7:-4]
            dx, dy = predict(model, ref, np.loadtxt(f))
            save_map(dx, d, f'cnn_{a.model}_{case}_dx.txt'); save_map(dy, d, f'cnn_{a.model}_{case}_dy.txt')
            print('diagnostic', case, flush=True)
        return
    times = []
    for k in GEOM[a.dataset]['pairs']:
        ref, sam = load_pair(a.dataset, k)
        t = time.time(); dx, dy = predict(model, ref, sam); times.append(time.time() - t)
        save_map(dx, out, f'pair_{k:02d}_dx.txt'); save_map(dy, out, f'pair_{k:02d}_dy.txt')
        print(f'pair {k:02d}: {times[-1]:.2f} s', flush=True)
    with open(os.path.join(out, 'timing.txt'), 'w') as f:
        f.write(f'CPU inference time per pair, {a.threads} threads: mean {np.mean(times):.2f} s, '
                f'median {np.median(times):.2f} s (n = {len(times)})\n')


if __name__ == '__main__':
    main()
