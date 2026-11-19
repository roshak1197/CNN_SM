"""Single-shot comparison (section S8): the CNN of this work and two published networks on one pair at a time.
  CNN        cnn/checkpoints/CNN.pt, joint min-max normalisation of the pair (input padded to a multiple of 32)
  StrainNet  StrainNet-h (Boukhtache et al. 2021; optical DIC), input in [0, 1] (joint min-max), following their
             inference.py: half-resolution output upsampled x2, d = -out * div_flow + 1
  ESPINNet   Qiao et al. 2026, weights training_model_001500.pt. The network takes exactly 20 reference + 20 sample
             frames; with ONE pair the pair is replicated 20 times (its designed regime is N >= 2: this column shows
             what happens when it is used single-shot; SPINNet, the single-shot version, has no public weights).
             Input divided by mean(reference). Multi-frame ESPINNet on the real datasets (all pairs; MIST: 15 frames
             mapped to 20 channels by repetition) is also computed, as a reference of what the network does in its regime.
Datasets: sphere, shepp_logan (this work) and mist, fly, espinnet, star (third party, prepare_third_party.py).
Output: results/comparison/<ds>/<method>/pair_KK_{dx,dy}.txt.gz in px, raw sign of each network (the sign convention is
aligned once per method and dataset in comparison.py). Needs torch and the third-party code (third_party/).
Usage: python run_networks.py [datasets...]
"""
import json, os, sys, time, types
import numpy as np
import torch
import torch.nn.functional as F
from common import GEOM, RESULTS, THIRD, THIRD_DIR, load_pair_any
from run_cnn import load_model
from run_networks_io import save

torch.set_num_threads(int(os.environ.get('THREADS', 16)))
sys.modules.setdefault('h5py', types.ModuleType('h5py'))          # imported but unused by ESPINNet's estimate_single
sys.path.insert(0, os.path.join(THIRD_DIR, 'StrainNet', 'StrainNet'))
sys.path.insert(0, os.path.join(THIRD_DIR, 'ESPINNet'))
import models as sn_models                                          # noqa: E402  (StrainNet)
from DS_network import Network                                      # noqa: E402  (ESPINNet)
from estimate_single import args_setting, estimate                  # noqa: E402

CNN = load_model('CNN')
_sn = torch.load(os.path.join(THIRD_DIR, 'StrainNet', 'pretrained_download', 'StrainNet', 'StrainNet-h',
                              'StrainNet-h.pth.tar'), map_location='cpu', weights_only=True)
SN, SN_DIV = sn_models.__dict__['StrainNet_h'](_sn).eval(), float(_sn['div_flow'])
_st = json.load(open(os.path.join(THIRD_DIR, 'ESPINNet', 'trained_model', 'setting_001500.json')))
_st['device'], _st['fp16'] = 'cpu', False
ESP_ARGS = args_setting(_st)
ESP = Network(ESP_ARGS)
ESP.load_state_dict(torch.load(os.path.join(THIRD_DIR, 'ESPINNet', 'trained_model', 'training_model_001500.pt'),
                               map_location='cpu')['model_state_dict'])
ESP.eval()


def minmax(r, s):
    lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
    return (r - lo) / (hi - lo), (s - lo) / (hi - lo)


@torch.no_grad()
def cnn(r, s):
    a, b = minmax(r, s); H, W = r.shape
    x = F.pad(torch.from_numpy(np.stack([a, b])[None].astype(np.float32)), (0, (-W) % 32, 0, (-H) % 32), mode='reflect')
    return CNN(x[:, 0:1], x[:, 1:2])['flow_l1'][0].numpy()[:, :H, :W]


@torch.no_grad()
def strainnet(r, s):
    a, b = minmax(r, s); H, W = r.shape
    t = lambda z: F.pad(torch.from_numpy(np.ascontiguousarray(z, np.float32))[None, None], (0, (-W) % 64, 0, (-H) % 64),
                        mode='reflect').repeat(1, 3, 1, 1)
    o = F.interpolate(SN(torch.cat([t(a), t(b)], 1)), scale_factor=2, mode='bilinear')[0, :, :H, :W].numpy()
    return np.array([-o[0] * SN_DIV + 1.0, -o[1] * SN_DIV + 1.0])


def espinnet_stack(R, S):
    """R, S: (20, H, W) frames."""
    x = torch.from_numpy((np.concatenate([R, S]) / R.mean())[None].astype(np.float32))
    fl, _, _, _ = estimate(x, ESP, ESP_ARGS)
    return F.interpolate(torch.from_numpy(fl), size=R.shape[1:], mode='bilinear', align_corners=True).numpy()[0]


def espinnet1(r, s):
    return espinnet_stack(np.repeat(r[None], 20, 0), np.repeat(s[None], 20, 0))


METHODS = {'cnn': cnn, 'strainnet': strainnet, 'espinnet1': espinnet1}


def main():
    for ds in sys.argv[1:] or ['sphere', 'shepp_logan', 'star', 'mist', 'fly', 'espinnet']:
        pairs = (GEOM.get(ds) or THIRD[ds])['pairs']; t = time.time(); R_all, S_all = [], []
        for k in pairs:
            r, s = load_pair_any(ds, k); R_all.append(r); S_all.append(s)
            for m, f in METHODS.items():
                if m == 'cnn' and ds in GEOM:
                    continue                                    # already in results/<ds>/cnn_CNN (run_cnn.py)
                if not os.path.exists(os.path.join(RESULTS, 'comparison', ds, m, f'pair_{k:02d}_dy.txt.gz')):
                    save(f(r, s), ds, m, f'pair_{k:02d}')            # existing maps are kept (resumable)
            print(ds, k, f'{time.time() - t:.0f} s', flush=True)
        if ds in ('mist', 'fly', 'espinnet'):
            R_all, S_all = np.array(R_all), np.array(S_all)
            idx = np.round(np.linspace(0, len(pairs) - 1, 20)).astype(int)     # 15 -> 20 channels by repetition
            save(espinnet_stack(R_all[idx], S_all[idx]), ds, 'espinnetN', 'all_pairs')


if __name__ == '__main__':
    main()
