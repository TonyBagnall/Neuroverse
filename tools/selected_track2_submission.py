"""Single-file MRHydra/QUANT inference, adapted from aeon 1.5.0.

Include LICENSE.txt (aeon BSD 3-Clause). No aeon/numba/sklearn inference imports.
The competition supplies NumPy, Torch and benchmark_utils. Fitted numeric
parameters are stored in weights.npz; no pickles or additional Python modules.
"""
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from benchmark_utils.base_solver import CompetSolver

_EXPECTED_CHANS = 41
_EXPECTED_TIMES = 480


def _norm_ch(name):
    s = str(name).strip()
    if s[:4].upper() == 'EEG ':
        s = s[4:].strip()
    return s.split('-', 1)[0].strip().upper()


def _build_mr_cache(p, device='cpu'):
    dev = torch.device(device)
    indices = list(combinations(range(9), 3))
    counts = p['mr_counts']
    channels = p['mr_channels']
    n = int(len(p['mr_biases']))
    n1 = int(len(p['diff_biases']))
    if n != n1:
        raise ValueError('Unsupported unequal raw/difference feature layout')
    passes = []
    max_len = _EXPECTED_TIMES
    for diff, prefix in enumerate(('mr_', 'diff_')):
        dilations = [int(d) for d in p[prefix + 'dilations']]
        features = [int(f) for f in p[prefix + 'features']]
        biases = torch.as_tensor(p[prefix + 'biases'], dtype=torch.float32, device=dev)
        combo = 0
        channel_start = 0
        feat_start = 0
        dil_entries = []
        for j, (d, num) in enumerate(zip(dilations, features)):
            w = np.zeros((84, 10, _EXPECTED_CHANS), dtype=np.float32)
            for k, (i0, i1, i2) in enumerate(indices):
                channel_end = channel_start + int(counts[combo])
                pick = channels[channel_start:channel_end]
                np.add.at(w[k, 0], pick, 1.0)
                np.add.at(w[k, 1 + i0], pick, 1.0)
                np.add.at(w[k, 1 + i1], pick, 1.0)
                np.add.at(w[k, 1 + i2], pick, 1.0)
                combo += 1
                channel_start = channel_end
            feat_end = feat_start + 84 * num
            b_dil = biases[feat_start:feat_end].view(84, num)
            dil_entries.append({
                'j_mod2': j % 2,
                'd': d,
                'padding': 4 * d,
                'num': num,
                'w': torch.from_numpy(w.reshape(84, 10 * _EXPECTED_CHANS)).to(dev),
                'b_even': b_dil[0::2].contiguous().view(1, 42, num, 1),
                'b_odd': b_dil[1::2].contiguous().view(1, 42, num, 1),
            })
            feat_start = feat_end
        passes.append({'diff': diff, 'dilations': dil_entries})
    return {
        'device': dev,
        'n': n,
        'n1': n1,
        'passes': passes,
        'ones': torch.ones(max_len, dtype=torch.float32, device=dev),
        'arange_f32': torch.arange(max_len, dtype=torch.float32, device=dev),
        'arange_i16': torch.arange(max_len, dtype=torch.int16, device=dev),
        'zero_i16': torch.tensor(0, dtype=torch.int16, device=dev),
    }


def _pool_stats(sc, sb, ones, arange_f32, arange_i16, zero_i16):
    # sc: (B, 42, L), sb: (1, 42, num, 1)
    B, K, L = sc.shape
    num = sb.shape[2]
    sc4 = sc.unsqueeze(2)
    pf = (sc4 > sb).to(torch.float32)
    pc = torch.matmul(pf, ones[:L])
    ls = torch.matmul(pf, arange_f32[:L])
    pf.mul_(sc4)
    cs = pf.sum(dim=-1)

    ppv = pc * (1.0 / L)
    has_pos = pc > 0
    denom = pc.clamp_min(1.0)
    mpv = torch.where(has_pos, cs / denom + sb.squeeze(-1), 0.0)
    mipv = torch.where(has_pos, ls / denom, -1.0)

    if L <= 1:
        lspv = torch.zeros((B, K, num), dtype=torch.float32, device=sc.device)
    else:
        neg = (sc4 < sb).reshape(-1, L)
        ti = arange_i16[:L]
        cm = torch.where(neg, ti, zero_i16)
        step = 1
        while step < L:
            cm[:, step:] = torch.maximum(cm[:, step:], cm[:, :-step])
            step *= 2
        mg = torch.where(neg[:, 1:], ti[1:] - cm[:, :-1], zero_i16).amax(dim=-1)
        lspv = torch.maximum(mg, (L - 1) - cm[:, -1]).view(B, K, num).to(torch.float32)
    return ppv, lspv, mpv, mipv


@torch.inference_mode()
def _multi_rocket_torch(x, cache):
    B, C, orig_T = x.shape
    device = x.device
    ones = cache['ones']
    ar_f32 = cache['arange_f32']
    ar_i16 = cache['arange_i16']
    z_i16 = cache['zero_i16']
    if orig_T > ones.shape[0]:
        ones = torch.ones(orig_T, dtype=torch.float32, device=device)
        ar_f32 = torch.arange(orig_T, dtype=torch.float32, device=device)
        ar_i16 = torch.arange(orig_T, dtype=torch.int16, device=device)
    all_parts = []
    for pass_info in cache['passes']:
        series = x if pass_info['diff'] == 0 else torch.diff(x, dim=-1)
        points = series.shape[-1]
        a = -series
        g = series + series + series
        ppv_list, lspv_list, mpv_list, mipv_list = [], [], [], []
        for entry in pass_info['dilations']:
            d = entry['d']
            padding = entry['padding']
            num = entry['num']
            ag = torch.zeros((B, 10, C, points), dtype=torch.float32, device=device)
            alpha = ag[:, 0]
            alpha.copy_(a)
            ag[:, 5] = g
            left = d
            end = orig_T - padding  # matches aeon source using original length even for diff
            for idx in range(4):
                alpha[:, :, -end:] = alpha[:, :, -end:] + a[:, :, :end]
                ag[:, 1 + idx, :, -end:] = g[:, :, :end]
                end += d
            for idx in range(5, 9):
                alpha[:, :, :-left] = alpha[:, :, :-left] + a[:, :, left:]
                ag[:, 1 + idx, :, :-left] = g[:, :, left:]
                left += d
            c_all = torch.matmul(entry['w'], ag.view(B, 10 * C, points))
            c_even = c_all[:, 0::2, :]
            c_odd = c_all[:, 1::2, :]
            if entry['j_mod2'] == 0:
                c_odd = c_odd[:, :, padding:-padding]
            else:
                c_even = c_even[:, :, padding:-padding]
            s_even = _pool_stats(c_even, entry['b_even'], ones, ar_f32, ar_i16, z_i16)
            s_odd = _pool_stats(c_odd, entry['b_odd'], ones, ar_f32, ar_i16, z_i16)
            for lst, se, so in zip((ppv_list, lspv_list, mpv_list, mipv_list), s_even, s_odd):
                lst.append(torch.stack((se, so), dim=2).reshape(B, 84 * num))
        all_parts.extend((
            torch.cat(ppv_list, dim=1),
            torch.cat(lspv_list, dim=1),
            torch.cat(mpv_list, dim=1),
            torch.cat(mipv_list, dim=1),
        ))
    out = torch.cat(all_parts, dim=1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def multi_rocket(x, p):
    cache = p.get('_mr_cache')
    if cache is None:
        cache = _build_mr_cache(p, 'cpu')
    t = torch.as_tensor(x, dtype=torch.float32, device=cache['device'])
    return _multi_rocket_torch(t, cache).cpu().numpy()


@torch.inference_mode()
def _hydra_torch(x, p):
    difference = x.diff(dim=-1)
    weights = p['hydra_weights']
    index = p['hydra_index']
    parts = []
    h = int(p['hydra_h'])
    k = int(p['hydra_k'])
    B = x.shape[0]
    dev = x.device
    for j, d in enumerate(p['hydra_dilations']):
        d = int(d)
        for which in range(weights.shape[1]):
            series = x if which == 0 else difference
            values = F.conv1d(
                series[:, index[j, which]].sum(2),
                weights[j, which],
                dilation=d,
                padding=4 * d,
                groups=h,
            ).view(B, h, k, -1)
            maximum, max_index = values.max(2)
            minimum, min_index = values.min(2)
            count_max = torch.zeros((B, h, k), dtype=torch.float32, device=dev)
            count_min = torch.zeros_like(count_max)
            count_max.scatter_add_(-1, max_index, maximum)
            count_min.scatter_add_(-1, min_index, torch.ones_like(minimum))
            parts.extend((count_max, count_min))
    output = torch.cat(parts, 1).view(B, -1).clamp(0).sqrt()
    return ((output - p['hydra_mu']) * (output != 0)) / p['hydra_sigma']


def hydra(x, p):
    weights = p['hydra_weights']
    dev = weights.device if isinstance(weights, torch.Tensor) else torch.device('cpu')
    p_local = p
    if not isinstance(weights, torch.Tensor):
        p_local = dict(p)
        p_local['hydra_weights'] = torch.as_tensor(p['hydra_weights'], dtype=torch.float32, device=dev)
        p_local['hydra_index'] = torch.as_tensor(p['hydra_index'], dtype=torch.long, device=dev)
        p_local['hydra_mu'] = torch.as_tensor(p['hydra_mu'], dtype=torch.float32, device=dev)
        p_local['hydra_sigma'] = torch.as_tensor(p['hydra_sigma'], dtype=torch.float32, device=dev)
    elif p['hydra_index'].dtype != torch.long:
        p_local = dict(p)
        p_local['hydra_index'] = p['hydra_index'].to(dtype=torch.long)
    t = torch.as_tensor(x, dtype=torch.float32, device=dev)
    return _hydra_torch(t, p_local).cpu().numpy()


@torch.inference_mode()
def quant_features(x, p):
    x = torch.as_tensor(x, dtype=torch.float32, device='cpu')
    reps = [
        x,
        F.avg_pool1d(F.pad(x.diff(dim=-1), (2, 2), 'replicate'), 5, 1),
        x.diff(n=2, dim=-1),
        torch.fft.rfft(x).abs(),
    ]
    representations = []
    divisor = int(p['quant_divisor'])
    for i, z in enumerate(reps):
        features = []
        for a, b in p['quant_intervals_' + str(i)]:
            values = z[..., int(a):int(b)]
            length = values.shape[-1]
            if length == 1:
                q = values
            else:
                count = 1 + (length - 1) // divisor
                if count == 1:
                    q = values.quantile(torch.tensor([0.5]), dim=-1).permute(1, 2, 0)
                else:
                    q = values.quantile(torch.linspace(0, 1, count), dim=-1).permute(1, 2, 0)
                    q[..., 1::2] -= values.mean(-1, keepdim=True)
            features.append(q.squeeze(1))
        representations.append(torch.cat(features, -1))
    values = torch.cat(representations, -1)
    return values.numpy().reshape(len(x), -1)


def forest_probabilities(features, p):
    output = np.zeros((len(features), 3), dtype=np.float64)
    for tree in range(int(p['tree_count'])):
        prefix = 'tree_' + str(tree) + '_'
        left = p[prefix + 'left']
        right = p[prefix + 'right']
        feature = p[prefix + 'feature']
        threshold = p[prefix + 'threshold']
        values = p[prefix + 'value']
        for i, row in enumerate(features):
            node = 0
            while left[node] != -1:
                node = int(left[node] if row[feature[node]] <= threshold[node] else right[node])
            output[i] += values[node]
    return output / int(p['tree_count'])


class SelectedModel:
    def __init__(self, path, mode='ensemble', device='cpu', ch_names=None, n_classes=3):
        self.device = torch.device(device)
        self.n_classes = int(n_classes)
        with np.load(path, allow_pickle=False) as source:
            self.p = {k: source[k].copy() for k in source.files}
        self.mode = str(self.p['mode']) if 'mode' in self.p else mode
        if self.mode not in ('MRHydra', 'QUANT', 'ensemble'):
            raise ValueError('Unsupported classifier mode')

        self.channel_perm = None
        if 'channel_names' in self.p and ch_names:
            expected_norm = [_norm_ch(c) for c in self.p['channel_names'].tolist()]
            incoming_map = {}
            for idx, nm in enumerate(ch_names):
                incoming_map.setdefault(_norm_ch(nm), idx)
            if all(nm in incoming_map for nm in expected_norm):
                perm = [incoming_map[nm] for nm in expected_norm]
                if not (len(ch_names) == _EXPECTED_CHANS and perm == list(range(_EXPECTED_CHANS))):
                    self.channel_perm = torch.tensor(perm, dtype=torch.long, device=self.device)

        if self.mode in ('MRHydra', 'ensemble'):
            self.p['hydra_weights'] = torch.as_tensor(self.p['hydra_weights'], dtype=torch.float32, device=self.device)
            self.p['hydra_index'] = torch.as_tensor(self.p['hydra_index'], dtype=torch.long, device=self.device)
            self.p['hydra_mu'] = torch.as_tensor(self.p['hydra_mu'], dtype=torch.float32, device=self.device)
            self.p['hydra_sigma'] = torch.as_tensor(self.p['hydra_sigma'], dtype=torch.float32, device=self.device)
            self.p['_mr_cache'] = _build_mr_cache(self.p, self.device)
            self.p['_mr_mean_t'] = torch.as_tensor(self.p['mr_mean'], dtype=torch.float32, device=self.device)
            self.p['_mr_scale_t'] = torch.as_tensor(self.p['mr_scale'], dtype=torch.float32, device=self.device)
            self.p['_ridge_coef_t'] = torch.as_tensor(self.p['ridge_coef'], dtype=torch.float64, device=self.device)
            self.p['_ridge_intercept_t'] = torch.as_tensor(self.p['ridge_intercept'], dtype=torch.float64, device=self.device)

    def __deepcopy__(self, memo):
        # Model is strictly stateless across windows and sessions; share read-only tensors.
        clone = object.__new__(type(self))
        memo[id(self)] = clone
        clone.__dict__.update(self.__dict__)
        return clone

    def reset_state(self):
        pass

    def _prepare_tensor(self, x):
        if isinstance(x, torch.Tensor):
            t = x.to(device=self.device, dtype=torch.float32)
        else:
            t = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim != 3:
            t = t.reshape(t.shape[0], -1, t.shape[-1])
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        if self.channel_perm is not None and t.shape[1] > int(self.channel_perm.max()):
            t = t.index_select(1, self.channel_perm)
        if t.shape[2] != _EXPECTED_TIMES:
            t = F.interpolate(t, size=_EXPECTED_TIMES, mode='linear', align_corners=False)
        if t.shape[1] < _EXPECTED_CHANS:
            t = F.pad(t, (0, 0, 0, _EXPECTED_CHANS - t.shape[1]))
        elif t.shape[1] > _EXPECTED_CHANS:
            t = t[:, :_EXPECTED_CHANS, :]
        return t

    @torch.inference_mode()
    def scores(self, x):
        t = self._prepare_tensor(x)
        parts = []
        if self.mode in ('MRHydra', 'ensemble'):
            a = _hydra_torch(t, self.p)
            b = (_multi_rocket_torch(t, self.p['_mr_cache']) - self.p['_mr_mean_t']) / self.p['_mr_scale_t']
            feats = torch.cat((a, b), dim=1).to(torch.float64)
            vals = feats @ self.p['_ridge_coef_t'].T + self.p['_ridge_intercept_t']
            vals = ((vals + 1.0) * 0.5).clamp(0.0, 1.0)
            vals = vals / vals.sum(dim=1, keepdim=True).clamp_min(1e-20)
            parts.append(vals.cpu().numpy())
        if self.mode in ('QUANT', 'ensemble'):
            parts.append(forest_probabilities(quant_features(t.cpu(), self.p), self.p))
        out = sum(parts) / len(parts)
        if self.n_classes < out.shape[1]:
            out = out[:, :self.n_classes]
            out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-20)
        elif self.n_classes > out.shape[1]:
            out = np.pad(out, ((0, 0), (0, self.n_classes - out.shape[1])))
        return out

    def predict(self, x):
        return np.argmax(self.scores(x), axis=1).astype(np.int64)


class Solver(CompetSolver):
    name = 'Aeon-MRHydra-QUANT'
    requirements = []

    def load_model(self, meta):
        folder = Path(meta['submission_dir'])
        return SelectedModel(
            folder / 'weights.npz',
            device=meta.get('device', getattr(self, 'device', 'cpu')),
            ch_names=meta.get('ch_names'),
            n_classes=meta.get('n_classes', 3),
        )
