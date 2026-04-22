import json
import math
import os

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np

if HAS_TORCH:
    class ANFIS(nn.Module):

        def __init__(self, n_inputs=3, n_mfs=2, n_outputs=2):
            super().__init__()
            self.n_inputs = n_inputs
            self.n_mfs = n_mfs
            self.n_outputs = n_outputs
            self.n_rules = n_mfs ** n_inputs

            self.centers = nn.Parameter(torch.zeros(n_inputs, n_mfs))
            self.log_sigmas = nn.Parameter(torch.zeros(n_inputs, n_mfs))

            self.consequent = nn.Parameter(
                torch.zeros(n_outputs, self.n_rules, n_inputs + 1))

            self._init_params()

        def _init_params(self):

            for i in range(self.n_inputs):
                if self.n_mfs == 1:
                    self.centers.data[i, 0] = 0.5
                else:
                    for j in range(self.n_mfs):
                        self.centers.data[i, j] = j / (self.n_mfs - 1)

            self.log_sigmas.data.fill_(math.log(1.0 / self.n_mfs))

            nn.init.normal_(self.consequent, mean=0.0, std=0.1)

            for out_idx in range(self.n_outputs):

                if out_idx == 0:
                    self.consequent.data[out_idx, :, -1] = 0.1
                else:
                    self.consequent.data[out_idx, :, -1] = 0.6

        def forward(self, x):
            batch_size = x.shape[0]

            x_expanded = x.unsqueeze(2)
            sigmas = torch.exp(self.log_sigmas)

            mu = torch.exp(-0.5 * ((x_expanded - self.centers) / sigmas) ** 2)

            rule_strengths = self._compute_rule_strengths(mu)

            w_sum = rule_strengths.sum(dim=1, keepdim=True).clamp(min=1e-8)
            w_bar = rule_strengths / w_sum

            x_aug = torch.cat([x, torch.ones(batch_size, 1, device=x.device)],
                              dim=1)

            outputs = []
            for out_idx in range(self.n_outputs):

                f_k = x_aug @ self.consequent[out_idx].T

                out = (w_bar * f_k).sum(dim=1)
                outputs.append(out)

            result = torch.stack(outputs, dim=1)

            return result

        def compute_normalized_strengths(self, x):
            x_expanded = x.unsqueeze(2)
            sigmas = torch.exp(self.log_sigmas)
            mu = torch.exp(-0.5 * ((x_expanded - self.centers) / sigmas) ** 2)
            rule_strengths = self._compute_rule_strengths(mu)
            w_sum = rule_strengths.sum(dim=1, keepdim=True).clamp(min=1e-8)
            return rule_strengths / w_sum

        def build_lse_design(self, x):
            batch_size = x.shape[0]
            w_bar = self.compute_normalized_strengths(x)
            x_aug = torch.cat(
                [x, torch.ones(batch_size, 1, device=x.device)], dim=1)

            A = w_bar.unsqueeze(2) * x_aug.unsqueeze(1)
            return A.reshape(batch_size, -1)

        def update_consequent_lse(self, A, Y, reg=1e-4):
            n_feat = A.shape[1]
            AtA = A.T @ A + reg * torch.eye(n_feat, device=A.device)
            AtY = A.T @ Y
            p = torch.linalg.solve(AtA, AtY)

            p_reshaped = p.T.reshape(
                self.n_outputs, self.n_rules, self.n_inputs + 1)
            with torch.no_grad():
                self.consequent.data.copy_(p_reshaped)

        def _compute_rule_strengths(self, mu):
            batch_size = mu.shape[0]

            import itertools
            mf_indices = list(itertools.product(range(self.n_mfs),
                                                repeat=self.n_inputs))

            rule_list = []
            for k, indices in enumerate(mf_indices):
                w = torch.ones(batch_size, device=mu.device)
                for inp_idx, mf_idx in enumerate(indices):
                    w = w * mu[:, inp_idx, mf_idx]
                rule_list.append(w)

            return torch.stack(rule_list, dim=1)

        def get_premise_params(self):
            return [self.centers, self.log_sigmas]

        def get_consequent_params(self):
            return [self.consequent]

class ANFISInference:

    def __init__(self, centers, sigmas, consequent, n_inputs=3, n_mfs=2,
                 n_outputs=2):
        self.centers = np.array(centers, dtype=np.float64)
        self.sigmas = np.array(sigmas, dtype=np.float64)
        self.consequent = np.array(consequent, dtype=np.float64)
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.n_outputs = n_outputs
        self.n_rules = n_mfs ** n_inputs

        import itertools
        self._mf_indices = list(itertools.product(range(n_mfs),
                                                  repeat=n_inputs))

    def predict(self, s, d, u):
        x = np.array([s, d, u], dtype=np.float64)

        diff = x.reshape(-1, 1) - self.centers
        mu = np.exp(-0.5 * (diff / self.sigmas) ** 2)

        strengths = np.ones(self.n_rules)
        for k, indices in enumerate(self._mf_indices):
            for inp_idx, mf_idx in enumerate(indices):
                strengths[k] *= mu[inp_idx, mf_idx]

        w_sum = strengths.sum()
        if w_sum < 1e-8:
            w_sum = 1e-8
        w_bar = strengths / w_sum

        x_aug = np.append(x, 1.0)
        outputs = []
        for out_idx in range(self.n_outputs):
            f_k = self.consequent[out_idx] @ x_aug
            out = np.dot(w_bar, f_k)
            outputs.append(out)

        lt = float(outputs[0])
        ut = float(outputs[1])
        return lt, ut

    @classmethod
    def from_file(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            centers=data['centers'],
            sigmas=data['sigmas'],
            consequent=data['consequent'],
            n_inputs=data.get('n_inputs', 3),
            n_mfs=data.get('n_mfs', 2),
            n_outputs=data.get('n_outputs', 2),
        )

def export_weights(model, path):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required to export weights")
    data = {
        'n_inputs': model.n_inputs,
        'n_mfs': model.n_mfs,
        'n_outputs': model.n_outputs,
        'centers': model.centers.detach().cpu().numpy().tolist(),
        'sigmas': torch.exp(model.log_sigmas).detach().cpu().numpy().tolist(),
        'consequent': model.consequent.detach().cpu().numpy().tolist(),
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
