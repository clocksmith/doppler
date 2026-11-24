"""Train the tiny GPT-2-architecture fixture used by the protected-backend prototype.

Research fixture, not a Doppler-catalogued model. One pre-LN GPT-2 block (d=16, one head, MLP 64, GELU tanh,
tied head) over a 32-symbol alphabet and an 8-position context. It is trained on letter progressions with stride
1, 2 or 3 ("ace" -> "gikm"), so the next symbol depends on earlier positions and the attention path matters. A training penalty keeps each
row's largest attention score inside a band, so the encrypted softmax denominator stays in a narrow range.

Writes tiny-model.json: the float64 weights, the alphabet, and calibration ranges of every nonlinear input over the
full task, measured on the trained model. Deterministic for a fixed torch version and seed.

    python3 tools/privacy-protected/train_tiny.py --out tools/privacy-protected/tiny-model.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math

import torch
import torch.nn.functional as F

ALPHABET = "abcdefghijklmnopqrstuvwxyz .,!?'"
V, C, D, HID = 32, 8, 16, 64
STRIDES = (1, 2, 3)


def sequences():
    out = []
    for start in range(26):
        for stride in STRIDES:
            out.append([(start + stride * i) % 26 for i in range(C + 1)])
    return out


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = torch.nn.Parameter(torch.randn(V, D) * 0.3)
        self.wpe = torch.nn.Parameter(torch.randn(C, D) * 0.1)
        self.ln1 = torch.nn.LayerNorm(D)
        self.attn = torch.nn.Linear(D, 3 * D)
        self.proj = torch.nn.Linear(D, D)
        self.ln2 = torch.nn.LayerNorm(D)
        self.fc1 = torch.nn.Linear(D, HID)
        self.fc2 = torch.nn.Linear(HID, D)
        self.lnf = torch.nn.LayerNorm(D)

    def forward(self, idx, probe=None, rowmax=None):
        t = idx.shape[1]
        x = self.wte[idx] + self.wpe[:t]
        a = self.ln1(x)
        q, k, v = self.attn(a).split(D, dim=-1)
        s = q @ k.transpose(-1, -2) / math.sqrt(D)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), 1)
        s = s.masked_fill(mask, float("-inf"))
        if rowmax is not None:
            rowmax.append(s.max(dim=-1).values)
        h = x + self.proj(torch.softmax(s, dim=-1) @ v)
        m = self.ln2(h)
        u = self.fc1(m)
        h2 = h + self.fc2(F.gelu(u, approximate="tanh"))
        if probe is not None:
            probe["scores"].append(s[~mask.expand_as(s)].detach())
            probe["ln2_var"].append(h.var(dim=-1, unbiased=False).detach().flatten())
            probe["gelu_in"].append(u.detach().flatten())
            probe["lnf_var"].append(h2.var(dim=-1, unbiased=False).detach().flatten())
        return self.lnf(h2) @ self.wte.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--score-floor", type=float, default=6.0)
    ap.add_argument("--score-ceiling", type=float, default=8.0)
    ap.add_argument("--score-penalty", type=float, default=1.0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    model = Tiny().double()
    data = torch.tensor(sequences())
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(a.steps):
        rowmax = []
        logits = model(data[:, :C], rowmax=rowmax)
        loss = F.cross_entropy(logits.reshape(-1, V), data[:, 1:].reshape(-1))
        # Encryption-friendliness: keep every row's largest attention score in a narrow band so the softmax
        # denominator stays in a range a low-degree inverse approximation covers (see README, Approximations).
        loss = loss + a.score_penalty * (F.relu(a.score_floor - rowmax[0]).pow(2)
                                    + F.relu(rowmax[0] - a.score_ceiling).pow(2)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    probe = {"scores": [], "ln2_var": [], "gelu_in": [], "lnf_var": []}
    with torch.no_grad():
        logits = model(data[:, :C], probe)
        acc = (logits.argmax(-1) == data[:, 1:]).double().mean().item()
        top2 = logits.topk(2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1]).min().item()
        # also every prefix length (shorter contexts appear at generation time)
        for t in range(1, C):
            model(data[:, :t], probe)
    ranges = {k: [torch.cat(v).min().item(), torch.cat(v).max().item()] for k, v in probe.items()}
    sd = {k: v.detach().double().tolist() for k, v in model.state_dict().items()}
    weights = {
        "wte": sd["wte"], "wpe": sd["wpe"],
        "ln1_g": sd["ln1.weight"], "ln1_b": sd["ln1.bias"],
        "attn_w": sd["attn.weight"], "attn_b": sd["attn.bias"],     # torch Linear: [out, in]
        "proj_w": sd["proj.weight"], "proj_b": sd["proj.bias"],
        "ln2_g": sd["ln2.weight"], "ln2_b": sd["ln2.bias"],
        "fc1_w": sd["fc1.weight"], "fc1_b": sd["fc1.bias"],
        "fc2_w": sd["fc2.weight"], "fc2_b": sd["fc2.bias"],
        "lnf_g": sd["lnf.weight"], "lnf_b": sd["lnf.bias"],
        "ln_eps": 1e-5,
    }
    body = {
        "schema": "doppler.protected-tiny-model/v1",
        "note": "Research fixture trained by train_tiny.py; not a catalogued Doppler model.",
        "alphabet": ALPHABET, "vocab": V, "context": C, "dim": D, "hidden": HID, "heads": 1,
        "activation": "gelu-tanh", "tiedHead": True,
        "training": {"seed": a.seed, "steps": a.steps, "scoreFloor": a.score_floor, "scoreCeiling": a.score_ceiling, "scorePenalty": a.score_penalty, "torch": torch.__version__, "loss": loss.item(),
                     "teacherForcedAccuracy": acc, "minTop1Margin": margin},
        "calibration": ranges,
        "weights": weights,
    }
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    with open(a.out, "w") as f:
        f.write(text)
    print(json.dumps({"loss": loss.item(), "accuracy": acc, "minMargin": margin, "ranges": ranges,
                      "sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=1))


if __name__ == "__main__":
    main()
