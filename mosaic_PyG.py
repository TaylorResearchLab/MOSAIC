"""End-to-end MOSAIC embedding calculation."""

import gzip
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_add
from torch_geometric.data import Data

# Constants and mappings
DIM = 128
EPS_NODE = 1e-3
EPS_GLOBAL = 2e-4
MAX_ROUNDS = 50
USE_CONTENT_ATT = True
DIST2_COLORING = False

CHUNK_SIZE = 2_000_000
EDGE_FILE = "edges.tsv.gz"

EVIDENCE_MAP = {
    "ECO:0000315": 1.0,
    "4_star": 0.95,
    "EXP_STRONG": 0.9,
    "COEXP3": 0.7,
    "TXT5": 0.5,
    "TXT1": 0.3,
    "NA": 0.1,
}

EDGE_PRIOR = {
    "has_phenotype": 1.25,
    "gene_associated_with_disease": 1.5,
    "gene_associated_with_phenotype": 1.25,
    "regulates": 1.5,
    "part_of": 1.0,
    "is_a": 1.0,
}


device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 1 - STREAM LOAD EDGE LIST  →  INT TENSORS
# ---------------------------------------------------------------------------
id2idx = {}
etype_dict = {}
edge_chunks = []


def curie_id(x: str) -> int:
    if x not in id2idx:
        id2idx[x] = len(id2idx)
    return id2idx[x]


print("Streaming edge-list …")
reader = pd.read_csv(
    EDGE_FILE,
    sep="\t",
    compression="gzip",
    names=["src", "dst", "etype", "evidence"],
    chunksize=CHUNK_SIZE,
)
for ch in reader:
    src_i = ch["src"].map(curie_id).to_numpy()
    dst_i = ch["dst"].map(curie_id).to_numpy()

    etype_i = []
    for lab in ch["etype"]:
        if lab not in etype_dict:
            etype_dict[lab] = len(etype_dict)
        etype_i.append(etype_dict[lab])
    etype_i = np.asarray(etype_i, dtype=np.int64)

    conf = [EVIDENCE_MAP.get(e if pd.notna(e) else "NA", 0.1) for e in ch["evidence"]]
    conf = np.asarray(conf, dtype=np.float32)

    edge_chunks.append(np.stack([src_i, dst_i, etype_i, conf], axis=1))

edge_mat = np.concatenate(edge_chunks, axis=0)
num_nodes = len(id2idx)
edge_index = torch.tensor(edge_mat[:, :2].T, dtype=torch.long)
edge_type = torch.tensor(edge_mat[:, 2], dtype=torch.long)
edge_conf = torch.tensor(edge_mat[:, 3], dtype=torch.float32)
print(f"Loaded  |V|={num_nodes:,}  |E|={edge_index.size(1):,}")

# ---------------------------------------------------------------------------
# 2 - ONE-HOP (Δ+1) ADAPTIVE RANDOM GREEDY COLOURING
# ---------------------------------------------------------------------------
# Fast edge-oracle: store undirected pairs in a Python set of 64-bit ints
edge_set = set((int(u) << 32) | int(v) for u, v in edge_index.T.cpu().numpy())
edge_set |= set((int(v) << 32) | int(u) for u, v in edge_index.T.cpu().numpy())
max_deg = int(np.bincount(edge_index[0].cpu(), minlength=num_nodes).max())


def has_edge(u: int, v: int) -> bool:
    return ((u << 32) | v) in edge_set


def adaptive_random_greedy_color(n: int, delta: int, oracle) -> np.ndarray:
    import random

    classes = [[] for _ in range(delta + 1)]
    color = np.full(n, -1, dtype=np.int32)
    vertices = list(range(n))
    random.shuffle(vertices)
    for v in vertices:
        while True:
            c = random.randint(0, delta)
            if all(not oracle(u, v) for u in classes[c]):
                color[v] = c
                classes[c].append(v)
                break
    return color


print("Colouring …")
colors_np = adaptive_random_greedy_color(num_nodes, max_deg, has_edge)
colors = torch.tensor(colors_np, dtype=torch.long)

# ---------------------------------------------------------------------------
# 3 - BUILD PyG DATA OBJECT
# ---------------------------------------------------------------------------
x_init = torch.randn(num_nodes, DIM)

data = Data(
    x=x_init,
    edge_index=edge_index,
    edge_type=edge_type,
    edge_conf=edge_conf,
    colors=colors,
).to(device)


# ---------------------------------------------------------------------------
# 4 - SEMANTIC MESSAGE LAYER
# ---------------------------------------------------------------------------


class SemanticLayer(nn.Module):
    def __init__(self, dim: int, num_types: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_types, 2 * dim, dim))
        priors = [
            EDGE_PRIOR.get(lbl, 1.0)
            for lbl, _ in sorted(etype_dict.items(), key=lambda kv: kv[1])
        ]
        self.w_type = nn.Parameter(torch.tensor(priors))
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, ei, etype, conf):
        src, dst = ei
        base = torch.cat([x[dst], x[src]], dim=1) @ self.W[etype]  # (E,D)
        dot = (base * x[dst]).sum(dim=1) if USE_CONTENT_ATT else 1.0
        score = dot * self.w_type[etype] * conf

        score_exp = torch.exp(score)
        denom = scatter_add(score_exp, dst, dim=0, dim_size=x.size(0))
        alpha = score_exp / (denom[dst] + 1e-15)
        msg = scatter_add(alpha.unsqueeze(1) * base, dst, dim=0, dim_size=x.size(0))
        return self.gru(msg, x)


layer = SemanticLayer(DIM, len(etype_dict)).to(device)

# ---------------------------------------------------------------------------
# 5 - FIXED-POINT ITERATION
# ---------------------------------------------------------------------------
for r in range(MAX_ROUNDS):
    tot_delta, stable = 0.0, 0
    for c in torch.unique(data.colors):
        mask = data.colors == c
        x_prev = data.x.clone()
        data.x = layer(data.x, data.edge_index, data.edge_type, data.edge_conf)
        diff = (data.x - x_prev).norm(dim=1)
        tot_delta += diff.sum().item()
        stable += (diff < EPS_NODE).sum().item()
    if tot_delta / num_nodes < EPS_GLOBAL and stable == num_nodes:
        print(f"Converged in {r+1} rounds  (avg Δ={tot_delta / num_nodes:.2e})")
        break
else:
    print("Reached MAX_ROUNDS without full convergence")


torch.save(data.x.cpu(), "mosaic_embeddings.pt")
print("Embeddings saved to mosaic_embeddings.pt")
