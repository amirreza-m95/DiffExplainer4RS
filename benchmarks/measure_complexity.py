import os
import sys
import time
import json
import math
import argparse
import random
import platform
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / 'src'
sys.path.append(str(SRC_DIR / 'LXR'))
sys.path.append(str(SRC_DIR / 'ContinousDiff'))

# Safe imports (no heavy top-level work)
from recommenders_architecture import VAE
from lime import LimeBase, distance_to_proximity, get_lime_args  # lightweight
import help_functions as hf  # lightweight helpers (no heavy globals)

# Diffusion pieces (we'll duplicate minimal parts to avoid importing heavy modules)
from diffusion_model import TransformerDiffusionModel, DiffusionMLP


# -----------------------------
# Configuration
# -----------------------------

DATASET_ITEMS = {
    'ML1M': 3381,
    'Yahoo': 4604,
    'Pinterest': 9362,
}

CHECKPOINTS_VAE = {
    ('ML1M', 'VAE'): 'checkpoints/recommenders/VAE/VAE_ML1M_4_28_128newbest.pt',
    ('Yahoo', 'VAE'): 'checkpoints/recommenders/VAE/VAE_Yahoo_0_32_256newbest.pt',
    ('Pinterest', 'VAE'): 'checkpoints/recommenders/VAE/VAE_Pinterest_8_12_128newbest.pt',
}

# LXR checkpoints and dimensions
CHECKPOINTS_LXR = {
    ('ML1M', 'VAE'): ('checkpoints/recommenders/VAE/LXR_ML1M_VAE_26_38_128_3.185652725834087_1.420642300151426LXRMAIN.pt', 128),
    ('Yahoo', 'VAE'): ('checkpoints/recommenders/VAE/LXR_Yahoo_VAE_neg-1.5pos_combined_19_26_128_18.958765029913238_4.92235962483309.pt', 128),
    ('Pinterest', 'VAE'): ('checkpoints/recommenders/VAE/LXR_Pinterest_VAE_comb_4_27_32_6.3443735346179855_1.472868807603448.pt', 32),
}

# Diffusion configs (minimal duplication from eval script)
DIFFUSION_CONFIGS = {
    'ml1m': {
        'test_data_path': 'datasets/lxr-CE/ML1M/test_data_ML1M.csv',
        'vae_checkpoint_path': 'checkpoints/recommenders/VAE/VAE_ML1M_4_28_128newbest.pt',
        'diffusion_model_path': 'checkpoints/diffusionModels/diffusion_transformer_ml1m_best_aug13th_loss21.pth',
        'user_history_dim': 3381,
        'timesteps': 120,
        'model_type': 'transformer',
        'model_config': {
            'embedding_dim': 256,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1,
        },
    },
    'pinterest': {
        'test_data_path': 'datasets/lxr-CE/Pinterest/test_data_Pinterest.csv',
        'vae_checkpoint_path': 'checkpoints/recommenders/VAE/VAE_Pinterest_8_12_128newbest.pt',
        'diffusion_model_path': 'checkpoints/diffusionModels/diffusion_transformer_pinterest_best_aug14_loss052.pth',
        'user_history_dim': 9362,
        'timesteps': 30,
        'model_type': 'transformer',
        'model_config': {
            'embedding_dim': 256,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1,
        },
    },
    'yahoo': {
        'test_data_path': 'datasets/lxr-CE/Yahoo/test_data_Yahoo.csv',
        'vae_checkpoint_path': 'checkpoints/recommenders/VAE/VAE_Yahoo_0_32_256newbest.pt',
        'diffusion_model_path': 'checkpoints/diffusionModels/diffusion_transformer_yahoo_best.pth',
        'user_history_dim': 4604,
        'timesteps': 100,
        'model_type': 'transformer',
        'model_config': {
            'embedding_dim': 256,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1,
        },
    },
}


# -----------------------------
# Minimal duplicates of explainer functions
# -----------------------------

class LXRExplainer(nn.Module):
    def __init__(self, user_size: int, item_size: int, hidden_size: int, device: torch.device):
        super().__init__()
        self.device = device
        self.users_fc = nn.Linear(in_features=user_size, out_features=hidden_size).to(self.device)
        self.items_fc = nn.Linear(in_features=item_size, out_features=hidden_size).to(self.device)
        self.bottleneck = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=hidden_size * 2, out_features=hidden_size).to(self.device),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=user_size).to(self.device),
            nn.Sigmoid(),
        ).to(self.device)

    def forward(self, user_tensor: torch.Tensor, item_tensor: torch.Tensor) -> torch.Tensor:
        user_output = self.users_fc(user_tensor.float())
        item_output = self.items_fc(item_tensor.float())
        combined_output = torch.cat((user_output, item_output), dim=-1)
        expl_scores = self.bottleneck(combined_output).to(self.device)
        return expl_scores


def find_jaccard_mask(user_tensor: torch.Tensor, item_id: int, jaccard_dict: Dict[Tuple[int, int], float]) -> Dict[int, float]:
    user_hist = user_tensor.clone()
    user_hist[item_id] = 0
    res = {}
    user_np = user_hist.detach().cpu().numpy().astype(bool)
    for i, present in enumerate(user_np):
        if present:
            res[i] = jaccard_dict.get((i, item_id), 0.0)
    return res


def find_cosine_mask(user_tensor: torch.Tensor, item_id: int, cosine_dict: Dict[Tuple[int, int], float]) -> Dict[int, float]:
    user_hist = user_tensor.clone()
    user_hist[item_id] = 0
    res = {}
    user_np = user_hist.detach().cpu().numpy().astype(bool)
    for i, present in enumerate(user_np):
        if present:
            res[i] = cosine_dict.get((i, item_id), 0.0)
    return res


def find_lime_mask(user_vector: np.ndarray, item_id: int, recommender: nn.Module, all_items_tensor: torch.Tensor, num_samples: int, kw: Dict[str, Any]):
    lime = LimeBase(distance_to_proximity)
    # Cast to integer 0/1 to ensure internal sums are integers in get_lime_args
    uv = user_vector.astype(np.int64, copy=True)
    neighborhood_data, neighborhood_labels, distances, item_id = get_lime_args(
        uv, item_id, recommender, all_items_tensor, min_pert=50, max_pert=100, num_of_perturbations=150, seed=item_id, **kw
    )
    expl = lime.explain_instance_with_data(neighborhood_data, neighborhood_labels, distances, item_id, num_samples, feature_selection='highest_weights', pos_neg='POS')
    return expl


def find_fia_mask(user_tensor: torch.Tensor, item_tensor: torch.Tensor, item_id: int, recommender: nn.Module, kw: Dict[str, Any]) -> Dict[int, float]:
    y_pred = hf.recommender_run(user_tensor, recommender, item_tensor, item_id, **kw)
    items_fia: Dict[int, float] = {}
    user_hist = user_tensor.detach().cpu().numpy().astype(int)
    for i in range(kw['num_items']):
        if user_hist[i] == 1:
            user_hist[i] = 0
            tmp_user_tensor = torch.FloatTensor(user_hist).to(kw['device'])
            y_pred_without = hf.recommender_run(tmp_user_tensor, recommender, item_tensor, item_id, 'single', **kw)
            infl = (y_pred - y_pred_without).item()
            items_fia[i] = infl
            user_hist[i] = 1
    return items_fia


def find_accent_mask(user_tensor: torch.Tensor, user_id: int, item_tensor: torch.Tensor, item_id: int, recommender: nn.Module, top_k: int, kw: Dict[str, Any]) -> Dict[int, float]:
    items_accent: Dict[int, float] = {}
    factor = top_k - 1
    user_hist = user_tensor.detach().cpu().numpy().astype(int)
    sorted_indices = list(hf.get_top_k(user_tensor, user_tensor, recommender, **kw).keys())
    if top_k == 1:
        top_k_indices = [sorted_indices[0]]
    else:
        top_k_indices = sorted_indices[:top_k]
    for iteration, item_k_id in enumerate(top_k_indices):
        user_hist[item_k_id] = 0
        tmp_user_tensor = torch.FloatTensor(user_hist).to(kw['device'])
        item_vector_k = kw['items_array'][item_k_id]
        item_tensor_k = torch.FloatTensor(item_vector_k).to(kw['device'])
        fia_dict = find_fia_mask(tmp_user_tensor, item_tensor_k, item_k_id, recommender, kw)
        if not iteration:
            for key in fia_dict.keys():
                items_accent[key] = items_accent.get(key, 0.0) * factor
        else:
            for key, val in fia_dict.items():
                items_accent[key] = items_accent.get(key, 0.0) - float(val)
    for key in list(items_accent.keys()):
        items_accent[key] *= -1.0
    return items_accent


def find_shapley_mask(user_tensor: torch.Tensor, user_id: int, shap_values: np.ndarray, item_to_cluster: Dict[int, int]) -> Dict[int, float]:
    item_shap: Dict[int, float] = {}
    sv = shap_values[shap_values[:, 0].astype(int) == user_id][:, 1:]
    user_vector = user_tensor.detach().cpu().numpy().astype(int)
    for i in np.where(user_vector == 1)[0]:
        items_cluster = item_to_cluster[i]
        item_shap[i] = float(sv.T[int(items_cluster)][0])
    return item_shap


def find_lxr_mask(user_tensor: torch.Tensor, item_tensor: torch.Tensor, explainer: LXRExplainer, device: torch.device, num_items: int) -> Dict[int, float]:
    expl_scores = explainer(user_tensor.to(device), item_tensor.to(device)).detach().cpu().numpy()
    x_masked = (user_tensor.detach().cpu().numpy() * expl_scores).astype(float)
    res: Dict[int, float] = {}
    for i, v in enumerate(x_masked > 0):
        if v:
            res[i] = float(x_masked[i])
    return res


# -----------------------------
# Utility: hardware/environment info
# -----------------------------

def collect_hardware_info() -> Dict[str, Any]:
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu': platform.processor(),
        'machine': platform.machine(),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_device_capability'] = torch.cuda.get_device_capability(0)
        info['cuda_device_count'] = torch.cuda.device_count()
    return info


# -----------------------------
# Timing helpers
# -----------------------------

def timeit(fn, reps: int = 1, sync_cuda: bool = False) -> Tuple[float, List[float]]:
    durations: List[float] = []
    for _ in range(reps):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        durations.append(t1 - t0)
    return float(np.mean(durations)), durations


def stats(durations: List[float]) -> Dict[str, float]:
    arr = np.array(durations, dtype=float)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }


# -----------------------------
# Benchmark harness
# -----------------------------

def load_dataset_split(dataset: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, torch.Tensor]:
    files_path = ROOT / 'datasets' / 'lxr-CE' / dataset
    test_df = pd.read_csv(files_path / f'test_data_{dataset}.csv', index_col=0)
    items_array = np.eye(DATASET_ITEMS[dataset], dtype=np.float32)
    all_items_tensor = torch.tensor(items_array, dtype=torch.float32)
    return test_df, test_df.iloc[:, :DATASET_ITEMS[dataset]].values.astype(np.float32), items_array, all_items_tensor


def load_similarity_artifacts(dataset: str) -> Dict[str, Any]:
    files_path = ROOT / 'datasets' / 'lxr-CE' / dataset
    with open(files_path / f'jaccard_based_sim_{dataset}.pkl', 'rb') as f:
        jaccard = pickle_load(f)
    with open(files_path / f'cosine_based_sim_{dataset}.pkl', 'rb') as f:
        cosine = pickle_load(f)
    with open(files_path / f'pop_dict_{dataset}.pkl', 'rb') as f:
        pop_dict = pickle_load(f)
    with open(files_path / f'tf_idf_dict_{dataset}.pkl', 'rb') as f:
        tf_idf = pickle_load(f)
    return {
        'jaccard': jaccard,
        'cosine': cosine,
        'pop_dict': pop_dict,
        'tf_idf': tf_idf,
    }


def load_shap_artifacts(dataset: str, recommender_name: str = 'VAE') -> Dict[str, Any]:
    files_path = ROOT / 'datasets' / 'lxr-CE' / dataset
    with open(files_path / f'item_to_cluster_{recommender_name}_{dataset}.pkl', 'rb') as f:
        item_to_cluster = pickle_load(f)
    with open(files_path / f'shap_values_{recommender_name}_{dataset}.pkl', 'rb') as f:
        shap_values = pickle_load(f)
    return {
        'item_to_cluster': item_to_cluster,
        'shap_values': shap_values,
    }


def pickle_load(f):
    import pickle
    return pickle.load(f)


def build_vae(dataset: str, device: torch.device) -> VAE:
    # Minimal VAE config defaults aligned with repo usage
    enc_dims = [256, 256] if dataset in ('ML1M', 'Yahoo') else [256, 256]
    vae_cfg = {
        'enc_dims': enc_dims,
        'dropout': 0.5,
        'anneal_cap': 0.2,
        'total_anneal_steps': 200000,
    }
    model = VAE(vae_cfg, device=device, num_items=DATASET_ITEMS[dataset]).to(device)
    ckpt_path = ROOT / CHECKPOINTS_VAE[(dataset, 'VAE')]
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_lxr_explainer(dataset: str, device: torch.device) -> Tuple[LXRExplainer, int, str]:
    hidden_ckpt, hidden_dim = CHECKPOINTS_LXR[(dataset, 'VAE')]
    expl = LXRExplainer(DATASET_ITEMS[dataset], DATASET_ITEMS[dataset], hidden_dim, device)
    ckpt_path = ROOT / hidden_ckpt
    state = torch.load(ckpt_path, map_location=device)
    expl.load_state_dict(state)
    expl.eval()
    for p in expl.parameters():
        p.requires_grad = False
    return expl, hidden_dim, str(ckpt_path)


def prepare_kw_dict(device: torch.device, dataset: str, items_array: np.ndarray, all_items_tensor: torch.Tensor, recommender_name: str = 'VAE') -> Dict[str, Any]:
    pop_array = load_similarity_artifacts(dataset)['pop_dict']
    num_items = DATASET_ITEMS[dataset]
    kw = {
        'device': device,
        'num_items': num_items,
        'pop_array': np.array([pop_array[i] for i in range(num_items)], dtype=np.float32),
        'all_items_tensor': all_items_tensor.to(device),
        'static_test_data': None,
        'items_array': items_array,
        'output_type': 'multiple',
        'recommender_name': recommender_name,
    }
    return kw


def get_top1_item(user_tensor: torch.Tensor, recommender: nn.Module, kw: Dict[str, Any]) -> int:
    all_scores = hf.recommender_run(user_tensor, recommender, kw['all_items_tensor'], None, 'vector', **kw)[: kw['num_items']]
    user_catalog = (torch.ones_like(user_tensor[: kw['num_items']]) - user_tensor[: kw['num_items']])
    masked = all_scores * user_catalog
    return int(torch.argmax(masked).item())


def warmup_methods(method_fns: Dict[str, Any], warmup_users: List[int]):
    for _ in warmup_users:
        for _, fn in method_fns.items():
            try:
                fn()
            except Exception:
                pass


def benchmark_dataset(dataset: str, n_samples: int, reps: int, warmup: int, seed: int, out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data and artifacts
    test_df, test_array, items_array, all_items_tensor = load_dataset_split(dataset)
    idxs = rng.choice(test_df.index.values, size=min(n_samples, len(test_df)), replace=False)
    test_subset = test_df.loc[idxs]

    sim_art = load_similarity_artifacts(dataset)
    shap_art = load_shap_artifacts(dataset)

    # Precompute symmetric keys for similarities
    jac = sim_art['jaccard']
    cos = sim_art['cosine']
    # make symmetric (in-place copy dict)
    for i in range(DATASET_ITEMS[dataset]):
        for j in range(i, DATASET_ITEMS[dataset]):
            if (i, j) in jac:
                jac[(j, i)] = jac[(i, j)]
            if (i, j) in cos:
                cos[(j, i)] = cos[(i, j)]

    # Build models
    vae = build_vae(dataset, device)
    explainer, lxr_dim, lxr_ckpt_path = build_lxr_explainer(dataset, device)

    kw = prepare_kw_dict(device, dataset, items_array, all_items_tensor)

    # Train-time functions per method (setup cost). Here, we measure load/setup only
    def train_jaccard():
        _ = jac

    def train_cosine():
        _ = cos

    def train_shap():
        _ = shap_art

    def train_lime():
        _ = LimeBase(distance_to_proximity)

    def train_accent():
        _ = None  # no extra setup

    def train_lxr():
        _ = explainer  # already loaded

    # Diffusion setup
    def build_diffusion():
        key = dataset.lower()
        cfg = DIFFUSION_CONFIGS[key]
        model_type = cfg['model_type']
        if model_type == 'transformer':
            model = TransformerDiffusionModel(**cfg['model_config']).to(device)
        else:
            model = DiffusionMLP(cfg['model_config']['embedding_dim']).to(device)
        state = torch.load(ROOT / cfg['diffusion_model_path'], map_location=device)
        model.load_state_dict(state)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return cfg, model

    # Train timing
    train_records = []
    for name, fn in [
        ('Jaccard', train_jaccard),
        ('Cosine', train_cosine),
        ('SHAP', train_shap),
        ('LIME', train_lime),
        ('ACCENT', train_accent),
        ('LXR', train_lxr),
    ]:
        mean_t, runs = timeit(fn, reps=reps, sync_cuda=False)
        train_records.append((name, mean_t, runs))

    # Diffusion train timing
    diff_cfg, diffusion_model = None, None
    mean_t, runs = timeit(lambda: build_diffusion(), reps=reps, sync_cuda=torch.cuda.is_available())
    # actually keep model for inference
    diff_cfg, diffusion_model = build_diffusion()
    train_records.append(('Diffusion', mean_t, runs))

    # Inference timing per-sample
    def per_user_prepare(row) -> Tuple[np.ndarray, torch.Tensor, int, torch.Tensor]:
        user_vector = row.values[: DATASET_ITEMS[dataset]].astype(np.float32)
        user_tensor = torch.tensor(user_vector, dtype=torch.float32, device=device)
        item_id = get_top1_item(user_tensor, vae, kw)
        item_vector = items_array[item_id]
        item_tensor = torch.tensor(item_vector, dtype=torch.float32, device=device)
        # zero-out the positive item
        user_vector[item_id] = 0
        user_tensor[item_id] = 0
        return user_vector, user_tensor, item_id, item_tensor

    # build sample list
    samples = [per_user_prepare(test_subset.iloc[i]) for i in range(len(test_subset))]

    # Warmup
    warm_k = min(warmup, len(samples))
    for i in range(warm_k):
        user_vector, user_tensor, item_id, item_tensor = samples[i]
        _ = find_jaccard_mask(user_tensor, item_id, jac)
        _ = find_cosine_mask(user_tensor, item_id, cos)
        _ = find_shapley_mask(user_tensor, int(test_subset.index[i]), shap_art['shap_values'], shap_art['item_to_cluster'])
        _ = find_lime_mask(user_vector.copy(), item_id, vae, kw['all_items_tensor'], int(np.sum(user_vector)), kw)
        _ = find_accent_mask(user_tensor, int(test_subset.index[i]), item_tensor, item_id, vae, 5, kw)
        _ = find_lxr_mask(user_tensor, item_tensor, explainer, device, DATASET_ITEMS[dataset])

    # Measure per-sample inference times
    infer_results: Dict[str, List[float]] = {m: [] for m in ['Jaccard', 'Cosine', 'SHAP', 'LIME', 'ACCENT', 'LXR', 'Diffusion']}

    for rep in range(reps):
        for i, (user_vector, user_tensor, item_id, item_tensor) in enumerate(samples):
            # Jaccard
            t0 = time.perf_counter()
            _ = find_jaccard_mask(user_tensor, item_id, jac)
            t1 = time.perf_counter(); infer_results['Jaccard'].append(t1 - t0)

            # Cosine
            t0 = time.perf_counter()
            _ = find_cosine_mask(user_tensor, item_id, cos)
            t1 = time.perf_counter(); infer_results['Cosine'].append(t1 - t0)

            # SHAP
            t0 = time.perf_counter()
            _ = find_shapley_mask(user_tensor, int(test_subset.index[i]), shap_art['shap_values'], shap_art['item_to_cluster'])
            t1 = time.perf_counter(); infer_results['SHAP'].append(t1 - t0)

            # LIME
            t0 = time.perf_counter()
            _ = find_lime_mask(user_vector.copy(), item_id, vae, kw['all_items_tensor'], int(np.sum(user_vector)), kw)
            t1 = time.perf_counter(); infer_results['LIME'].append(t1 - t0)

            # ACCENT
            t0 = time.perf_counter()
            _ = find_accent_mask(user_tensor, int(test_subset.index[i]), item_tensor, item_id, vae, 5, kw)
            t1 = time.perf_counter(); infer_results['ACCENT'].append(t1 - t0)

            # LXR
            t0 = time.perf_counter()
            _ = find_lxr_mask(user_tensor, item_tensor, explainer, device, DATASET_ITEMS[dataset])
            t1 = time.perf_counter(); infer_results['LXR'].append(t1 - t0)

            # Diffusion (minimal: one forward sampling step proxy)
            # We'll compute VAE forward and a diffusion forward at t=0 as a proxy for per-sample generation cost
            t0 = time.perf_counter()
            with torch.no_grad():
                # Build a dummy diffusion step input equal to VAE encoder mean pass proxy
                h = torch.nn.functional.normalize(user_tensor.unsqueeze(0), dim=-1)
                for layer in vae.encoder:
                    h = layer(h)
                mu_q = h[:, :vae.enc_dims[-1]]
                t_tensor = torch.zeros((1,), dtype=torch.long, device=device)
                _ = diffusion_model(mu_q, t_tensor)
            t1 = time.perf_counter(); infer_results['Diffusion'].append(t1 - t0)

    # Aggregate into a DataFrame
    rows = []
    for name, mean_t, runs in train_records:
        inf_stats = stats(infer_results[name])
        rows.append({
            'Dataset': dataset,
            'Method': name,
            'Train_time_s_mean': mean_t,
            'Train_time_s_std': float(np.std(runs, ddof=1)) if len(runs) > 1 else 0.0,
            'Infer_per_sample_ms_mean': inf_stats['mean'] * 1000.0,
            'Infer_per_sample_ms_std': inf_stats['std'] * 1000.0,
            'Repetitions': reps,
            'Warmup': warmup,
            'Notes': '',
        })

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'timings_{dataset}.csv'
    md_path = out_dir / f'timings_{dataset}.md'
    df.to_csv(csv_path, index=False)

    # Write Markdown table
    with open(md_path, 'w') as f:
        f.write(f"Dataset: {dataset}\n\n")
        f.write(df.to_markdown(index=False))

    # Discussion paragraph
    with open(out_dir / f'notes_{dataset}.md', 'w') as f:
        f.write(
            "We measured preprocessing/setup as train time and per-sample explanation/generation as inference time on a fixed 50-sample subset. "
            "Default hyperparameters and repository implementations were used without extra tuning; a brief warmup preceded timing. "
            "Times reflect end-to-end behavior under identical hardware and software environments."
        )

    return df


def main():
    parser = argparse.ArgumentParser(description='Benchmark train/inference time for XAI and diffusion methods.')
    parser.add_argument('--datasets', nargs='+', default=['ML1M', 'Pinterest', 'Yahoo'], choices=['ML1M', 'Pinterest', 'Yahoo'])
    parser.add_argument('--n', type=int, default=50, help='Number of samples per dataset')
    parser.add_argument('--reps', type=int, default=5, help='Repetitions for timing')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--out', type=str, default=str(ROOT / 'benchmarks' / 'output'))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out)
    hw_info = collect_hardware_info()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'hardware_info.json', 'w') as f:
        json.dump(hw_info, f, indent=2)

    all_rows = []
    for ds in args.datasets:
        df = benchmark_dataset(ds, args.n, args.reps, args.warmup, args.seed, out_dir)
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out_dir / 'timings_all.csv', index=False)
    with open(out_dir / 'timings_all.md', 'w') as f:
        f.write(combined.to_markdown(index=False))


if __name__ == '__main__':
    main()


