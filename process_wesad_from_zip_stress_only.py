# process_wesad_from_zip_stress_only.py
# Extrai STRESS do WESAD (ECG, RESP, GSR/EDA, TEMP) e exporta CSV p/ Edge Impulse:
# timestamp,ecg,resp,gsr,temp @ 4 Hz (ms desde 0)

import os, re, pickle, zipfile
import numpy as np, pandas as pd
from collections import Counter

OUT_DIR   = "wesad_csv"
FS_CHEST  = 700
FS_OUT    = 4
STRESS_LABEL = 2  # 1=baseline, 2=stress, 3=amusement

# ---------- utilitários ----------
def find_zip():
    here = os.path.dirname(os.path.abspath(__file__))
    for f in os.listdir(here):
        if f.lower().endswith(".zip"):
            return os.path.join(here, f)
    return None

def safe_mkdir(p): os.makedirs(p, exist_ok=True)

def is_valid_signal(x):
    try:
        arr = np.asarray(x)
        return arr.size > 0 and np.isfinite(arr).any()
    except Exception:
        return False

def to1d(x):
    if x is None:
        return None
    arr = np.asarray(x).squeeze()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.astype(float)

def resample_mean(x, fs_in, fs_out=FS_OUT):
    x = to1d(x)
    if x is None or x.size == 0:
        return np.array([], dtype=float)
    block = int(round(fs_in / fs_out))
    if block <= 0 or len(x) < block:
        return np.array([], dtype=float)
    n = len(x) // block
    return x[:n*block].reshape(n, block).mean(axis=1)

def downsample_labels_to_len(labels, target_len):
    labels = to1d(labels)
    if labels is None or target_len <= 0:
        return np.array([], dtype=int)
    if len(labels) == target_len:
        return labels.astype(int)
    block = max(1, int(round(len(labels) / target_len)))
    n_blocks = len(labels) // block
    trimmed = labels[:n_blocks*block].reshape(n_blocks, block)
    mode_vals = [Counter(row.tolist()).most_common(1)[0][0] for row in trimmed]
    return np.asarray(mode_vals, dtype=int)

def member_id(path):
    m = re.search(r"(s\d{1,2})", path.replace("\\", "/"), re.IGNORECASE)
    return m.group(1).lower() if m else "unknown"

def get_label_array(d):
    lab = d.get("label", None)
    if lab is None:
        return None
    if isinstance(lab, dict) and "label" in lab:
        lab = lab["label"]
    lab = np.asarray(lab)
    lab = np.ravel(lab)  # 1D
    return lab

# ---------- extração de canais (SEM 'or' entre arrays) ----------
def load_channels(d):
    sig   = d.get("signal", {})
    chest = sig.get("chest", {})
    wrist = sig.get("wrist", {})

    # ECG & RESP (peito, 700 Hz)
    ecg_chest  = chest.get("ECG", None)
    resp_chest = chest.get("Resp", None)
    ecg_4hz  = resample_mean(ecg_chest,  FS_CHEST) if is_valid_signal(ecg_chest)  else None
    resp_4hz = resample_mean(resp_chest, FS_CHEST) if is_valid_signal(resp_chest) else None

    # EDA/GSR: preferir wrist (4 Hz); senão chest (700->4)
    eda_wrist = wrist.get("EDA", None)
    if is_valid_signal(eda_wrist):
        eda_4hz = to1d(eda_wrist)                 # já 4 Hz
    else:
        eda_chest = chest.get("EDA", None)
        eda_4hz = resample_mean(eda_chest, FS_CHEST) if is_valid_signal(eda_chest) else None

    # TEMP: preferir wrist (4 Hz); senão chest (700->4)
    temp_wrist = wrist.get("TEMP", None)
    if not is_valid_signal(temp_wrist):
        temp_wrist = wrist.get("Temp", None)
    if is_valid_signal(temp_wrist):
        temp_4hz = to1d(temp_wrist)               # já 4 Hz
    else:
        temp_chest = chest.get("Temp", None)
        temp_4hz = resample_mean(temp_chest, FS_CHEST) if is_valid_signal(temp_chest) else None

    return ecg_4hz, resp_4hz, eda_4hz, temp_4hz

# ---------- construir CSV ----------
def build_df_stress(ecg_4hz, resp_4hz, eda, temp, labels):
    ecg_4hz = to1d(ecg_4hz)
    resp_4hz = to1d(resp_4hz)
    eda = to1d(eda)
    temp = to1d(temp)
    labels = np.asarray(labels).reshape(-1)

    if any(x is None for x in (ecg_4hz, resp_4hz, eda, temp, labels)):
        return None, "faltam canais/labels"

    target_len = min(len(ecg_4hz), len(resp_4hz), len(eda), len(temp))
    if target_len <= 0:
        return None, "comprimento zero"

    labels_ds = downsample_labels_to_len(labels, target_len)[:target_len]
    stress_idx = np.where(labels_ds == STRESS_LABEL)[0]
    if stress_idx.size == 0:
        return None, "sem stress"

    ecg_st  = ecg_4hz[:target_len][stress_idx]
    resp_st = resp_4hz[:target_len][stress_idx]
    eda_st  = eda[:target_len][stress_idx]
    temp_st = temp[:target_len][stress_idx]

    timestamp_ms = (np.arange(stress_idx.size) * (1000 // FS_OUT)).astype(int)

    df = pd.DataFrame({
        "timestamp": timestamp_ms,
        "ecg":  ecg_st,
        "resp": resp_st,
        "gsr":  eda_st,
        "temp": temp_st
    })
    return df, None

# ---------- main ----------
def main():
    zip_path = find_zip()
    if not zip_path:
        print("[!] Nenhum .zip encontrado.")
        return
    print(f"[i] A usar ZIP: {zip_path}")
    safe_mkdir(OUT_DIR)

    with zipfile.ZipFile(zip_path, "r") as z:
        all_pkls = [m for m in z.namelist() if m.lower().endswith(".pkl")]
        if not all_pkls:
            print("[!] Nenhum .pkl encontrado no ZIP.")
            return

        classic = [m for m in all_pkls if re.search(r"/s\d{1,2}/s\d{1,2}\.pkl$", m.replace("\\", "/"), re.IGNORECASE)]
        candidates = classic if classic else all_pkls
        if not classic:
            print("[i] Padrão clássico não encontrado; vou tentar todos os .pkl.")

        exported, skipped = 0, 0
        for m in candidates:
            sid = member_id(m)
            try:
                with z.open(m, "r") as f:
                    d = pickle.load(f, encoding="latin1")

                labels = get_label_array(d)
                ecg_4hz, resp_4hz, eda, temp = load_channels(d)

                def ok(x):
                    try:
                        return x is not None and np.asarray(x).size > 0
                    except Exception:
                        return False

                print(f"[i] {sid}: ECG={'ok' if ok(ecg_4hz) else 'x'} "
                      f"RESP={'ok' if ok(resp_4hz) else 'x'} "
                      f"EDA={'ok' if ok(eda) else 'x'} "
                      f"TEMP={'ok' if ok(temp) else 'x'} "
                      f"| labels={len(labels) if labels is not None else 0}")

                df, reason = build_df_stress(ecg_4hz, resp_4hz, eda, temp, labels)
                if df is None:
                    print(f"[i] {sid}: a saltar ({reason}).")
                    skipped += 1
                    continue

                out = os.path.join(OUT_DIR, f"{sid}_stress.csv")
                df.to_csv(out, index=False)
                print(f"[✓] {sid}: {len(df)} amostras exportadas -> {out}")
                exported += 1

            except Exception as e:
                print(f"[!] Erro em {m}: {e}")
                skipped += 1

        print(f"[i] Concluído. Exportados: {exported}. Saltados: {skipped}.")

if __name__ == "__main__":
    main()
