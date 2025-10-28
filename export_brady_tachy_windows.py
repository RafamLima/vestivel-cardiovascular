# export_brady_tachy_windows.py
# ------------------------------------------------------------
# Procura janelas de 60 s com bradicardia (<60 bpm) e
# taquicardia (>100 bpm) no MIT-BIH e exporta CSVs:
#   timestamp(ms), ecg
# Pronto para upload no Edge Impulse.
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import wfdb
from wfdb import processing

# ======== CONFIG ========
# Pasta com os ficheiros .hea/.dat/.atr
DATA_DIR = r"C:\Users\Utilizador\OneDrive\Ambiente de Trabalho\MEM\2º Ano\Tese\valores para AI\datasets\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0"

OUT_DIR  = r"C:\Users\Utilizador\OneDrive\Ambiente de Trabalho\ei_segments"
WINDOW_SEC = 60
BRADY_MAX = 60.0    # bpm
TACHY_MIN = 100.0   # bpm
CHANNEL_INDEX = 0   # 0 = MLII na maioria dos registos

os.makedirs(OUT_DIR, exist_ok=True)

def list_records(data_dir):
    recs = []
    for f in os.listdir(data_dir):
        if f.endswith(".hea"):
            recs.append(os.path.splitext(f)[0])
    return sorted(set(recs))

def load_record(path_no_ext):
    rec = wfdb.rdrecord(path_no_ext)
    fs = rec.fs
    sig = rec.p_signal[:, CHANNEL_INDEX]
    return sig, fs

def detect_rpeaks(sig, fs):
    # Filtro leve + XQRS robusto
    xqrs = processing.XQRS(sig=sig, fs=fs)
    xqrs.detect()
    rpeaks = np.asarray(xqrs.qrs_inds, dtype=int)
    return rpeaks

def hr_series(rpeaks, fs, smooth_n=5):
    if len(rpeaks) < 3:
        return np.array([], dtype=int), np.array([], dtype=float)
    rr = np.diff(rpeaks) / fs               # s
    hr = 60.0 / rr                          # bpm
    # alinhar a amostras (usar índice do batimento seguinte)
    ts_idx = rpeaks[1:]
    # suavizar um pouco p/ reduzir jitter
    hr_s = pd.Series(hr).rolling(smooth_n, center=True, min_periods=1).median().values
    return ts_idx, hr_s

def windows_by_hr(hr_idx, hr_vals, fs, predicate, min_len_sec=60):
    if len(hr_idx) == 0:
        return []
    mask = predicate(hr_vals).astype(bool)
    sig_len = int(hr_idx[-1] + 2*fs)  # margem
    ok = np.zeros(sig_len, dtype=bool)
    for i in range(len(hr_idx) - 1):
        a, b = hr_idx[i], hr_idx[i+1]
        ok[a:b] = mask[i]
    # varrer segmentos
    min_len = int(min_len_sec * fs)
    wins = []
    i = 0
    N = len(ok)
    while i < N:
        if ok[i]:
            j = i
            while j < N and ok[j]:
                j += 1
            if (j - i) >= min_len:
                # cortar exatamente 60 s a partir do início
                wins.append((i, i + min_len))
            i = j
        else:
            i += 1
    return wins

def export_csv(sig, fs, a, b, label, rec, k):
    seg = sig[a:b]
    ts = (np.arange(len(seg)) * (1000.0 / fs)).astype(float)  # ms
    df = pd.DataFrame({"timestamp": ts, "ecg": seg})
    fname = f"{label}.{rec}_win{k:02d}.csv"
    out_path = os.path.join(OUT_DIR, fname)
    df.to_csv(out_path, index=False)
    return out_path

def process_record(data_dir, rec):
    path = os.path.join(data_dir, rec)
    sig, fs = load_record(path)
    rpeaks = detect_rpeaks(sig, fs)
    hr_idx, hr_vals = hr_series(rpeaks, fs)

    brady_wins = windows_by_hr(hr_idx, hr_vals, fs, lambda x: x < BRADY_MAX, WINDOW_SEC)
    tachy_wins = windows_by_hr(hr_idx, hr_vals, fs, lambda x: x > TACHY_MIN, WINDOW_SEC)

    out_files = []
    for k, (a, b) in enumerate(brady_wins):
        out_files.append(export_csv(sig, fs, a, b, "bradicardia", rec, k))
    for k, (a, b) in enumerate(tachy_wins):
        out_files.append(export_csv(sig, fs, a, b, "taquicardia", rec, k))
    return out_files, (len(brady_wins), len(tachy_wins))

def main():
    records = list_records(DATA_DIR)
    print(f"Encontrados {len(records)} registos .hea")
    total_b = total_t = 0
    all_files = []
    for rec in records:
        files, (nb, nt) = process_record(DATA_DIR, rec)
        if nb or nt:
            print(f"[{rec}] bradi={nb}, taqui={nt} -> {len(files)} ficheiros")
            total_b += nb; total_t += nt
            all_files.extend(files)
    print("\nResumo:")
    print(f"  Bradicardia: {total_b} janelas")
    print(f"  Taquicardia: {total_t} janelas")
    print("CSV exportados em:", OUT_DIR)
    for f in all_files:
        print(" -", os.path.basename(f))

if __name__ == "__main__":
    main()
