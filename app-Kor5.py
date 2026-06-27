# -*- coding: utf-8 -*-
"""
Aplikasi Analisis Data: Korelasi & Regresi OLS
===============================================
Dua halaman dalam satu file:

  Halaman 1 — Analisis Korelasi
      Unggah Excel -> pilih variabel target -> pilih independen ->
      matriks korelasi, heatmap, narasi otomatis, multikolinearitas.

  Halaman 2 — Regresi OLS
      Pilih variabel dependen & independen -> persamaan linier berwarna,
      kartu metrik (R², Adj R², MAE, RMSE, F, n), tabel koefisien dengan
      t-stat & signifikansi, plot Actual vs Predicted.

OLS dihitung manual (numpy + scipy), tanpa statsmodels -> tetap single-file
dan ringan dependensi.

Cara menjalankan:
    pip install streamlit pandas numpy plotly scipy openpyxl
    streamlit run app_korelasi.py

Author: untuk otaQku / Orbex GenAI Neuro Lab
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# --------------------------------------------------------------------------- #
# Konfigurasi halaman & gaya
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Analisis Korelasi & Regresi OLS", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.6rem; }
        h1, h2, h3 { color: #1f2a44; }
        .kuat   { background:#e7f6ec; border-left:5px solid #2e8b57; padding:0.7rem 1rem; border-radius:8px; margin:0.35rem 0;}
        .sedang { background:#fff6e5; border-left:5px solid #d39e00; padding:0.7rem 1rem; border-radius:8px; margin:0.35rem 0;}
        .lemah  { background:#fdecec; border-left:5px solid #c0392b; padding:0.7rem 1rem; border-radius:8px; margin:0.35rem 0;}
        .catatan{ background:#eef2fb; border-left:5px solid #3b5bdb; padding:0.8rem 1rem; border-radius:8px;}
        /* Banner persamaan */
        .eqbox { background:#161b29; color:#fff; padding:1.4rem 1.6rem; border-radius:12px;
                 font-size:1.45rem; font-weight:600; letter-spacing:.3px; line-height:2.2rem; }
        /* Kartu metrik */
        .card { border:2px solid #ccc; border-radius:12px; padding:0.9rem 1rem; text-align:center; background:#fff; }
        .card .ttl { font-size:0.82rem; font-weight:700; }
        .card .val { font-size:1.9rem; font-weight:800; color:#1f2a44; margin:.2rem 0; }
        .card .sub { font-size:0.72rem; color:#777; font-style:italic; }
        /* Tabel koefisien */
        table.coef { border-collapse:collapse; width:100%; font-size:0.92rem; }
        table.coef th { background:#f3f5fb; padding:.55rem .7rem; text-align:center; color:#1f2a44; }
        table.coef td { padding:.55rem .7rem; border-bottom:1px solid #eee; text-align:center; }
        table.coef td.var { text-align:left; font-weight:600; }
        td.tcol { background:#fdecf2; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Konstanta & util
# --------------------------------------------------------------------------- #
METODE = {
    "Pearson (hubungan linier)": "pearson",
    "Spearman (hubungan monoton / rank)": "spearman",
    "Kendall (rank, sampel kecil)": "kendall",
}
UJI = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}
PALET = ["#e8590c", "#2f9e44", "#1c7ed6", "#9c36b5", "#e8a700", "#0ca678", "#f03e3e", "#4263eb"]


@st.cache_data(show_spinner=False)
def baca_excel(konten: bytes):
    xls = pd.ExcelFile(io.BytesIO(konten), engine="openpyxl")
    return {sh: xls.parse(sh) for sh in xls.sheet_names}


def fmt_eq(v: float) -> str:
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def fmt_tabel(v: float) -> str:
    a = abs(v)
    if a >= 1000:
        return f"{v:,.2f}"
    if a >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"


def bintang(p: float) -> str:
    if p < 0.01:
        return "★★★"
    if p < 0.05:
        return "★★"
    if p < 0.10:
        return "★"
    return "–"


# ============================ HALAMAN 1: KORELASI ========================== #
def klasifikasi_kekuatan(r: float):
    a = abs(r)
    if a >= 0.8:
        return "Sangat kuat", "kuat"
    if a >= 0.6:
        return "Kuat", "kuat"
    if a >= 0.4:
        return "Sedang", "sedang"
    if a >= 0.2:
        return "Lemah", "lemah"
    return "Sangat lemah / dapat diabaikan", "lemah"


def arah(r: float):
    if r > 0:
        return "positif"
    if r < 0:
        return "negatif"
    return "nol"


def hitung_korelasi_pvalue(df, target, independen, metode):
    fungsi = UJI[metode]
    baris = []
    for var in independen:
        pasangan = df[[target, var]].dropna()
        if len(pasangan) < 3:
            r, p = np.nan, np.nan
        else:
            r, p = fungsi(pasangan[target], pasangan[var])
        label, kelas = klasifikasi_kekuatan(r) if pd.notna(r) else ("Data kurang", "lemah")
        baris.append(
            {
                "Variabel Independen": var,
                "Korelasi (r)": round(r, 3) if pd.notna(r) else np.nan,
                "Arah": arah(r) if pd.notna(r) else "-",
                "Kekuatan": label,
                "p-value": round(p, 4) if pd.notna(p) else np.nan,
                "Signifikan (α=0,05)": ("Ya" if pd.notna(p) and p < 0.05 else "Tidak"),
                "n": len(pasangan),
                "_kelas": kelas,
                "_abs": abs(r) if pd.notna(r) else 0,
            }
        )
    return pd.DataFrame(baris).sort_values("_abs", ascending=False).reset_index(drop=True)


def cari_multikolinearitas(matriks, independen, ambang=0.7):
    pasangan = []
    for i in range(len(independen)):
        for j in range(i + 1, len(independen)):
            a, b = independen[i], independen[j]
            r = matriks.loc[a, b]
            if pd.notna(r) and abs(r) >= ambang:
                pasangan.append((a, b, round(r, 3)))
    return sorted(pasangan, key=lambda x: abs(x[2]), reverse=True)


def heatmap_korelasi(matriks):
    z = matriks.values
    teks = [[f"{v:.2f}" if pd.notna(v) else "" for v in baris] for baris in z]
    fig = go.Figure(
        data=go.Heatmap(
            z=z, x=matriks.columns, y=matriks.index, zmin=-1, zmax=1,
            colorscale="RdBu", reversescale=True, text=teks, texttemplate="%{text}",
            textfont={"size": 11}, colorbar=dict(title="r"),
            hovertemplate="%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(450, 60 * len(matriks)), margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(tickangle=-45), yaxis=dict(autorange="reversed"), plot_bgcolor="white",
    )
    return fig


def halaman_korelasi(df, kolom_numerik, metode, nama_metode, ambang_multikol):
    st.title("📊 Analisis Korelasi Antar Variabel")

    st.subheader("🎯 Pemilihan Variabel")
    c1, c2 = st.columns(2)
    with c1:
        target = st.selectbox("1️⃣ Variabel target / dependen", kolom_numerik, key="korr_target")
    kandidat = [k for k in kolom_numerik if k != target]
    with c2:
        independen = st.multiselect("2️⃣ Variabel independen", kandidat, default=kandidat, key="korr_indep")

    if not independen:
        st.warning("Pilih minimal satu variabel independen.")
        return

    data = df[[target] + independen].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    matriks = data.corr(method=metode)

    st.subheader("🧮 Matriks Korelasi")
    st.dataframe(matriks.style.format("{:.3f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                 use_container_width=True)

    st.subheader("🔥 Heatmap Korelasi")
    st.plotly_chart(heatmap_korelasi(matriks), use_container_width=True)

    st.subheader(f"📌 Korelasi Independen terhadap «{target}»")
    hasil = hitung_korelasi_pvalue(data, target, independen, metode)
    st.dataframe(hasil.drop(columns=["_kelas", "_abs"]), use_container_width=True)

    st.subheader("📝 Narasi Otomatis")
    valid = hasil[hasil["Korelasi (r)"].notna()]
    if valid.empty:
        st.warning("Korelasi tidak dapat dihitung (data tidak memadai).")
        return
    t0 = valid.iloc[0]
    st.markdown(
        f"""<div class="catatan">Dengan metode <b>{nama_metode.split(' (')[0]}</b>, dari
        <b>{len(independen)}</b> variabel independen terhadap <b>{target}</b>, hubungan terkuat
        ada pada <b>{t0['Variabel Independen']}</b> (r = {t0['Korelasi (r)']},
        {t0['Kekuatan'].lower()}, arah {arah(t0['Korelasi (r)'])}): bila
        <i>{t0['Variabel Independen']}</i> naik, <i>{target}</i> cenderung
        {'naik' if t0['Korelasi (r)'] > 0 else 'turun'}.</div>""",
        unsafe_allow_html=True,
    )
    st.markdown("**Rincian per variabel (terkuat lebih dulu):**")
    for _, row in valid.iterrows():
        sig = "signifikan" if row["Signifikan (α=0,05)"] == "Ya" else "**tidak** signifikan"
        st.markdown(
            f"""<div class="{row['_kelas']}"><b>{row['Variabel Independen']}</b> — r = {row['Korelasi (r)']} ·
            <b>{row['Kekuatan'].lower()}</b> · arah {arah(row['Korelasi (r)'])} · {sig}
            (p = {row['p-value']}, n = {row['n']}).</div>""",
            unsafe_allow_html=True,
        )

    kuat = valid[valid["_abs"] >= 0.6]
    if not kuat.empty:
        st.success("✅ **Prediktor kuat** (|r| ≥ 0,6): " + ", ".join(kuat["Variabel Independen"]) +
                   ". Paling layak diprioritaskan dalam pemodelan.")
    else:
        st.info("ℹ️ Tidak ada hubungan kuat (|r| ≥ 0,6). Pertimbangkan variabel lain atau hubungan non-linier.")

    pasangan = cari_multikolinearitas(matriks, independen, ambang_multikol)
    if pasangan:
        st.subheader("⚠️ Peringatan Multikolinearitas")
        st.markdown(f"Pasangan dengan |r| ≥ {ambang_multikol:.2f} (informasi tumpang tindih):")
        for a, b, r in pasangan:
            st.markdown(f"- **{a}** ↔ **{b}** (r = {r})")


# ============================ HALAMAN 2: REGRESI OLS ======================= #
def jalankan_ols(y, X_raw, nama_var):
    """OLS manual. Mengembalikan dict berisi seluruh statistik."""
    n = len(y)
    X = np.column_stack([np.ones(n), X_raw])           # tambah intercept
    k = X.shape[1]                                      # jumlah parameter
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
    except np.linalg.LinAlgError:
        # fallback bila SVD gagal konvergen: regularisasi ringan (ridge ~0)
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX + 1e-8 * np.eye(k))
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    resid = y - y_hat
    df_resid = n - k

    SSR = float(resid @ resid)
    TSS = float(((y - y.mean()) ** 2).sum())
    R2 = 1 - SSR / TSS if TSS > 0 else np.nan
    adjR2 = 1 - (1 - R2) * (n - 1) / df_resid if df_resid > 0 else np.nan

    sigma2 = SSR / df_resid if df_resid > 0 else np.nan
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t_stat = beta / se
    p_val = 2 * stats.t.sf(np.abs(t_stat), df_resid)

    F = (R2 / (k - 1)) / ((1 - R2) / df_resid) if (df_resid > 0 and R2 < 1) else np.nan
    pF = stats.f.sf(F, k - 1, df_resid) if np.isfinite(F) else np.nan

    MAE = float(np.abs(resid).mean())
    RMSE = float(np.sqrt((resid ** 2).mean()))

    # --- Koefisien terstandarisasi (beta baku): b_j * sd(x_j)/sd(y)
    p = X_raw.shape[1]
    std_y = y.std(ddof=1)
    std_x = X_raw.std(axis=0, ddof=1)
    beta_std = np.full(k, np.nan)
    for i in range(1, k):
        beta_std[i] = beta[i] * std_x[i - 1] / std_y if std_y > 0 else np.nan

    # --- VIF: elemen diagonal invers matriks korelasi antar prediktor
    vif = np.full(k, np.nan)
    if p >= 2:
        try:
            R = np.corrcoef(X_raw, rowvar=False)
            if np.all(np.isfinite(R)):
                diag = np.clip(np.diag(np.linalg.pinv(R)), 1.0, None)
                for i in range(1, k):
                    vif[i] = diag[i - 1]
        except np.linalg.LinAlgError:
            pass  # biarkan VIF = NaN bila SVD gagal
    elif p == 1:
        vif[1] = 1.0

    nama_lengkap = ["(Intercept)"] + list(nama_var)
    return {
        "nama": nama_lengkap, "beta": beta, "se": se, "t": t_stat, "p": p_val,
        "beta_std": beta_std, "vif": vif,
        "n": n, "k": k, "R2": R2, "adjR2": adjR2, "F": F, "pF": pF,
        "MAE": MAE, "RMSE": RMSE, "y": y, "y_hat": y_hat,
    }


def banner_persamaan(dep, hasil):
    nama, beta = hasil["nama"], hasil["beta"]
    intercept = beta[0]
    bagian = [f'<span style="color:#f5b301">{fmt_eq(intercept)}</span>']
    for i, var in enumerate(nama[1:], start=1):
        warna = PALET[(i - 1) % len(PALET)]
        op = " − " if beta[i] < 0 else " + "
        bagian.append(
            f'<span style="color:#9aa4b2">{op}</span>'
            f'<span style="color:{warna}">{fmt_eq(abs(beta[i]))} × {var}</span>'
        )
    eq = (f'<span style="color:#cdd5e0">{dep}</span> '
          f'<span style="color:#9aa4b2">=</span> ' + "".join(bagian))
    st.markdown(f'<div class="eqbox">{eq}</div>', unsafe_allow_html=True)


def kartu(col, judul, nilai, sub, warna):
    col.markdown(
        f"""<div class="card" style="border-color:{warna}">
        <div class="ttl" style="color:{warna}">{judul}</div>
        <div class="val">{nilai}</div><div class="sub">{sub}</div></div>""",
        unsafe_allow_html=True,
    )


def warna_vif(v):
    if not np.isfinite(v):
        return "#888"
    if v >= 10:
        return "#e03131"   # serius
    if v >= 5:
        return "#e8590c"   # waspada
    return "#2f9e44"       # aman


def tabel_koefisien(hasil):
    rows = ""
    for i in range(1, len(hasil["nama"])):          # lewati intercept
        b = hasil["beta"][i]
        warna = "#2f9e44" if b >= 0 else "#e03131"
        tanda = "+" if b >= 0 else "−"
        bstd = hasil["beta_std"][i]
        bstd_txt = f"{bstd:+.3f}" if np.isfinite(bstd) else "–"
        v = hasil["vif"][i]
        vif_txt = f"{v:.2f}" if np.isfinite(v) else "–"
        rows += (
            f"<tr><td class='var'>{hasil['nama'][i]}</td>"
            f"<td style='color:{warna};font-weight:600'>{tanda}{fmt_tabel(abs(b))}</td>"
            f"<td>{bstd_txt}</td>"
            f"<td class='tcol'>{hasil['t'][i]:.2f}</td>"
            f"<td>{hasil['p'][i]:.4f}</td>"
            f"<td style='color:#2f9e44;font-weight:700'>{bintang(hasil['p'][i])}</td>"
            f"<td style='color:{warna_vif(v)};font-weight:600'>{vif_txt}</td></tr>"
        )
    tbl = (
        "<table class='coef'><tr><th>Variabel</th><th>Koefisien</th><th>β baku</th>"
        "<th>t-stat</th><th>p-value</th><th>Sig.</th><th>VIF</th></tr>" + rows + "</table>"
        "<div style='font-size:0.74rem;color:#888;margin-top:.4rem'>"
        "★★★ p&lt;0,01 · ★★ p&lt;0,05 · ★ p&lt;0,10 &nbsp;|&nbsp; "
        "VIF &lt;5 aman · 5–10 waspada · &gt;10 multikolinearitas serius &nbsp;|&nbsp; "
        "β baku = pengaruh dalam satuan simpangan baku (bisa dibandingkan antar variabel)</div>"
    )
    st.markdown(tbl, unsafe_allow_html=True)


def plot_actual_predicted(hasil, dep):
    y, yh = hasil["y"], hasil["y_hat"]
    lo, hi = float(min(y.min(), yh.min())), float(max(y.max(), yh.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect fit",
        line=dict(color="#e8590c", dash="dash", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=y, y=yh, mode="markers", name="Observasi",
        marker=dict(color="#4263eb", size=8, opacity=0.65),
        hovertemplate="Actual %{x:.1f}<br>Predicted %{y:.1f}<extra></extra>",
    ))
    fig.add_annotation(x=lo + 0.05 * (hi - lo), y=hi - 0.08 * (hi - lo),
                       text=f"<b>R² = {hasil['R2']:.3f}</b>", showarrow=False,
                       font=dict(color="#1c7ed6", size=16))
    fig.update_layout(
        height=440, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
        xaxis=dict(title=f"Actual {dep}", showgrid=True, gridcolor="#eee"),
        yaxis=dict(title=f"Predicted {dep}", showgrid=True, gridcolor="#eee"),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


def halaman_ols(df, kolom_numerik):
    st.title("📈 Regresi Linier OLS")

    c1, c2 = st.columns(2)
    with c1:
        dep = st.selectbox("1️⃣ Variabel dependen (Y)", kolom_numerik, key="ols_dep")
    kandidat = [k for k in kolom_numerik if k != dep]
    with c2:
        indep = st.multiselect("2️⃣ Variabel independen (X)", kandidat, default=kandidat, key="ols_indep")

    if not indep:
        st.warning("Pilih minimal satu variabel independen.")
        return

    # Bersihkan: konversi numerik -> ganti inf jadi NaN -> buang baris NaN
    data = (
        df[[dep] + indep]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if data.empty:
        st.error("Tidak ada baris data yang valid setelah membersihkan nilai kosong/tak hingga.")
        return

    # Buang prediktor dengan variansi nol (kolom konstan) -> penyebab umum error SVD
    konstan = [v for v in indep if data[v].nunique() <= 1]
    if konstan:
        st.warning(
            "⚠️ Variabel berikut bernilai konstan (variansi nol) sehingga dikeluarkan dari model: "
            f"**{', '.join(konstan)}**."
        )
        indep = [v for v in indep if v not in konstan]
    if not indep:
        st.error("Semua variabel independen terpilih bernilai konstan. Pilih variabel lain.")
        return
    if data[dep].nunique() <= 1:
        st.error(f"Variabel dependen «{dep}» bernilai konstan (tidak ada variasi untuk dijelaskan).")
        return

    if len(data) <= len(indep) + 1:
        st.error(f"Data valid ({len(data)} baris) terlalu sedikit untuk {len(indep)} prediktor. "
                 "Kurangi variabel atau lengkapi data.")
        return

    y = data[dep].to_numpy(dtype=float)
    X = data[indep].to_numpy(dtype=float)
    try:
        hasil = jalankan_ols(y, X, indep)
    except np.linalg.LinAlgError:
        st.error(
            "Perhitungan gagal konvergen — biasanya karena ada variabel yang **identik / "
            "kombinasi linier sempurna** dari variabel lain, atau skala nilai yang ekstrem. "
            "Coba kurangi salah satu variabel yang sangat berkorelasi (cek tab Korelasi), "
            "atau periksa apakah ada kolom duplikat."
        )
        return

    # --- Persamaan
    st.markdown("#### Persamaan Regresi")
    banner_persamaan(dep, hasil)
    st.write("")

    # --- Kartu metrik (termasuk MAE, RMSE yang diminta)
    pF_txt = "p<0.001" if hasil["pF"] < 0.001 else f"p={hasil['pF']:.3f}"
    cols = st.columns(6)
    kartu(cols[0], "R²", f"{hasil['R2']:.3f}", f"{hasil['R2']*100:.0f}% variansi dijelaskan", "#1c7ed6")
    kartu(cols[1], "Adj R²", f"{hasil['adjR2']:.3f}", "Koreksi jumlah prediktor", "#2b8a3e")
    kartu(cols[2], "MAE", fmt_eq(hasil["MAE"]), "Rata-rata galat absolut", "#e8590c")
    kartu(cols[3], "RMSE", fmt_eq(hasil["RMSE"]), "Akar galat kuadrat", "#c2255c")
    kartu(cols[4], "F-stat", pF_txt, "Signifikansi model", "#2f9e44")
    kartu(cols[5], "n", f"{hasil['n']}", "Observasi terpakai", "#f08c00")
    st.write("")

    # --- Koefisien & plot berdampingan
    kiri, kanan = st.columns([1, 1])
    with kiri:
        st.markdown("#### Koefisien Regresi")
        tabel_koefisien(hasil)
    with kanan:
        st.markdown("#### Actual vs Predicted")
        st.plotly_chart(plot_actual_predicted(hasil, dep), use_container_width=True)

    # --- Narasi
    st.markdown("#### 📝 Interpretasi")
    sig_vars = [hasil["nama"][i] for i in range(1, len(hasil["nama"])) if hasil["p"][i] < 0.05]
    kualitas = ("baik" if hasil["R2"] >= 0.6 else "moderat" if hasil["R2"] >= 0.3 else "rendah")
    teks = (
        f"Model menjelaskan **{hasil['R2']*100:.1f}%** variansi **{dep}** "
        f"(Adj R² = {hasil['adjR2']:.3f}) — kemampuan penjelas tergolong **{kualitas}**. "
        f"Secara keseluruhan model **{'signifikan' if hasil['pF'] < 0.05 else 'tidak signifikan'}** "
        f"(F-test, {pF_txt}). "
    )
    if sig_vars:
        teks += f"Prediktor signifikan (p<0,05): **{', '.join(sig_vars)}**. "
    else:
        teks += "Tidak ada prediktor yang signifikan pada α=0,05. "
    teks += (f"Rata-rata kesalahan prediksi (MAE) = **{fmt_eq(hasil['MAE'])}**, "
             f"RMSE = **{fmt_eq(hasil['RMSE'])}** (satuan {dep}).")

    # Prediktor paling berpengaruh berdasarkan β baku (lintas skala)
    idx_std = [i for i in range(1, len(hasil["nama"])) if np.isfinite(hasil["beta_std"][i])]
    if idx_std:
        i_kuat = max(idx_std, key=lambda i: abs(hasil["beta_std"][i]))
        teks += (f" Berdasarkan **koefisien baku (β)**, prediktor paling berpengaruh adalah "
                 f"**{hasil['nama'][i_kuat]}** (β = {hasil['beta_std'][i_kuat]:+.3f}) — "
                 f"ini pembanding yang adil karena tidak terpengaruh perbedaan satuan antar variabel.")
    st.markdown(f'<div class="catatan">{teks}</div>', unsafe_allow_html=True)

    # Peringatan VIF
    vif_serius = [(hasil["nama"][i], hasil["vif"][i]) for i in range(1, len(hasil["nama"]))
                  if np.isfinite(hasil["vif"][i]) and hasil["vif"][i] >= 10]
    vif_waspada = [(hasil["nama"][i], hasil["vif"][i]) for i in range(1, len(hasil["nama"]))
                   if np.isfinite(hasil["vif"][i]) and 5 <= hasil["vif"][i] < 10]
    if vif_serius:
        daftar = ", ".join(f"{n} (VIF={v:.1f})" for n, v in vif_serius)
        st.error(f"🚨 **Multikolinearitas serius (VIF ≥ 10):** {daftar}. "
                 "Koefisien & tanda dapat tidak stabil — pertimbangkan membuang salah satu "
                 "variabel yang berkorelasi tinggi.")
    elif vif_waspada:
        daftar = ", ".join(f"{n} (VIF={v:.1f})" for n, v in vif_waspada)
        st.warning(f"⚠️ **VIF 5–10 (perlu diwaspadai):** {daftar}.")
    elif len(hasil["nama"]) > 2:
        st.success("✅ VIF semua prediktor < 5 — tidak ada indikasi multikolinearitas yang mengganggu.")

    with st.expander("📖 Catatan asumsi OLS (penting)"):
        st.markdown(
            """
            - **Linearitas**: hubungan X–Y diasumsikan linier. Periksa plot Actual vs Predicted —
              titik harus menyebar di sekitar garis 45°.
            - **Independensi & homoskedastisitas residual**: ragam galat sebaiknya konstan.
            - **Multikolinearitas (VIF)**: VIF = 1/(1−R²ⱼ), R²ⱼ dari regresi tiap prediktor
              terhadap prediktor lain. Aturan praktis: VIF < 5 aman, 5–10 perlu diwaspadai,
              > 10 menandakan multikolinearitas serius yang membuat koefisien tidak stabil.
            - **β baku (standardized)**: koefisien setelah semua variabel diubah ke skor-z,
              sehingga besarnya **bisa dibandingkan langsung** antar prediktor walau satuannya berbeda.
            - **R² tidak menyiratkan kausalitas**; nilai tinggi bisa karena overfitting bila prediktor banyak —
              gunakan **Adj R²** sebagai pembanding.
            - **MAE vs RMSE**: RMSE lebih sensitif terhadap galat besar/outlier.
            """
        )

    # --- Unduh
    ringkasan = pd.DataFrame({
        "Variabel": hasil["nama"], "Koefisien": hasil["beta"],
        "Beta_baku": hasil["beta_std"], "Std.Error": hasil["se"],
        "t-stat": hasil["t"], "p-value": hasil["p"], "VIF": hasil["vif"],
    })
    metrik = pd.DataFrame({
        "Metrik": ["R2", "Adj R2", "MAE", "RMSE", "F-stat", "p(F)", "n"],
        "Nilai": [hasil["R2"], hasil["adjR2"], hasil["MAE"], hasil["RMSE"],
                  hasil["F"], hasil["pF"], hasil["n"]],
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as w:
        ringkasan.to_excel(w, sheet_name="Koefisien", index=False)
        metrik.to_excel(w, sheet_name="Metrik", index=False)
    st.download_button("⬇️ Unduh hasil regresi (Excel)", data=buffer.getvalue(),
                       file_name="hasil_regresi_ols.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ====================== HALAMAN 3: PATH / MODERASI ======================== #
def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _trunc(s, n=18):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def warna_jalur(b, p):
    """Hijau untuk positif-signifikan, merah untuk negatif-signifikan, abu untuk ns."""
    if not np.isfinite(b) or not np.isfinite(p):
        return "#adb5bd", "6,5"
    if p >= 0.05:
        return "#adb5bd", "6,5"          # tidak signifikan -> garis putus abu
    return ("#2f9e44" if b >= 0 else "#e03131"), "0"


def _node(x, y, w, h, judul, sub, fill, border):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
        f'stroke="{border}" stroke-width="2"/>'
        f'<text x="{x + w/2}" y="{y + h/2 - 3}" text-anchor="middle" '
        f'font-size="15" font-weight="700" fill="#1f2a44">{_esc(_trunc(judul))}</text>'
        f'<text x="{x + w/2}" y="{y + h/2 + 16}" text-anchor="middle" '
        f'font-size="11" fill="#666">{_esc(sub)}</text>'
    )


def svg_path_moderasi(Xf, mod, dep, bX, pX, bM, pM, bI, pI):
    """Diagram path statistik moderasi untuk satu variabel fokus."""
    cX, dX = warna_jalur(bX, pX)
    cM, dM = warna_jalur(bM, pM)
    cI, dI = warna_jalur(bI, pI)

    def lbl(b, p):
        return f"β={b:+.2f} {bintang(p)}" if np.isfinite(b) else "–"

    yx, ym, yi = 60, 165, 270           # pusat-y node kiri
    yY = 165
    svg = f'''
    <svg viewBox="0 0 780 340" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:780px">
      <defs>
        <marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3"
                orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L7,3 L0,6 Z" fill="context-stroke"/>
        </marker>
      </defs>
      {_node(30, yx-30, 190, 60, Xf, "Independen (X)", "#e7f0ff", "#1c7ed6")}
      {_node(30, ym-30, 190, 60, mod, "Moderator (M)", "#fff4e6", "#e8590c")}
      {_node(30, yi-30, 190, 60, f"{_trunc(Xf,8)} × {_trunc(mod,8)}", "Interaksi (X×M)", "#f3e8ff", "#9c36b5")}
      {_node(560, yY-30, 190, 60, dep, "Target (Y)", "#e6fcf5", "#0ca678")}

      <line x1="220" y1="{yx}" x2="558" y2="{yY-14}" stroke="{cX}" stroke-width="2.5"
            stroke-dasharray="{dX}" marker-end="url(#ah)"/>
      <text x="360" y="{(yx+yY)/2 - 16}" font-size="13" font-weight="600" fill="{cX}">{lbl(bX,pX)}</text>

      <line x1="220" y1="{ym}" x2="558" y2="{yY}" stroke="{cM}" stroke-width="2.5"
            stroke-dasharray="{dM}" marker-end="url(#ah)"/>
      <text x="330" y="{ym - 8}" font-size="13" font-weight="600" fill="{cM}">{lbl(bM,pM)}</text>

      <line x1="220" y1="{yi}" x2="558" y2="{yY+14}" stroke="{cI}" stroke-width="3.5"
            stroke-dasharray="{dI}" marker-end="url(#ah)"/>
      <text x="350" y="{(yi+yY)/2 + 26}" font-size="13" font-weight="700" fill="{cI}">{lbl(bI,pI)} ← moderasi</text>
    </svg>'''
    return svg


def svg_model_ideal():
    """Diagram konseptual moderasi (textbook)."""
    return '''
    <svg viewBox="0 0 760 270" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:760px">
      <defs>
        <marker id="ah2" markerWidth="9" markerHeight="9" refX="7" refY="3"
                orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L7,3 L0,6 Z" fill="context-stroke"/>
        </marker>
      </defs>
      <rect x="40" y="170" width="170" height="60" rx="10" fill="#e7f0ff" stroke="#1c7ed6" stroke-width="2"/>
      <text x="125" y="205" text-anchor="middle" font-size="15" font-weight="700" fill="#1f2a44">Independen (X)</text>
      <rect x="550" y="170" width="170" height="60" rx="10" fill="#e6fcf5" stroke="#0ca678" stroke-width="2"/>
      <text x="635" y="205" text-anchor="middle" font-size="15" font-weight="700" fill="#1f2a44">Target (Y)</text>
      <rect x="295" y="20" width="170" height="60" rx="10" fill="#fff4e6" stroke="#e8590c" stroke-width="2"/>
      <text x="380" y="55" text-anchor="middle" font-size="15" font-weight="700" fill="#1f2a44">Moderator (M)</text>

      <line x1="210" y1="200" x2="548" y2="200" stroke="#1f2a44" stroke-width="2.5" marker-end="url(#ah2)"/>
      <text x="360" y="190" text-anchor="middle" font-size="12" fill="#1f2a44">pengaruh utama (b₁)</text>

      <line x1="380" y1="82" x2="380" y2="198" stroke="#e8590c" stroke-width="3" stroke-dasharray="2,0" marker-end="url(#ah2)"/>
      <text x="392" y="140" font-size="12" font-weight="700" fill="#e8590c">interaksi (b₃)</text>
      <text x="392" y="156" font-size="11" fill="#888">M mengubah kekuatan X→Y</text>
    </svg>'''


def analisa_moderasi(data, dep, indep, mod):
    y = data[dep].to_numpy(dtype=float)
    means = {v: float(data[v].mean()) for v in indep + [mod]}
    Xc = {v: data[v].to_numpy(dtype=float) - means[v] for v in indep}
    Mc = data[mod].to_numpy(dtype=float) - means[mod]

    # Model penuh (dengan interaksi)
    cols, names = [], []
    for v in indep:
        cols.append(Xc[v]); names.append(v)
    cols.append(Mc); names.append(mod)
    for v in indep:
        cols.append(Xc[v] * Mc); names.append(f"{v} × {mod}")
    hasil = jalankan_ols(y, np.column_stack(cols), names)

    # Model tereduksi (tanpa interaksi) untuk ΔR²
    cols_r, names_r = [], []
    for v in indep:
        cols_r.append(Xc[v]); names_r.append(v)
    cols_r.append(Mc); names_r.append(mod)
    reduced = jalankan_ols(y, np.column_stack(cols_r), names_r)

    dR2 = hasil["R2"] - reduced["R2"]
    q = len(indep)
    n, kf = hasil["n"], hasil["k"]
    Fchg = (dR2 / q) / ((1 - hasil["R2"]) / (n - kf)) if (n - kf > 0 and hasil["R2"] < 1) else np.nan
    pchg = stats.f.sf(Fchg, q, n - kf) if np.isfinite(Fchg) and Fchg > 0 else np.nan

    return {"hasil": hasil, "reduced": reduced, "dR2": dR2, "Fchg": Fchg, "pchg": pchg,
            "means": means, "Xc": Xc, "Mc": Mc, "y": y, "indep": indep, "mod": mod}


def plot_simple_slopes(res, Xf, dep):
    hasil = res["hasil"]
    nama = hasil["nama"]
    b0 = hasil["beta"][0]
    bX = hasil["beta"][nama.index(Xf)]
    bM = hasil["beta"][nama.index(res["mod"])]
    bI = hasil["beta"][nama.index(f"{Xf} × {res['mod']}")]

    xc = res["Xc"][Xf]
    mc = res["Mc"]
    sd_m = float(mc.std(ddof=1))
    mean_xf = res["means"][Xf]
    xs_c = np.linspace(xc.min(), xc.max(), 50)
    xs_asli = xs_c + mean_xf

    fig = go.Figure()
    level = {"M rendah (−1 SD)": -sd_m, "M rata-rata": 0.0, "M tinggi (+1 SD)": sd_m}
    warna = {"M rendah (−1 SD)": "#1c7ed6", "M rata-rata": "#868e96", "M tinggi (+1 SD)": "#e8590c"}
    for label, mlv in level.items():
        y_pred = b0 + bX * xs_c + bM * mlv + bI * xs_c * mlv
        slope = bX + bI * mlv
        fig.add_trace(go.Scatter(
            x=xs_asli, y=y_pred, mode="lines", name=f"{label} (slope={slope:+.3g})",
            line=dict(color=warna[label], width=3),
        ))
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
        xaxis=dict(title=Xf, showgrid=True, gridcolor="#eee"),
        yaxis=dict(title=f"Prediksi {dep}", showgrid=True, gridcolor="#eee"),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig, bX, bI


def halaman_moderasi(df, kolom_numerik):
    st.title("🧭 Analisis Path / Moderasi")
    st.caption("Menguji apakah pengaruh X terhadap Y berubah tergantung tingkat moderator M.")

    c1, c2, c3 = st.columns(3)
    with c1:
        dep = st.selectbox("🎯 Target / dependen (Y)", kolom_numerik, key="mod_dep")
    sisa = [k for k in kolom_numerik if k != dep]
    with c2:
        indep = st.multiselect("➡️ Independen (X)", sisa, default=sisa[:1], key="mod_indep")
    with c3:
        kand_mod = [k for k in sisa if k not in indep]
        mod = st.selectbox("🔀 Moderator (M)", kand_mod, key="mod_mod") if kand_mod else None

    if not indep:
        st.warning("Pilih minimal satu variabel independen.")
        return
    if mod is None:
        st.warning("Tidak ada variabel tersisa untuk dijadikan moderator. Kurangi variabel independen.")
        return

    data = (df[[dep] + indep + [mod]].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan).dropna())
    konstan = [v for v in indep + [mod, dep] if data[v].nunique() <= 1]
    if konstan:
        st.error(f"Variabel konstan (variansi nol) tidak bisa dipakai: {', '.join(konstan)}.")
        return
    k_penuh = 1 + 2 * len(indep) + 1
    if len(data) <= k_penuh:
        st.error(f"Data valid ({len(data)} baris) terlalu sedikit untuk model interaksi "
                 f"({k_penuh} parameter). Kurangi variabel independen atau lengkapi data.")
        return

    try:
        res = analisa_moderasi(data, dep, indep, mod)
    except np.linalg.LinAlgError:
        st.error("Perhitungan gagal — kemungkinan ada variabel identik / kombinasi linier sempurna.")
        return
    hasil = res["hasil"]

    # --- Kartu metrik
    pchg_txt = "p<0.001" if (np.isfinite(res["pchg"]) and res["pchg"] < 0.001) else \
               (f"p={res['pchg']:.3f}" if np.isfinite(res["pchg"]) else "–")
    cols = st.columns(4)
    kartu(cols[0], "R² (penuh)", f"{hasil['R2']:.3f}", "Dengan interaksi", "#1c7ed6")
    kartu(cols[1], "Adj R²", f"{hasil['adjR2']:.3f}", "Koreksi prediktor", "#2b8a3e")
    kartu(cols[2], "ΔR² interaksi", f"{res['dR2']:.3f}", f"Sumbangan moderasi · {pchg_txt}", "#9c36b5")
    kartu(cols[3], "n", f"{hasil['n']}", "Observasi terpakai", "#f08c00")
    st.write("")

    # --- Variabel fokus untuk diagram & simple slopes
    Xf = st.selectbox("Variabel fokus untuk diagram & simple slopes", indep, key="mod_focal")
    nama = hasil["nama"]
    iXf, iM, iI = nama.index(Xf), nama.index(mod), nama.index(f"{Xf} × {mod}")

    kiri, kanan = st.columns([1, 1])
    with kiri:
        st.markdown("#### Path Diagram (variabel fokus)")
        st.markdown(
            svg_path_moderasi(
                Xf, mod, dep,
                hasil["beta_std"][iXf], hasil["p"][iXf],
                hasil["beta_std"][iM], hasil["p"][iM],
                hasil["beta_std"][iI], hasil["p"][iI],
            ),
            unsafe_allow_html=True,
        )
        st.caption("Garis tebal ungu = jalur interaksi (moderasi). "
                   "Garis putus-putus abu = jalur tidak signifikan (p≥0,05). β = koefisien baku.")
    with kanan:
        st.markdown("#### Simple Slopes")
        fig_ss, bX_ss, bI_ss = plot_simple_slopes(res, Xf, dep)
        st.plotly_chart(fig_ss, use_container_width=True)
        st.caption("Garis dengan kemiringan berbeda = moderasi nyata: efek X berubah menurut tingkat M.")

    # --- Tabel koefisien (model penuh)
    st.markdown("#### Koefisien Model Penuh")
    tabel_koefisien(hasil)

    # --- Narasi
    st.markdown("#### 📝 Interpretasi Moderasi")
    p_int = hasil["p"][iI]
    b_int = hasil["beta"][iI]
    if np.isfinite(p_int) and p_int < 0.05:
        arah_mod = "memperkuat" if b_int > 0 else "memperlemah"
        teks = (
            f"Suku interaksi **{Xf} × {mod}** **signifikan** (b = {fmt_tabel(b_int)}, p = {p_int:.4f}) "
            f"→ **moderasi terbukti**: semakin tinggi **{mod}**, hubungan **{Xf}→{dep}** cenderung "
            f"**{arah_mod}**. Interaksi menambah **{res['dR2']*100:.1f}%** variansi terjelaskan "
            f"({pchg_txt}). Karena ada interaksi signifikan, tafsirkan efek **{Xf}** lewat *simple slopes*, "
            f"bukan dari koefisien utamanya saja."
        )
        st.markdown(f'<div class="catatan">{teks}</div>', unsafe_allow_html=True)
        st.success("✅ Model moderasi sesuai dengan pola ideal: jalur interaksi bermakna.")
    else:
        teks = (
            f"Suku interaksi **{Xf} × {mod}** **tidak signifikan** "
            f"(b = {fmt_tabel(b_int)}, p = {p_int:.4f}) → **tidak ada bukti moderasi** oleh {mod} "
            f"pada hubungan {Xf}→{dep}. ΔR² hanya {res['dR2']*100:.1f}% ({pchg_txt}). "
            f"Untuk model yang lebih hemat, pertimbangkan model tanpa interaksi (cukup pengaruh utama)."
        )
        st.markdown(f'<div class="catatan">{teks}</div>', unsafe_allow_html=True)
        st.info("ℹ️ Jalur interaksi tidak bermakna — moderator ini belum tentu relevan untuk X tersebut.")

    # --- Model ideal
    st.markdown("#### 🧩 Seperti Apa Model Idealnya?")
    st.markdown(svg_model_ideal(), unsafe_allow_html=True)
    st.markdown(
        """
        Model moderasi **ideal** memenuhi ciri berikut:

        1. **Suku interaksi (b₃ = X×M) signifikan** — ini inti moderasi. Tanpa interaksi signifikan,
           yang ada hanya pengaruh aditif biasa, bukan moderasi.
        2. **ΔR² dari interaksi bermakna** — penambahan suku interaksi menaikkan variansi terjelaskan
           secara signifikan (uji F-change), bukan sekadar naik tipis karena tambahan parameter.
        3. **Variabel di-*mean-center*** sebelum membentuk interaksi (sudah dilakukan otomatis di sini),
           sehingga koefisien utama tetap bisa ditafsirkan dan VIF tidak meledak karena multikolinearitas semu.
        4. ***Simple slopes* berbeda nyata** antar tingkat M (−1 SD, rata-rata, +1 SD): garis yang
           kemiringannya berubah = bukti visual moderasi. Idealnya disertai uji signifikansi tiap slope.
        5. **Asumsi OLS terpenuhi** (linearitas, homoskedastisitas, residual ~normal) dan **tidak ada
           multikolinearitas berlebih** di luar yang melekat pada suku interaksi.
        6. **Berlandaskan teori** — moderasi sebaiknya dihipotesiskan lebih dulu (mengapa M masuk akal
           mengubah X→Y), bukan ditemukan secara *post-hoc* dari banyak percobaan (risiko *false positive*).

        > Catatan: ini menguji **moderasi** (efek interaksi). Bila yang Bapak maksud adalah **mediasi**
        > (X → M → Y, M sebagai perantara/jalur tidak langsung), strukturnya berbeda — bisa saya tambahkan
        > sebagai mode terpisah.
        """
    )

    # --- Unduh
    ringkasan = pd.DataFrame({
        "Jalur": hasil["nama"], "Koefisien": hasil["beta"], "Beta_baku": hasil["beta_std"],
        "Std.Error": hasil["se"], "t-stat": hasil["t"], "p-value": hasil["p"], "VIF": hasil["vif"],
    })
    metrik = pd.DataFrame({
        "Metrik": ["R2_penuh", "Adj R2", "R2_tanpa_interaksi", "Delta R2", "F_change", "p_change", "n"],
        "Nilai": [hasil["R2"], hasil["adjR2"], res["reduced"]["R2"], res["dR2"],
                  res["Fchg"], res["pchg"], hasil["n"]],
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as w:
        ringkasan.to_excel(w, sheet_name="Jalur", index=False)
        metrik.to_excel(w, sheet_name="Metrik", index=False)
    st.download_button("⬇️ Unduh hasil moderasi (Excel)", data=buffer.getvalue(),
                       file_name="hasil_moderasi_path.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ====================== HALAMAN 4: PATH ANALYSIS ========================== #
def _z(x):
    x = np.asarray(x, dtype=float)
    s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def _ols_coef(Xdesign, y):
    """Koefisien OLS via lstsq (untuk bootstrap cepat)."""
    return np.linalg.lstsq(Xdesign, y, rcond=None)[0]


def bootstrap_indirect(data, dep, indep, med, B=1000, seed=42):
    """CI persentil bootstrap untuk efek tidak langsung a_i*b tiap X."""
    arr = data[indep + [med, dep]].to_numpy(dtype=float)
    n, p = len(arr), len(indep)
    rng = np.random.default_rng(seed)
    out = np.full((B, p), np.nan)
    for bb in range(B):
        idx = rng.integers(0, n, n)
        sub = arr[idx]
        sd = sub.std(0)
        sd[sd == 0] = 1.0
        z = (sub - sub.mean(0)) / sd
        Xz, Mz, Yz = z[:, :p], z[:, p], z[:, p + 1]
        try:
            ba = _ols_coef(np.column_stack([np.ones(n), Xz]), Mz)        # [int, a1..ap]
            by = _ols_coef(np.column_stack([np.ones(n), Xz, Mz]), Yz)    # [int, c1..cp, b]
            bcoef = by[-1]
            out[bb, :] = ba[1:] * bcoef
        except np.linalg.LinAlgError:
            continue
    lo = np.nanpercentile(out, 2.5, axis=0)
    hi = np.nanpercentile(out, 97.5, axis=0)
    return {indep[i]: (float(lo[i]), float(hi[i])) for i in range(p)}


def path_analysis(data, dep, indep, med, B=1000):
    cols = indep + [med, dep]
    Z = {c: _z(data[c].to_numpy(dtype=float)) for c in cols}
    Xmat = np.column_stack([Z[v] for v in indep])

    hM = jalankan_ols(Z[med], Xmat, indep)                       # M ~ X
    XY = np.column_stack([Z[v] for v in indep] + [Z[med]])
    hY = jalankan_ols(Z[dep], XY, indep + [med])                 # Y ~ X + M

    a = {v: hM["beta"][hM["nama"].index(v)] for v in indep}
    pa = {v: hM["p"][hM["nama"].index(v)] for v in indep}
    b = float(hY["beta"][hY["nama"].index(med)])
    pb = float(hY["p"][hY["nama"].index(med)])
    cprime = {v: hY["beta"][hY["nama"].index(v)] for v in indep}
    pc = {v: hY["p"][hY["nama"].index(v)] for v in indep}
    indirect = {v: a[v] * b for v in indep}
    total = {v: cprime[v] + indirect[v] for v in indep}
    ci = bootstrap_indirect(data, dep, indep, med, B=B)

    return {"a": a, "pa": pa, "b": b, "pb": pb, "cprime": cprime, "pc": pc,
            "indirect": indirect, "total": total, "ci": ci,
            "R2_M": hM["R2"], "R2_Y": hY["R2"], "n": hM["n"], "B": B,
            "hM": hM, "hY": hY, "indep": indep, "med": med, "dep": dep}


def svg_path_analysis(res):
    indep, med, dep = res["indep"], res["med"], res["dep"]
    k = len(indep)
    NW, NH = 168, 46
    ROW, TOP = 74, 92
    H = max(250, TOP + k * ROW + 20)
    ymid = TOP + (k - 1) * ROW / 2 + NH / 2
    yM = 18
    xX, xM, xY = 18, 326, 632
    Mcx, Mcy = xM + NW / 2, yM + NH / 2

    def box(x, y, judul, sub, fill, border):
        return (
            f'<rect x="{x}" y="{y}" width="{NW}" height="{NH}" rx="9" fill="{fill}" stroke="{border}" stroke-width="2"/>'
            f'<text x="{x+NW/2}" y="{y+NH/2-2}" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1f2a44">{_esc(_trunc(judul,16))}</text>'
            f'<text x="{x+NW/2}" y="{y+NH/2+15}" text-anchor="middle" font-size="10" fill="#666">{_esc(sub)}</text>'
        )

    def arrow(x1, y1, x2, y2, b, p, w=2.3):
        c, d = warna_jalur(b, p)
        lbl = f"{b:+.2f}{bintang(p).replace('–','')}" if np.isfinite(b) else ""
        mx, my = x1 + (x2 - x1) * 0.42, y1 + (y2 - y1) * 0.42
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{w}" '
            f'stroke-dasharray="{d}" marker-end="url(#ah4)"/>'
            f'<text x="{mx}" y="{my-3}" font-size="11.5" font-weight="600" fill="{c}">{lbl}</text>'
        )

    parts = [f'<svg viewBox="0 0 820 {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:820px">',
             '<defs><marker id="ah4" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" '
             'markerUnits="strokeWidth"><path d="M0,0 L7,3 L0,6 Z" fill="context-stroke"/></marker></defs>']
    # nodes M & Y
    parts.append(box(xM, yM, med, "Antara / Moderator (M)", "#fff4e6", "#e8590c"))
    parts.append(box(xY, ymid - NH / 2, dep, "Target (Y)", "#e6fcf5", "#0ca678"))
    # b path: M -> Y
    parts.append(arrow(xM + NW, Mcy + 6, xY - 2, ymid - 12, res["b"], res["pb"], w=3.2))
    # X nodes + a (X->M) + c' (X->Y)
    for i, v in enumerate(indep):
        yX = TOP + i * ROW
        cyX = yX + NH / 2
        parts.append(box(xX, yX, v, "Independen (X)", "#e7f0ff", "#1c7ed6"))
        parts.append(arrow(xX + NW, cyX, xM - 2, Mcy + 8, res["a"][v], res["pa"][v]))          # a
        parts.append(arrow(xX + NW, cyX, xY - 2, ymid + 14 + (i - (k-1)/2) * 3, res["cprime"][v], res["pc"][v]))  # c'
    parts.append('</svg>')
    return "".join(parts)


def klas_mediasi(ci, c_sig):
    """Klasifikasi mediasi dari CI bootstrap efek tidak langsung & signifikansi jalur langsung."""
    lo, hi = ci
    indirect_sig = not (lo <= 0 <= hi)
    if indirect_sig and c_sig:
        return "Mediasi parsial", "sedang"
    if indirect_sig and not c_sig:
        return "Mediasi penuh", "kuat"
    if (not indirect_sig) and c_sig:
        return "Hanya efek langsung", "lemah"
    return "Tidak ada efek", "lemah"


def halaman_path(df, kolom_numerik):
    st.title("🕸️ Path Analysis (Analisis Jalur)")
    st.caption("Menggambar & menguji jalur antar variabel: X → M → Y, termasuk efek langsung, "
               "tidak langsung, dan total. Semua dalam koefisien terstandarisasi (β).")

    c1, c2, c3 = st.columns(3)
    with c1:
        dep = st.selectbox("🎯 Target / dependen (Y)", kolom_numerik, key="path_dep")
    sisa = [k for k in kolom_numerik if k != dep]
    with c2:
        indep = st.multiselect("➡️ Independen (X)", sisa, default=sisa[:2], key="path_indep")
    with c3:
        kand = [k for k in sisa if k not in indep]
        med = st.selectbox("🔀 Variabel antara / moderator (M)", kand, key="path_med") if kand else None

    if not indep:
        st.warning("Pilih minimal satu variabel independen.")
        return
    if med is None:
        st.warning("Tidak ada variabel tersisa untuk M. Kurangi variabel independen.")
        return

    data = (df[[dep] + indep + [med]].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan).dropna())
    konstan = [v for v in indep + [med, dep] if data[v].nunique() <= 1]
    if konstan:
        st.error(f"Variabel konstan tidak bisa dipakai: {', '.join(konstan)}.")
        return
    if len(data) <= len(indep) + 3:
        st.error(f"Data valid ({len(data)} baris) terlalu sedikit. Kurangi variabel atau lengkapi data.")
        return

    n_boot = st.select_slider("Jumlah resampling bootstrap (efek tidak langsung)",
                              options=[500, 1000, 2000, 5000], value=1000, key="path_boot")

    try:
        res = path_analysis(data, dep, indep, med, B=n_boot)
    except np.linalg.LinAlgError:
        st.error("Perhitungan gagal — kemungkinan ada variabel identik / kombinasi linier sempurna.")
        return

    # --- Kartu metrik
    cols = st.columns(4)
    kartu(cols[0], "R² (M)", f"{res['R2_M']:.3f}", f"X menjelaskan {res['med']}", "#e8590c")
    kartu(cols[1], "R² (Y)", f"{res['R2_Y']:.3f}", f"X+M menjelaskan {res['dep']}", "#0ca678")
    kartu(cols[2], "Jalur M→Y (b)", f"{res['b']:+.3f}", f"{bintang(res['pb'])}", "#9c36b5")
    kartu(cols[3], "n", f"{res['n']}", "Observasi terpakai", "#f08c00")
    st.write("")

    # --- Diagram jalur
    st.markdown("#### Diagram Jalur")
    st.markdown(svg_path_analysis(res), unsafe_allow_html=True)
    st.caption("Tiap panah berlabel koefisien jalur terstandarisasi (β). Garis tebal ungu = M→Y. "
               "Garis putus-putus abu = jalur tidak signifikan (p≥0,05). "
               "★★★ p<0,01 · ★★ p<0,05 · ★ p<0,10.")
    if len(indep) > 5:
        st.info("ℹ️ Banyak variabel X membuat diagram padat — gunakan tabel di bawah sebagai acuan utama.")

    # --- Tabel dekomposisi efek
    st.markdown("#### Dekomposisi Efek terhadap Target")
    rows = ""
    for v in indep:
        lo, hi = res["ci"][v]
        c_sig = np.isfinite(res["pc"][v]) and res["pc"][v] < 0.05
        label_med, kelas = klas_mediasi(res["ci"][v], c_sig)
        ind_sig = not (lo <= 0 <= hi)
        warna_ind = "#2f9e44" if ind_sig else "#adb5bd"
        rows += (
            f"<tr><td class='var'>{_esc(v)}</td>"
            f"<td>{res['a'][v]:+.3f} {bintang(res['pa'][v])}</td>"
            f"<td>{res['cprime'][v]:+.3f} {bintang(res['pc'][v])}</td>"
            f"<td style='color:{warna_ind};font-weight:600'>{res['indirect'][v]:+.3f}</td>"
            f"<td style='font-size:.85rem'>[{lo:+.3f}, {hi:+.3f}]</td>"
            f"<td>{res['total'][v]:+.3f}</td>"
            f"<td><b>{label_med}</b></td></tr>"
        )
    tbl = (
        "<table class='coef'><tr><th>X</th><th>a (X→M)</th><th>Langsung c′ (X→Y)</th>"
        "<th>Tidak langsung a·b</th><th>CI 95% (boot)</th><th>Total</th><th>Status</th></tr>"
        + rows + "</table>"
        "<div style='font-size:0.74rem;color:#888;margin-top:.4rem'>"
        "Efek tidak langsung = a·b (lewat M). <b>Signifikan bila CI 95% tidak memuat 0</b>. "
        "Mediasi penuh = tidak langsung signifikan & langsung tidak; parsial = keduanya signifikan.</div>"
    )
    st.markdown(tbl, unsafe_allow_html=True)

    # --- Narasi
    st.markdown("#### 📝 Interpretasi Jalur")
    mediator_sig = []
    for v in indep:
        lo, hi = res["ci"][v]
        if not (lo <= 0 <= hi):
            mediator_sig.append(v)
    teks = (
        f"Model jalur menjelaskan **{res['R2_Y']*100:.1f}%** variansi **{dep}** dan "
        f"**{res['R2_M']*100:.1f}%** variansi **{med}**. Jalur **{med}→{dep}** "
        f"(b = {res['b']:+.3f}, {bintang(res['pb'])}) "
        f"{'signifikan' if res['pb'] < 0.05 else 'tidak signifikan'}. "
    )
    if mediator_sig:
        teks += (f"Efek **{med}** memediasi pengaruh **{', '.join(mediator_sig)}** terhadap {dep} "
                 f"(efek tidak langsung signifikan secara bootstrap). ")
    else:
        teks += f"Tidak ada efek tidak langsung yang signifikan lewat {med}. "
    st.markdown(f'<div class="catatan">{teks}</div>', unsafe_allow_html=True)

    with st.expander("📖 Catatan path analysis (penting)"):
        st.markdown(
            """
            - **Jalur ditaksir lewat OLS** atas data ter-standarisasi (skor-z), jadi koefisien panah
              langsung dapat dibandingkan antar variabel.
            - **Efek tidak langsung = a·b**, diuji dengan **bootstrap persentil** (bukan Sobel) karena
              distribusi a·b tidak normal — pendekatan yang direkomendasikan literatur modern.
            - **Arah jalur berasal dari teori, bukan dari data.** Path analysis hanya menaksir kekuatan
              jalur yang *Anda* tetapkan; ia tidak membuktikan arah sebab-akibat.
            - Model ini **just-identified** (saturated) untuk struktur sederhana X→M→Y, sehingga indeks
              fit global (CFI/RMSEA) tidak informatif. Untuk model lebih kompleks dengan jalur dibatasi,
              gunakan SEM penuh (mis. lavaan/semopy).
            - Periksa juga **multikolinearitas** (tab Korelasi/OLS): X yang sangat berkorelasi membuat
              koefisien jalur tidak stabil.
            """
        )

    # --- Unduh
    dekom = pd.DataFrame({
        "X": indep,
        "a_X_ke_M": [res["a"][v] for v in indep],
        "p_a": [res["pa"][v] for v in indep],
        "langsung_cprime": [res["cprime"][v] for v in indep],
        "p_cprime": [res["pc"][v] for v in indep],
        "tidak_langsung_ab": [res["indirect"][v] for v in indep],
        "CI_low": [res["ci"][v][0] for v in indep],
        "CI_high": [res["ci"][v][1] for v in indep],
        "total": [res["total"][v] for v in indep],
    })
    info = pd.DataFrame({
        "Item": ["b_M_ke_Y", "p_b", "R2_M", "R2_Y", "n", "n_bootstrap"],
        "Nilai": [res["b"], res["pb"], res["R2_M"], res["R2_Y"], res["n"], res["B"]],
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as w:
        dekom.to_excel(w, sheet_name="Dekomposisi_Efek", index=False)
        info.to_excel(w, sheet_name="Info_Model", index=False)
    st.download_button("⬇️ Unduh hasil path analysis (Excel)", data=buffer.getvalue(),
                       file_name="hasil_path_analysis.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ================================ MAIN ==================================== #
with st.sidebar:
    st.header("⚙️ Pengaturan")
    berkas = st.file_uploader("Unggah file Excel (.xlsx)", type=["xlsx", "xls"])
    nama_metode = st.selectbox("Metode korelasi", list(METODE.keys()))
    metode = METODE[nama_metode]
    ambang_multikol = st.slider("Ambang multikolinearitas |r|", 0.5, 0.95, 0.70, 0.05)

if berkas is None:
    st.title("📊 Analisis Korelasi & Regresi OLS")
    st.info("⬅️ Unggah file Excel pada panel kiri untuk memulai.")
    st.markdown(
        """
        **Empat halaman tersedia (lihat tab di atas setelah data dimuat):**
        - **📊 Korelasi** — matriks, heatmap, narasi prediktor kuat, multikolinearitas.
        - **📈 Regresi OLS** — persamaan linier, R²/Adj R²/MAE/RMSE, koefisien (β baku & VIF), Actual vs Predicted.
        - **🧭 Moderasi** — uji interaksi X×M, path diagram, simple slopes, ΔR², model ideal.
        - **🕸️ Path Analysis** — jalur X→M→Y, efek langsung/tidak langsung/total (bootstrap), gambar diagram jalur.

        Pada tiap halaman Anda memilih variabel dependen lalu variabel independen.
        """
    )
    st.stop()

try:
    semua_sheet = baca_excel(berkas.getvalue())
except Exception as e:
    st.error(f"Gagal membaca file Excel: {e}")
    st.stop()

nama_sheet = st.sidebar.selectbox("Pilih sheet", list(semua_sheet.keys()))
df = semua_sheet[nama_sheet]

with st.expander("👀 Pratinjau data", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Dimensi: {df.shape[0]} baris × {df.shape[1]} kolom")

kolom_numerik = df.select_dtypes(include=[np.number]).columns.tolist()
if len(kolom_numerik) < 2:
    st.error("Diperlukan minimal 2 kolom numerik. Pastikan kolom angka tidak terbaca sebagai teks.")
    st.stop()

# Navigasi empat halaman lewat TAB (selalu terlihat di bagian atas)
tab_korr, tab_ols, tab_mod, tab_path = st.tabs(
    ["📊  Korelasi", "📈  Regresi OLS", "🧭  Moderasi", "🕸️  Path Analysis"]
)
with tab_korr:
    halaman_korelasi(df, kolom_numerik, metode, nama_metode, ambang_multikol)
with tab_ols:
    halaman_ols(df, kolom_numerik)
with tab_mod:
    halaman_moderasi(df, kolom_numerik)
with tab_path:
    halaman_path(df, kolom_numerik)