import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Hızlı, bellek açısından güvenli simülasyon
# ----------------------------
# Bu sürüm, makaledeki matematiğe uyar ancak
# O(N^3 * L) tam hesaplamadan kaçınır; (i) |k|-kabukları arasında
# k-vektörlerinin tabakalı örneklenmesi ve (ii) belleği sınırlamak için partiler halinde işleme ile.
#
# Aşağıdaki ayarları değiştirerek doğruluk ile hız arasında
# denge kurabilirsiniz.
# ===== Kullanıcı ayarları =====
N = 64                 # eksen başına ızgara boyutu (T^3 ayrıklaştırma)
L = 4000               # sınır örneklerinin sayısı (hız için 1e4’ten düşürüldü)
realizations = 5       # ortalaması alınacak gerçekleştirme sayısı
sigma = 1.0            # zarf genişliği
c1 = 1                 # holonomi indeksi
S_per_bin = 150        # |k|-kutusu başına maksimum k-örneği sayısı (tabakalı örnekleme)
KBATCH = 256           # bellek açısından güvenli matris çarpımları için k-parti boyutu
dtype = np.float32     # bellek/hız için float32 kullan

# ===== Türetilmiş dalga vektörü ızgarası =====

k_x = np.arange(-N/2, N/2)
k_y = np.arange(-N/2, N/2)
k_z = np.arange(-N/2, N/2)
k_grid = np.array(np.meshgrid(k_x, k_y, k_z, indexing='ij')).reshape(3, -1).T.astype(dtype)  # (K,3)
num_k = len(k_grid)
k_magnitudes = np.sqrt(np.sum(k_grid**2, axis=1))
k_bins = np.arange(0, np.max(k_magnitudes) + 1, 1, dtype=dtype)
num_bins = len(k_bins) - 1

# ===== |k|-kabukları boyunca k’nin tabakalı örneklenmesi =====

rng = np.random.default_rng()
sampled_indices = []
bin_membership = [[] for _ in range(num_bins)]

for i in range(num_bins):
    mask = (k_magnitudes >= k_bins[i]) & (k_magnitudes < k_bins[i + 1])
    idx = np.flatnonzero(mask)

    if idx.size == 0:
        continue

    take = min(S_per_bin, idx.size)
    chosen = rng.choice(idx, size=take, replace=False)
    sampled_indices.append(chosen)
    bin_membership[i] = chosen

if len(sampled_indices) == 0:
    raise RuntimeError("Hiç k-vektörü örneklenmedi; S_per_bin’i azaltmayı veya N’i artırmayı deneyin.")

sampled_indices = np.concatenate(sampled_indices)

k_samp = k_grid[sampled_indices]                         # (Ks,3)

k_samp_mag = k_magnitudes[sampled_indices]               # (Ks,)

Ks = k_samp.shape[0]

# ===== Zarf =====

def G_hat(k_mag):
    k_mag = np.asarray(k_mag, dtype=dtype)
    return np.exp(-(k_mag**2) / (2 * (dtype(sigma)**2))).astype(dtype)

G_k_samp = G_hat(k_samp_mag)

# ===== Sınır örnekleri (rastgele taban + holonomi) =====

beta = rng.random((L, 3), dtype=dtype)                   # (L,3) [0,1) aralığında
X_beta = np.mod(beta - dtype(0.5) * dtype(c1) * beta[:, [0]] * beta[:, [1]], 1.0).astype(dtype)  # (L,3)

# ===== Örneklenmiş k’ler üzerinde biriktiriciler =====
P_S_acc = np.zeros(Ks, dtype=dtype)
P_H_acc = np.zeros(Ks, dtype=dtype)
sqrtL = np.sqrt(L).astype(dtype)

# ===== Gerçekleştirmeler üzerinde ana döngü =====
for _ in range(realizations):
    # karmaşık gürültü ve holonomi
    xi = (rng.normal(0, 1, L) + 1j * rng.normal(0, 1, L)).astype(np.complex64)
    U = np.exp(1j * 2 * np.pi * rng.random(L)).astype(np.complex64)
    # belleği kontrol etmek için k üzerinde parti halinde işlem
    for start in range(0, Ks, KBATCH):
        end = min(start + KBATCH, Ks)
        Kblk = k_samp[start:end]                          # (Kb,3)

        # dot(K, X_beta.T): (Kb,L), hız için float32’de hesaplandı

        dot_block = (Kblk @ X_beta.T).astype(dtype)       # (Kb,L)

        phase_block = np.exp(-2j * np.pi * dot_block).astype(np.complex64)  # (Kb,L)

        # tüm blok için B_S, B_H: (Kb,)

        B_S_blk = (phase_block @ xi) / sqrtL

        B_H_blk = (phase_block @ (U * xi)) / sqrtL

        # blok için öncül spektrumlar

        P_S_blk = (np.abs(B_S_blk)**2).astype(dtype)

        P_H_blk = (np.real(B_H_blk * np.conj(B_S_blk))).astype(dtype)

        # helisel temelde pozitiflik projeksiyonu

        lambda_plus  = (G_k_samp[start:end]**2) * (P_S_blk + P_H_blk)

        lambda_minus = (G_k_samp[start:end]**2) * (P_S_blk - P_H_blk)

        lambda_plus  = np.maximum(lambda_plus,  dtype(0))

        lambda_minus = np.maximum(lambda_minus, dtype(0))

        P_S_proj = (lambda_plus + lambda_minus) / (2 * (G_k_samp[start:end]**2 + dtype(1e-30)))

        P_H_proj = (lambda_plus - lambda_minus) / (2 * (G_k_samp[start:end]**2 + dtype(1e-30)))

        P_S_acc[start:end] += P_S_proj / realizations

        P_H_acc[start:end] += P_H_proj / realizations

# ===== Örneklenmiş sonuçları tekrar |k|-kabuklarına kutula =====

P_S_binned = np.zeros(num_bins, dtype=dtype)
P_H_binned = np.zeros(num_bins, dtype=dtype)

for i in range(num_bins):
    chosen = bin_membership[i]
    if len(chosen) == 0:
        P_S_binned[i] = 0
        P_H_binned[i] = 0
    else:
        pass

# Tam k-indisinden -> örneklenmiş indise bir lookup haritası oluştur

lookup = -np.ones(num_k, dtype=np.int32)

lookup[sampled_indices] = np.arange(Ks, dtype=np.int32)

for i in range(num_bins):
    chosen_full = bin_membership[i]
    if len(chosen_full) == 0:
        continue

    chosen_samp = lookup[chosen_full]
    chosen_samp = chosen_samp[chosen_samp >= 0]

    if chosen_samp.size == 0:
        continue

    P_S_binned[i] = np.mean(P_S_acc[chosen_samp])
    P_H_binned[i] = np.mean(P_H_acc[chosen_samp])

# Çizim için kutu merkezlerini kullan

k_centers = (k_bins[:-1] + k_bins[1:]) / 2

# Son zarf uygulaması (görüntüleme)

P_S_binned_disp = P_S_binned * (G_hat(k_centers)**2)

P_H_binned_disp = P_H_binned * (G_hat(k_centers)**2)

# ===== Grafik =====

plt.figure(figsize=(9, 5.5))
plt.plot(k_centers, P_S_binned_disp, label=r'$P_S(|k|)$')
plt.plot(k_centers, P_H_binned_disp, '--', label=r'$P_H(|k|)$')
plt.xlabel(r'$|k|$')
plt.ylabel('Güç Spektrumları')
plt.title(fr'Simüle Edilmiş Spektrumlar (tabakalı örnekleme)  $c_1={c1}$, $N={N}$, $L={L}$, $R={realizations}$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('spectra_plot.pdf', bbox_inches='tight')

plt.show()

# ===== Tanılamalar (kutulanmış görüntüleme dizilerine dayalı) =====

deviation = float(np.mean(np.abs(P_S_binned_disp - np.mean(P_S_binned_disp))))
violation_count = int(np.sum(P_S_binned_disp < np.abs(P_H_binned_disp)))
bias = float(np.mean(np.abs(P_H_binned_disp[P_S_binned_disp < np.abs(P_H_binned_disp)]))) if violation_count > 0 else 0.0
print(f"Ortalama sapma: {deviation:.4e} (~ 1/sqrt(N_shell · R))")
print(f"İhlal yüzdesi: {100 * violation_count / max(1, len(P_S_binned_disp)):.2f}%")
print(f"Projeksiyon yanlılığı: {bias:.4e}")
print(f"Kullanılan toplam k-örnekleri: {Ks} / {num_k}")
print(f"Şekil spectra_plot.pdf olarak kaydedildi")
