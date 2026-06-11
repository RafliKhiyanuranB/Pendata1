# TUGAS ANALISA DATA
## Reproduksi: Explainability pada Model Forecasting
### Sumber: https://skforecast.org/0.15.1/user_guides/explainability.html

# ❓ Pertanyaan 1: Analisa Prediksi Tentang Apa?

Model pada tutorial ini melakukan prediksi terhadap **permintaan energi listrik harian (Daily Electricity Demand)** di negara bagian **Victoria, Australia**.

Secara spesifik:
- **Yang diprediksi:** Total konsumsi listrik (dalam satuan MWh) untuk hari berikutnya
- **Tujuan:** Membantu pengelola jaringan listrik memperkirakan beban permintaan di masa depan agar pasokan listrik bisa direncanakan dengan efisien
- **Konteks:** Permintaan listrik sangat dipengaruhi oleh suhu udara (panas = AC menyala, dingin = pemanas menyala) dan hari dalam seminggu (hari kerja vs akhir pekan)

---

# ❓ Pertanyaan 2: Bentuk Data Training

## 2.1 Sumber Dataset

Dataset yang digunakan bernama **`vic_electricity`**, yaitu data permintaan listrik setengah-jam (*half-hourly*) dari Victoria, Australia.

| Properti | Nilai |
|----------|-------|
| Nama dataset | `vic_electricity` |
| Ukuran asli | 52.608 baris × 4 kolom |
| Frekuensi asli | Setiap 30 menit |
| Frekuensi setelah agregasi | Harian (per hari) |
| Kolom asli | Time, Demand, Temperature, Holiday |

## 2.2 Proses Preprocessing

Data setengah-jam diagregasi ke frekuensi **harian** dengan cara:
- Kolom `Demand` → **dijumlahkan** (total konsumsi per hari)
- Kolom `Temperature` → **dirata-rata** (suhu rata-rata per hari)
- Ditambahkan kolom `day_of_week` (0 = Senin, 6 = Minggu)

## 2.3 Struktur Data Training (Input & Output)

Setelah forecaster dibuat dengan **7 lag**, data training secara otomatis dibentuk menjadi:

### ✅ INPUT (Fitur / X_train)

| Kolom | Tipe | Penjelasan |
|-------|------|-----------|
| `lag_1` | Numerik | Nilai Demand **1 hari** sebelumnya (t-1) |
| `lag_2` | Numerik | Nilai Demand **2 hari** sebelumnya (t-2) |
| `lag_3` | Numerik | Nilai Demand **3 hari** sebelumnya (t-3) |
| `lag_4` | Numerik | Nilai Demand **4 hari** sebelumnya (t-4) |
| `lag_5` | Numerik | Nilai Demand **5 hari** sebelumnya (t-5) |
| `lag_6` | Numerik | Nilai Demand **6 hari** sebelumnya (t-6) |
| `lag_7` | Numerik | Nilai Demand **7 hari** sebelumnya (t-7) |
| `Temperature` | Numerik | Suhu rata-rata hari tersebut (variabel eksogen) |
| `day_of_week` | Numerik | Hari dalam seminggu: 0=Senin … 6=Minggu |

### ✅ OUTPUT (Target / y_train)

| Kolom | Tipe | Penjelasan |
|-------|------|-----------|
| `Demand` | Numerik | Total permintaan listrik **hari berikutnya** (MWh) |

### Ilustrasi Logika Data Training

```
Tanggal       | lag_7 | lag_6 | lag_5 | lag_4 | lag_3 | lag_2 | lag_1 | Temp | dow || TARGET (Demand)
--------------|-------|-------|-------|-------|-------|-------|-------|------|-----||----------------
2012-01-09    | D(2)  | D(3)  | D(4)  | D(5)  | D(6)  | D(7)  | D(8)  | 22.1 |  0  || D(9)
2012-01-10    | D(3)  | D(4)  | D(5)  | D(6)  | D(7)  | D(8)  | D(9)  | 24.3 |  1  || D(10)
...
```
> `D(n)` = nilai Demand pada tanggal ke-n di bulan Januari 2012

---

# ❓ Pertanyaan 3: Apa Itu Lag?

## Definisi

**Lag** adalah nilai dari variabel target (atau variabel lain) pada **waktu lampau** yang digunakan sebagai fitur (input) untuk melatih model prediksi.

Dalam konteks time series forecasting:

```
Waktu:   t-7   t-6   t-5   t-4   t-3   t-2   t-1   →   t (yang ingin diprediksi)
         ↑     ↑     ↑     ↑     ↑     ↑     ↑
        lag_7 lag_6 lag_5 lag_4 lag_3 lag_2 lag_1
```

## Mengapa Lag Diperlukan?

Model machine learning (seperti LightGBM, Random Forest, dll.) **tidak bisa langsung memahami urutan waktu**. Lag adalah cara kita "mengubah" data deret waktu menjadi format tabular yang bisa dipahami oleh model biasa.

Tanpa lag, model tidak tahu bahwa data hari ini berkaitan dengan data kemarin.

## Contoh Nyata

Jika kita ingin memprediksi Demand listrik hari Kamis:
- `lag_1` = Demand hari Rabu (kemarin)
- `lag_2` = Demand hari Selasa
- `lag_3` = Demand hari Senin
- `lag_4` = Demand hari Minggu
- `lag_5` = Demand hari Sabtu
- `lag_6` = Demand hari Jumat
- `lag_7` = Demand hari Kamis minggu lalu

## Konsep Autoregression

Penggunaan lag seperti ini disebut **autoregression** — model menggunakan **nilai dirinya sendiri di masa lalu** untuk memprediksi nilai di masa depan. Inilah inti dari `ForecasterRecursive` di skforecast.

---

# ❓ Pertanyaan 4: Proses Analisis yang Dilakukan

## Gambaran Umum Alur Kerja

```
[DATA MENTAH]
(half-hourly, 52.608 baris)
       ↓
[PREPROCESSING]
Agregasi harian + tambah fitur kalender
       ↓
[SPLIT DATA]
Train (2012–2014) | Test (2015)
       ↓
[BUAT & LATIH MODEL]
ForecasterRecursive + LGBMRegressor + 7 lags
       ↓
[EXPLAINABILITY ANALYSIS]
  ├── Feature Importance (bawaan model)
  ├── Permutation Importance
  ├── SHAP Values (Summary, Bar, Force Plot)
  └── Partial Dependence Plot (PDP)
```

---

## Tahap 1 — Load dan Preprocessing Data

Data diambil menggunakan fungsi `fetch_dataset("vic_electricity")` dari library skforecast. Data asli memiliki frekuensi setengah-jam, lalu diproses:

```python
data = data.resample('D').agg({'Demand': 'sum', 'Temperature': 'mean'})
data['day_of_week'] = data.index.dayofweek
```

Hasil: data harian dengan 3 kolom → `Demand`, `Temperature`, `day_of_week`.

---

## Tahap 2 — Split Data Train dan Test

```python
end_train = '2014-12-31'
data_train = data.loc[:end_train]   # 2012–2014 untuk training
data_test  = data.loc[end_train:]   # 2015 untuk testing
```

---

## Tahap 3 — Membuat dan Melatih Model Forecasting

```python
forecaster = ForecasterRecursive(
    regressor = LGBMRegressor(random_state=15926, verbose=-1),
    lags      = 7
)

forecaster.fit(
    y    = data_train['Demand'],
    exog = data_train[['Temperature', 'day_of_week']]
)
```

Model **LightGBM** dipilih karena termasuk algoritma *gradient boosting* berbasis pohon yang efisien dan mendukung SHAP TreeExplainer.

---

## Tahap 4 — Feature Importance (Metode 1)

**Feature Importance bawaan model** menghitung seberapa sering dan seberapa besar suatu fitur digunakan dalam pembentukan pohon keputusan di LightGBM.

```python
feature_importance = forecaster.get_feature_importances()
```

**Interpretasi:** Fitur dengan nilai importance lebih tinggi berarti lebih sering digunakan model untuk membuat keputusan split di pohon.

**Keterbatasan:** Metode ini bisa bias terhadap fitur dengan banyak nilai unik, sehingga perlu dikonfirmasi dengan metode lain.

---

## Tahap 5 — Permutation Importance (Metode 2)

**Permutation Importance** mengukur penurunan akurasi model ketika nilai satu fitur diacak secara acak (permutasi). Jika mengacak fitur A menyebabkan akurasi turun drastis → fitur A sangat penting.

```python
result = permutation_importance(
    estimator    = forecaster.regressor,
    X            = X_train,
    y            = y_train,
    n_repeats    = 10,
    random_state = 123
)
```

**Kelebihan dibanding Feature Importance biasa:** Lebih model-agnostic dan tidak bias terhadap tipe data fitur.

---

## Tahap 6 — SHAP Values (Metode 3 — Paling Detail)

**SHAP (SHapley Additive exPlanations)** adalah metode penjelasan model berbasis teori permainan (*game theory*). SHAP menghitung kontribusi tiap fitur terhadap setiap prediksi individual.

### Cara Kerja SHAP (Sederhana)
Bayangkan setiap fitur adalah "pemain" dalam sebuah tim. SHAP menghitung berapa "kontribusi adil" setiap pemain terhadap total "skor" (nilai prediksi), dengan mencoba semua kemungkinan kombinasi pemain.

### Komponen SHAP yang Digunakan

#### a) SHAP Summary Plot (Beeswarm)
```python
shap.summary_plot(shap_values, X_train)
```
- Sumbu Y = nama fitur (diurutkan dari paling penting ke bawah)
- Sumbu X = nilai SHAP (positif = mendorong prediksi naik, negatif = mendorong turun)
- Warna titik = nilai fitur (merah = tinggi, biru = rendah)
- **Cara baca:** Fitur di atas = lebih berpengaruh. Titik merah di kanan = nilai fitur tinggi mendorong prediksi naik.

#### b) SHAP Bar Plot (Global Importance)
```python
shap.summary_plot(shap_values, X_train, plot_type="bar")
```
- Menampilkan rata-rata nilai |SHAP| per fitur
- Lebih mudah dibaca, tapi tidak menampilkan arah pengaruh

#### c) SHAP Force Plot (Lokal — per prediksi)
```python
shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:])
```
- Menjelaskan **satu prediksi spesifik**
- Menampilkan fitur mana yang "mendorong naik" (merah) dan "mendorong turun" (biru) dari nilai dasar (base value)

---

## Tahap 7 — Partial Dependence Plot / PDP (Metode 4)

**PDP** menunjukkan hubungan rata-rata antara satu fitur dan target, sementara fitur lain dikontrol tetap.

```python
PartialDependenceDisplay.from_estimator(
    estimator = forecaster.regressor,
    X         = X_train,
    features  = ['Temperature']
)
```

**Interpretasi:** 
- Jika PDP Temperature berbentuk U (nilai rendah & tinggi → Demand naik) → ini masuk akal karena cuaca dingin dan panas sama-sama meningkatkan konsumsi listrik (pemanas vs AC).

---

## Ringkasan Keseluruhan Analisis

| Tahap | Metode | Tujuan |
|-------|--------|--------|
| 1 | Load & Preprocessing | Siapkan data harian + fitur kalender |
| 2 | Train/Test Split | Pisahkan data latih dan uji |
| 3 | ForecasterRecursive fit | Latih model prediksi dengan 7 lag |
| 4 | Feature Importance | Lihat fitur terpenting secara global (versi model) |
| 5 | Permutation Importance | Konfirmasi importance dengan metode lebih robust |
| 6 | SHAP Values | Jelaskan tiap prediksi secara individual & global |
| 7 | PDP | Visualisasi hubungan satu fitur vs target |

**Kesimpulan utama dari analisis:** Fitur `Temperature` dan `lag_1` (Demand kemarin) adalah dua faktor paling berpengaruh terhadap prediksi permintaan listrik harian di Victoria, Australia.

---

*Dokumen ini merupakan penjelasan tugas Penambangan Data — reproduksi dari https://skforecast.org/0.15.1/user_guides/explainability.html*
