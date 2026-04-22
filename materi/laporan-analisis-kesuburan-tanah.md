# Laporan Analisis Kesuburan Tanah
## Ujian Tengah Semester — Penambangan Data
### Menggunakan KNIME Analytics Platform

---

## Daftar Isi
1. [Pendahuluan](#1-pendahuluan)
2. [Node 1 — Excel Reader (Import Data)](#2-node-1--excel-reader-import-data)
3. [Node 2 — Missing Value (Penanganan Nilai Hilang)](#3-node-2--missing-value-penanganan-nilai-hilang)
4. [Node 3 — One to Many (Encoding Kategorikal)](#4-node-3--one-to-many-encoding-kategorikal)
5. [Node 4 — Normalizer (Normalisasi Data)](#5-node-4--normalizer-normalisasi-data)
6. [Node 5 — Table Partitioner (Pembagian Data)](#6-node-5--table-partitioner-pembagian-data)
7. [Node 6 — K Nearest Neighbor (Klasifikasi KNN)](#7-node-6--k-nearest-neighbor-klasifikasi-knn)
8. [Node 7 — Scorer (Evaluasi Model)](#8-node-7--scorer-evaluasi-model)
9. [Alur Workflow Lengkap](#9-alur-workflow-lengkap)
10. [Kesimpulan](#10-kesimpulan)

---

## 1. Pendahuluan

Analisis kesuburan tanah merupakan salah satu aplikasi penting dalam bidang pertanian berbasis data. Pada penelitian ini, dilakukan proses **klasifikasi kesuburan tanah** menggunakan algoritma **K-Nearest Neighbor (KNN)** yang diimplementasikan melalui platform KNIME Analytics. Dataset yang digunakan adalah `dataset_kesuburan_tanah_missing.xlsx` yang terdiri dari **2.000 baris data** dan **12 kolom fitur**, mencakup berbagai parameter tanah seperti pH, kandungan nitrogen, fosfor, kalium, karbon organik, serta tekstur tanah.

Tujuan utama dari analisis ini adalah membangun model klasifikasi yang mampu memprediksi apakah suatu sampel tanah dikategorikan sebagai **"Subur"** atau **"Tidak Subur"** berdasarkan karakteristik kimiawi dan fisiknya.

### Deskripsi Dataset

| Kolom | Tipe Data | Deskripsi |
|---|---|---|
| ID | Integer | Identifikasi unik setiap sampel tanah |
| pH Tanah | Float | Tingkat keasaman tanah (skala 3.5 – 9.0) |
| N Total (%) | Float | Persentase kandungan Nitrogen total |
| P Tersedia (ppm) | Float | Kandungan Fosfor tersedia dalam ppm |
| K Tersedia (meq/100g) | Float | Kandungan Kalium tersedia |
| C Organik (%) | Float | Persentase kandungan Karbon Organik |
| KTK (meq/100g) | Float | Kapasitas Tukar Kation |
| Kejenuhan Basa (%) | Float | Persentase Kejenuhan Basa |
| Tekstur Tanah | String | Jenis tekstur tanah (Debu, Liat, Pasir, dll.) |
| Kadar Air (%) | Float | Persentase kadar air tanah |
| Bulk Density (g/cm³) | Float | Kerapatan tanah |
| Label | String | Kelas target: "Subur" / "Tidak Subur" |

### Distribusi Label
- **Subur**: 1.000 sampel (50%)
- **Tidak Subur**: 1.000 sampel (50%)

Data terdistribusi **seimbang (balanced)**, sehingga tidak diperlukan teknik penyeimbangan kelas tambahan seperti oversampling atau undersampling.

### Ringkasan Missing Value pada Dataset

| Kolom | Jumlah Missing |
|---|---|
| N Total (%) | 160 |
| P Tersedia (ppm) | 240 |
| K Tersedia (meq/100g) | 140 |
| C Organik (%) | 200 |
| Tekstur Tanah | 100 |
| Kadar Air (%) | 180 |
| Bulk Density (g/cm³) | 120 |

---

## 2. Node 1 — Excel Reader (Import Data)

### Screenshot KNIME

![Node 1 - Excel Reader](images/node1_excel_reader.png)

### Deskripsi Node

Node pertama dalam workflow ini adalah **Excel Reader**, yang berfungsi sebagai pintu masuk data mentah ke dalam lingkungan KNIME. Node ini membaca file `.xlsx` langsung dari sistem file lokal dan mengonversinya ke dalam format tabel internal KNIME yang dapat diproses oleh node-node berikutnya.

### Konfigurasi yang Digunakan

Berdasarkan screenshot di atas, konfigurasi Excel Reader yang diterapkan adalah sebagai berikut:

- **Read from**: Local File System — data dibaca dari sistem file lokal pengguna.
- **Mode**: File — membaca satu file tunggal (bukan folder).
- **File Path**: `C:\Users\kiyan\Downloads\dataset_kesuburan_tanah_missing.xlsx`
- **Select Sheet**: First with data — KNIME secara otomatis memilih sheet pertama yang mengandung data.

### Hasil Output

Setelah node dieksekusi, KNIME berhasil memuat dataset dengan rincian sebagai berikut:

- **Jumlah Baris**: 2.000 baris data
- **Jumlah Kolom**: 12 kolom
- Kolom bertipe **numerik (Double/Integer)** mencakup: `pH Tanah`, `N Total (%)`, `P Tersedia (ppm)`, `K Tersedia (meq/100g)`, `C Organik (%)`, `KTK (meq/100g)`, `Kejenuhan Basa (%)`, `Kadar Air (%)`, `Bulk Density (g/cm³)`.
- Kolom bertipe **String** mencakup: `Tekstur Tanah` dan `Label`.

### Observasi Awal

Dari preview data yang tampil di bagian bawah dialog pada screenshot, terlihat bahwa terdapat sejumlah sel yang ditandai dengan tanda tanya (`?`) — ini merupakan representasi KNIME untuk **missing value** (nilai hilang). Nilai hilang ditemukan tersebar di berbagai kolom numerik maupun kolom kategorikal (`Tekstur Tanah`). Hal ini mengonfirmasi bahwa dataset memerlukan tahap **pra-pemrosesan** sebelum dapat digunakan untuk pelatihan model.

---

## 3. Node 2 — Missing Value (Penanganan Nilai Hilang)

### Screenshot KNIME

![Node 2 - Missing Value](images/node2_missing_value.png)

### Deskripsi Node

Node **Missing Value** digunakan untuk menangani nilai-nilai yang hilang (missing values) dalam dataset. Keberadaan missing value dapat menyebabkan error pada node-node berikutnya, terutama pada algoritma machine learning seperti KNN yang tidak dapat memproses data yang tidak lengkap. Oleh karena itu, imputasi nilai hilang merupakan langkah yang sangat krusial dalam pra-pemrosesan data.

### Konfigurasi yang Digunakan

Berdasarkan screenshot di atas, konfigurasi yang diterapkan pada tab **Default** adalah:

| Tipe Data | Metode Imputasi |
|---|---|
| Number (Integer) | **Mean** (Rata-rata) |
| Number (Float) | **Do nothing** (Tidak dilakukan apapun) |
| String | **Most Frequent Value** (Modus) |

### Penjelasan Detail Setiap Metode

#### a) Number (Integer) — Mean

Untuk kolom bertipe bilangan bulat (integer), missing value diisi menggunakan **nilai rata-rata (mean)** dari kolom tersebut.

**Rumus Mean:**

$$\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}$$

Di mana:
- $\bar{x}$ = nilai rata-rata kolom
- $x_i$ = nilai data ke-i yang tidak hilang
- $n$ = jumlah data yang tidak hilang (valid)

#### b) Number (Float) — Do Nothing

Untuk kolom bertipe desimal (float), dipilih opsi **"Do nothing"**, yang berarti missing value pada kolom numerik float **dibiarkan apa adanya** dan tidak diimputasi. Ini menjadi penting karena node KNN di KNIME dapat menangani missing value pada fitur numerik dengan cara tertentu, atau kolom tersebut akan diabaikan dalam kalkulasi jarak.

> **Catatan**: Beberapa kolom float seperti `N Total (%)`, `P Tersedia (ppm)`, `C Organik (%)`, `Kadar Air (%)`, dan `Bulk Density (g/cm³)` yang memiliki missing value tetap memiliki tanda `(!)` merah di output tabel, menunjukkan bahwa nilai hilang tersebut masih ada setelah node ini dieksekusi.

#### c) String — Most Frequent Value (Modus)

Untuk kolom kategorikal bertipe String, yaitu `Tekstur Tanah`, missing value diisi menggunakan **nilai yang paling sering muncul (modus/most frequent value)**.

**Rumus Modus:**

$$\text{Modus} = \arg\max_{v} \, \text{count}(v)$$

Di mana $v$ adalah setiap nilai unik yang muncul dalam kolom tersebut, dan dipilih nilai yang memiliki frekuensi kemunculan tertinggi.

### Hasil Output

Setelah node Missing Value dieksekusi, output menampilkan tabel dengan:
- **Jumlah Baris**: 2.000 (tidak ada baris yang dihapus)
- **Jumlah Kolom**: 12 (tidak ada perubahan)
- Kolom `Tekstur Tanah` yang sebelumnya memiliki nilai hilang kini telah terisi dengan nilai modus.
- Beberapa kolom float masih memiliki indikator missing value karena konfigurasi "Do nothing".

---

## 4. Node 3 — One to Many (Encoding Kategorikal)

### Screenshot KNIME

![Node 3 - One to Many](images/node3_one_to_many.png)

### Deskripsi Node

Node **One to Many** digunakan untuk melakukan **One-Hot Encoding** terhadap kolom kategorikal (String). Algoritma KNN bekerja dengan perhitungan jarak matematis antar data point, sehingga tidak dapat memproses data berbentuk teks secara langsung. Oleh karena itu, kolom `Tekstur Tanah` yang berisi kategori teks perlu dikonversi menjadi representasi numerik biner.

### Konfigurasi yang Digunakan

Berdasarkan screenshot di atas, konfigurasi **One to Many** adalah:

- **Mode seleksi kolom**: Manual
- **Kolom yang tersedia (Available columns)**: Label
- **Kolom yang ditransformasi (Columns to transform)**: `Tekstur Tanah`

### Mekanisme One-Hot Encoding

One-Hot Encoding bekerja dengan cara membuat **kolom biner baru** untuk setiap kategori unik yang terdapat dalam kolom asli. Setiap kolom baru bernilai `1` jika baris tersebut termasuk kategori tersebut, dan `0` jika tidak.

**Contoh Ilustrasi:**

Misalkan `Tekstur Tanah` memiliki kategori: Debu, Liat, Pasir, Lempung, Lempung Berdebu, Lempung Berliat.

| Tekstur Tanah | Tekstur=Debu | Tekstur=Liat | Tekstur=Pasir | Tekstur=Lempung | ... |
|---|---|---|---|---|---|
| Debu | 1 | 0 | 0 | 0 | ... |
| Liat | 0 | 1 | 0 | 0 | ... |
| Pasir | 0 | 0 | 1 | 0 | ... |

**Rumus One-Hot Encoding:**

$$x_{i,k} = \begin{cases} 1 & \text{jika sampel ke-}i\text{ memiliki kategori }k \\ 0 & \text{jika sampel ke-}i\text{ tidak memiliki kategori }k \end{cases}$$

### Hasil Output

Setelah proses encoding:
- **Jumlah Baris**: 2.000 (tetap)
- **Jumlah Kolom**: **18 kolom** (bertambah dari 12 menjadi 18, karena kolom `Tekstur Tanah` dipecah menjadi beberapa kolom biner baru sesuai jumlah kategori uniknya)

Penambahan 6 kolom baru ini merepresentasikan kategori-kategori tekstur tanah yang ada dalam dataset (Debu, Liat, Pasir, Lempung, Lempung Berdebu, Lempung Berliat).

---

## 5. Node 4 — Normalizer (Normalisasi Data)

### Screenshot KNIME

![Node 4 - Normalizer](images/node4_normalizer.png)

### Deskripsi Node

Node **Normalizer** digunakan untuk melakukan **normalisasi** pada kolom-kolom numerik. Normalisasi sangat penting dalam algoritma KNN karena KNN menggunakan perhitungan **jarak Euclidean** antar data point. Jika skala antar fitur sangat berbeda (misalnya `KTK` berkisar 0–100, sementara `N Total (%)` berkisar 0–1), maka fitur dengan skala lebih besar akan mendominasi perhitungan jarak dan menyebabkan hasil klasifikasi yang bias.

### Konfigurasi yang Digunakan

Berdasarkan screenshot di atas, konfigurasi Normalizer adalah:

| Parameter | Nilai |
|---|---|
| Normalization method | **Min-max** |
| Minimum | **0** |
| Maximum | **1** |

### Rumus Normalisasi Min-Max

Metode **Min-Max Normalization** mentransformasi setiap nilai fitur ke dalam rentang [0, 1] menggunakan rumus berikut:

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Di mana:
- $x'$ = nilai hasil normalisasi
- $x$ = nilai asli dari suatu data point
- $x_{\min}$ = nilai minimum dari seluruh data pada kolom tersebut
- $x_{\max}$ = nilai maksimum dari seluruh data pada kolom tersebut

**Contoh Perhitungan Nyata:**

Untuk kolom `pH Tanah` dengan rentang nilai 3.5 hingga 9.0:
- Nilai asli pada Row0: $x = 8.93$
- $x_{\min} = 3.50$, $x_{\max} = 9.00$

$$x' = \frac{8.93 - 3.50}{9.00 - 3.50} = \frac{5.43}{5.50} \approx 0.987$$

Hasil ini sesuai dengan nilai yang terlihat di tabel output pada screenshot, yaitu **0.987** pada Row0 untuk kolom `pH Tanah`.

### Hasil Output

Setelah normalisasi:
- **Jumlah Baris**: 2.000 (tetap)
- **Jumlah Kolom**: 18 (tetap)
- Seluruh nilai kolom numerik kini berada dalam rentang **[0, 1]**
- Contoh hasil: pH Tanah Row0 = **0.987**, N Total Row0 = **0.353**, P Tersedia Row0 = **0.074**

---

## 6. Node 5 — Table Partitioner (Pembagian Data)

### Screenshot KNIME

![Node 5 - Table Partitioner](images/node5_table_partitioner.png)

### Deskripsi Node

Node **Table Partitioner** digunakan untuk membagi dataset menjadi dua bagian: **data training** (untuk melatih model) dan **data testing** (untuk menguji model). Pembagian ini esensial agar evaluasi model dilakukan secara objektif pada data yang belum pernah dilihat oleh model sebelumnya — sehingga hasil evaluasi mencerminkan kemampuan generalisasi model yang sesungguhnya.

### Konfigurasi yang Digunakan

Berdasarkan screenshot di atas, konfigurasi Table Partitioner adalah:

| Parameter | Nilai |
|---|---|
| First partition type | **Relative (%)** |
| Relative size | **80** |
| Sampling strategy | **Stratified** |
| Group column | **Label** |
| Fixed random seed | **1678807467440** |

### Penjelasan Detail Konfigurasi

#### a) Relative Size — 80%

Dataset dibagi dengan rasio **80:20**, di mana:
- **Partisi 1 (Training set)**: 80% dari 2.000 data = **1.600 baris**
- **Partisi 2 (Testing set)**: 20% dari 2.000 data = **400 baris**

**Rumus:**

$$n_{\text{train}} = \left\lfloor N \times \frac{r}{100} \right\rfloor = \left\lfloor 2000 \times 0.80 \right\rfloor = 1600$$

$$n_{\text{test}} = N - n_{\text{train}} = 2000 - 1600 = 400$$

Di mana $N$ adalah total jumlah data dan $r$ adalah persentase partisi pertama.

#### b) Stratified Sampling

Metode **Stratified** dipilih agar proporsi kelas pada data training dan testing **tetap seimbang** sesuai dengan distribusi pada dataset asli. Karena dataset memiliki distribusi label yang seimbang (50% Subur dan 50% Tidak Subur), maka:

- Training set: ≈ 800 Subur + 800 Tidak Subur
- Testing set: ≈ 200 Subur + 200 Tidak Subur

Ini mencegah bias akibat distribusi kelas yang tidak merata pada salah satu partisi.

#### c) Fixed Random Seed

Nilai seed **1678807467440** digunakan untuk memastikan **reproducibility** — artinya setiap kali workflow dijalankan ulang, pembagian data akan menghasilkan partisi yang sama persis. Hal ini penting untuk konsistensi eksperimen ilmiah.

### Hasil Output

- **Partisi 1 (Training set)**: **1.600 baris**, 18 kolom
- **Partisi 2 (Testing set)**: **400 baris**, 18 kolom

---

## 7. Node 6 — K Nearest Neighbor (Klasifikasi KNN)

### Screenshot KNIME

![Node 6 - K Nearest Neighbor](images/node6_knn.png)

### Deskripsi Node

Node **K Nearest Neighbor (KNN)** adalah inti dari workflow ini — merupakan algoritma klasifikasi yang digunakan untuk memprediksi label kesuburan tanah. KNN adalah algoritma **non-parametrik** berbasis instance yang mengklasifikasikan data baru berdasarkan kedekatan (kemiripan) dengan data-data yang sudah ada dalam training set.

### Konfigurasi yang Digunakan

Berdasarkan screenshot dialog K Nearest Neighbor di atas:

| Parameter | Nilai |
|---|---|
| Column with class labels | **Label** |
| Number of neighbours to consider (k) | **5** |
| Weight neighbours by distance | **Tidak dicentang (False)** |
| Output class probabilities | **Tidak dicentang (False)** |

### Mekanisme Kerja Algoritma KNN

#### Langkah 1 — Hitung Jarak Euclidean

Untuk setiap data testing, KNN menghitung jarak ke **seluruh data training** menggunakan **Euclidean Distance**:

$$d(x, x_i) = \sqrt{\sum_{j=1}^{p} (x_j - x_{i,j})^2}$$

Di mana:
- $x$ = data testing yang ingin diklasifikasi
- $x_i$ = data training ke-$i$
- $p$ = jumlah fitur (18 kolom setelah encoding dan normalisasi)
- $x_j$ = nilai fitur ke-$j$ dari data testing
- $x_{i,j}$ = nilai fitur ke-$j$ dari data training ke-$i$

#### Langkah 2 — Temukan K Tetangga Terdekat

Setelah semua jarak dihitung, algoritma mengurutkan data training dari yang **terdekat (jarak terkecil) ke terjauh**, kemudian memilih **k = 5** data training dengan jarak terkecil sebagai "tetangga terdekat".

#### Langkah 3 — Voting Mayoritas

KNN menentukan kelas prediksi berdasarkan **voting mayoritas** dari k tetangga terdekat:

$$\hat{y} = \arg\max_{c \in C} \sum_{i=1}^{k} \mathbf{1}[y_i = c]$$

Di mana:
- $\hat{y}$ = kelas prediksi
- $C$ = himpunan kelas yang mungkin {"Subur", "Tidak Subur"}
- $y_i$ = label dari tetangga ke-$i$
- $\mathbf{1}[y_i = c]$ = fungsi indikator (bernilai 1 jika tetangga ke-$i$ memiliki kelas $c$)

**Contoh:** Jika dari 5 tetangga terdekat, 4 berlabel "Subur" dan 1 berlabel "Tidak Subur", maka data testing tersebut diprediksi sebagai **"Subur"**.

#### Mengapa k = 5?

Nilai **k = 5** adalah pilihan umum yang memberikan keseimbangan antara:
- **Bias**: k terlalu kecil (misal k=1) membuat model sangat sensitif terhadap noise.
- **Variance**: k terlalu besar membuat model terlalu umum dan kehilangan kemampuan membedakan kelas.

Pemilihan k ganjil (5) juga mencegah **tie-breaking** (hasil seri) pada klasifikasi biner.

### Koneksi Node

Node KNN dalam KNIME menerima **dua input**:
- **Input 1 (training port)**: Data training dari **Partisi 1** Table Partitioner (1.600 baris)
- **Input 2 (test port)**: Data testing dari **Partisi 2** Table Partitioner (400 baris)

### Hasil Output

- **Output (Classified Data)**: **400 baris**, 19 kolom — bertambah 1 kolom baru yaitu `Class [kNN]` yang berisi prediksi kelas untuk setiap data testing.

---

## 8. Node 7 — Scorer (Evaluasi Model)

### Screenshot KNIME

![Node 7 - Scorer](images/node7_scorer.png)

### Deskripsi Node

Node **Scorer** digunakan untuk mengevaluasi performa model klasifikasi dengan cara membandingkan **label prediksi** dari model KNN dengan **label aktual** dari data testing. Output utama dari node ini adalah **Confusion Matrix** dan berbagai **metrik akurasi**.

### Konfigurasi yang Digunakan

Berdasarkan screenshot di atas, konfigurasi Scorer adalah:

| Parameter | Nilai |
|---|---|
| First column (Prediksi) | **Class [kNN]** |
| Second column (Aktual) | **Label** |
| Sorting strategy | **Insertion order** |
| Reverse order | Tidak dicentang |
| Use name prefix | Tidak dicentang |
| Missing values | **Ignore** |

### Hasil Confusion Matrix

Berdasarkan tabel confusion matrix yang terlihat pada screenshot di atas:

| | **Prediksi: Tidak Subur** | **Prediksi: Subur** |
|---|---|---|
| **Aktual: Tidak Subur** | **114** (True Negative) | **0** (False Positive) |
| **Aktual: Subur** | **0** (False Negative) | **110** (True Positive) |

### Perhitungan Metrik Evaluasi

#### a) Accuracy (Akurasi)

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{110 + 114}{110 + 114 + 0 + 0} = \frac{224}{224} = 1.0 = \mathbf{100\%}$$

#### b) Precision (Presisi) — untuk kelas "Subur"

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{110}{110 + 0} = 1.0 = \mathbf{100\%}$$

#### c) Recall (Sensitivitas) — untuk kelas "Subur"

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{110}{110 + 0} = 1.0 = \mathbf{100\%}$$

#### d) F1-Score

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{1.0 \times 1.0}{1.0 + 1.0} = \mathbf{1.0}$$

#### e) Spesifisitas — untuk kelas "Tidak Subur"

$$\text{Specificity} = \frac{TN}{TN + FP} = \frac{114}{114 + 0} = 1.0 = \mathbf{100\%}$$

### Interpretasi Hasil

Keterangan nilai pada confusion matrix:
- **TP (True Positive)** = 110: Model memprediksi "Subur" dan aktualnya memang "Subur".
- **TN (True Negative)** = 114: Model memprediksi "Tidak Subur" dan aktualnya memang "Tidak Subur".
- **FP (False Positive)** = 0: Tidak ada data "Tidak Subur" yang diprediksi "Subur".
- **FN (False Negative)** = 0: Tidak ada data "Subur" yang diprediksi "Tidak Subur".

> **Total data testing yang dievaluasi = 114 + 110 = 224 sampel** (bukan 400, karena sisanya memiliki missing value yang diabaikan sesuai konfigurasi "Ignore" pada Scorer).

Model KNN dengan **k = 5** berhasil mencapai **akurasi sempurna 100%** pada data testing. Hal ini menunjukkan bahwa model sangat baik dalam mengklasifikasikan kesuburan tanah berdasarkan fitur-fitur yang tersedia.

---

## 9. Alur Workflow Lengkap

Berikut adalah keseluruhan alur **workflow KNIME** yang dibangun untuk analisis kesuburan tanah:

```
┌──────────────┐     ┌───────────────┐     ┌─────────────┐     ┌────────────┐
│ Excel Reader │ ──► │ Missing Value │ ──► │ One to Many │ ──► │ Normalizer │
│  (Input data)│     │ (Imputasi     │     │ (One-Hot    │     │ (Min-Max   │
│  2000 baris  │     │  missing val) │     │  Encoding)  │     │  [0,1])    │
│  12 kolom    │     │  12 kolom     │     │  18 kolom   │     │  18 kolom  │
└──────────────┘     └───────────────┘     └─────────────┘     └─────┬──────┘
                                                                      │
                                                                      ▼
                                                          ┌───────────────────┐
                                                          │  Table Partitioner│
                                                          │  80% Train (1600) │
                                                          │  20% Test  (400)  │
                                                          └────────┬──────────┘
                                                                   │
                                              ┌────────────────────┴────────────┐
                                              │ Train (1600)         Test (400) │
                                              ▼                                 │
                                   ┌──────────────────┐                        │
                                   │  K Nearest       │ ◄──────────────────────┘
                                   │  Neighbor (k=5)  │
                                   │  Output: 400 brs │
                                   │  + kolom prediksi│
                                   └────────┬─────────┘
                                            │
                                            ▼
                                   ┌──────────────────┐
                                   │     Scorer       │
                                   │  Confusion Matrix│
                                   │  Accuracy: 100%  │
                                   └──────────────────┘
```

### Ringkasan Transformasi Data per Node

| Node | Input Baris | Input Kolom | Output Baris | Output Kolom | Fungsi Utama |
|---|---|---|---|---|---|
| Excel Reader | — | — | 2.000 | 12 | Import data mentah dari file xlsx |
| Missing Value | 2.000 | 12 | 2.000 | 12 | Imputasi nilai hilang |
| One to Many | 2.000 | 12 | 2.000 | 18 | Encoding kolom kategorikal |
| Normalizer | 2.000 | 18 | 2.000 | 18 | Normalisasi min-max ke [0,1] |
| Table Partitioner | 2.000 | 18 | 1.600 / 400 | 18 | Pembagian data 80:20 |
| K Nearest Neighbor | 1.600 (train) + 400 (test) | 18 | 400 | 19 | Klasifikasi KNN k=5 |
| Scorer | 400 | 19 | — | — | Evaluasi confusion matrix |

---

## 10. Kesimpulan

Berdasarkan seluruh tahapan analisis yang telah dilakukan menggunakan KNIME Analytics Platform, dapat disimpulkan beberapa hal penting sebagai berikut:

**1. Kualitas Data**
Dataset `dataset_kesuburan_tanah_missing.xlsx` memiliki 2.000 sampel dengan distribusi label yang seimbang (1.000 Subur dan 1.000 Tidak Subur). Terdapat missing value pada 7 kolom dengan total sekitar 1.140 nilai hilang (sekitar 5,7% dari total sel data).

**2. Pra-pemrosesan Data**
Penanganan missing value dilakukan dengan strategi berbeda berdasarkan tipe data: imputasi mean untuk integer, do nothing untuk float, dan modus untuk string. Kolom kategorikal `Tekstur Tanah` berhasil dikonversi menjadi representasi numerik melalui one-hot encoding, sehingga jumlah kolom bertambah dari 12 menjadi 18. Seluruh fitur numerik telah dinormalisasi menggunakan metode min-max agar berada dalam skala yang seragam [0, 1].

**3. Pembagian Data**
Dataset dibagi secara stratified dengan rasio 80:20, menghasilkan 1.600 data training dan 400 data testing dengan preservasi proporsi kelas.

**4. Performa Model**
Algoritma K-Nearest Neighbor dengan parameter k=5 berhasil mengklasifikasikan kesuburan tanah dengan **akurasi 100%** pada data testing, dibuktikan dengan confusion matrix yang menunjukkan 114 prediksi benar untuk kelas "Tidak Subur" dan 110 prediksi benar untuk kelas "Subur", tanpa adanya kesalahan klasifikasi sama sekali.

**5. Rekomendasi**
Meskipun akurasi yang dicapai sangat tinggi, disarankan untuk:
- Melakukan validasi silang (cross-validation) guna memastikan generalisasi model.
- Mencoba nilai k yang berbeda untuk mengoptimalkan performa.
- Mengeksplorasi penggunaan metode imputasi yang lebih canggih untuk kolom float yang saat ini dibiarkan dengan missing value.
- Mempertimbangkan algoritma lain sebagai pembanding (misalnya Decision Tree atau Random Forest).

---

*Laporan ini dibuat berdasarkan analisis workflow KNIME untuk keperluan Ujian Tengah Semester Penambangan Data — Analisis Kesuburan Tanah.*