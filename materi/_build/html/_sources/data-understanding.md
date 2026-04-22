# Data Understanding

Pada bagian Data Understanding, ini merupakan langkah awal dari CRISP-DM setelah tahap pemahaman bisnis selesai. Di sini, tujuan utamanya ialah mendalami data yang akan diolah sebelum memasuki tahap persiapan/cleaning ataupun modeling.

Tahap Data Understanding bertujuan untuk memperoleh wawasan lengkap tentang data yang akan dipakai, sehingga tahapan selanjutnya seperti Data Preparation bisa berjalan dengan lebih efektif dan proses analisis bisa diarahakan dengan lebih baik serta mengurangi kemungkinan eror dalam pembuatan model.

## 1. Sumber Data

Untuk tugas kali ini saya mengambil dataset Iris Flower yang bisa didapat dari Kaggle. Dataset tersebut memuat hasil pengukuran morfologi dari tiga spesies bunga iris (Iris-setosa, Iris-versicolor, dan Iris-virginica). Terdapat 150 entri dengan lima kolom/fitur utama, di mana 4 di antaranya bertipe numerik: sepal length, sepal width, petal length, serta petal width, sedangkan fitur kelima “species” berupa label kategorikal yang digunakan untuk klasifikasi.
[Cek Iris Dataset di Kaggle](https://www.kaggle.com/datasets/arshid/iris-flower-dataset?resource=download)

## 2. Melihat dan Mengeksplorasi Dataset

Eksplorasi dilakukan salah satunya dengan memanfaatkan tools Python.

### 2.1 Koneksi Database ke Python

#### Menghubungkan MySQL ke Python

```
pip install mysql-connector-python
```

Instal library di atas terlebih dahulu. Paket tersebut diperlukan agar Python dapat terkoneksi ke database MySQL.

```
import mysql.connector
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="iris"
)
print("Berhasil terhubung ke MySQL!")

cursor = db.cursor()
cursor.execute("SELECT * FROM iris")
hasil = cursor.fetchall()

for data in hasil:
    print(data)

db.close()
```

Kode tersebut bisa langsung dipakai untuk mengambil data dari database MySQL yang sudah berisi tabel bernama "iris" hasil import dari csv.

#### Koneksi PostgreSQL

```
pip install psycopg2
```

Untuk koneksi ke PostgreSQL, cukup install library psycopg2. Setelah itu, contoh penggunaan:

```
import psycopg2
conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="",
    port="5432"
)
print("Berhasil connect ke PostgreSQL")
cursor = conn.cursor()

cursor.execute("SELECT * FROM iris")
rows = cursor.fetchall()
for row in rows:
    print(row)

cursor.close()
conn.close()
```

Pastikan nama database dan tabel di PostgreSQL-nya sudah sesuai.

### 2.2 Persiapan Tools (Pandas, Matplotlib) tanpa Database

```
from google.colab import drive
import pandas as pd
import matplotlib.pyplot as plt

drive.mount('/content/drive')
```

Penjelasan:
- Google Drive di-mount supaya file csv bisa diakses dari Colab.
- Pandas untuk analisa data.
- Matplotlib dipakai untuk visualisasi grafik.

### 2.3 Menentukan Lokasi File CSV

```
path = "/content/drive/MyDrive/tugas/IRIS.csv"
df = pd.read_csv(path)
```

Misalnya IRIS.csv ada di folder tugas pada Google Drive, gunakan path tersebut supaya bisa diload oleh pandas.

### 2.4 Melihat Struktur Data

```
df.head()
```

Perintah di atas menampilkan 5 baris pertama data yang termuat dalam IRIS.csv. Berikut contohnya:

| index | sepal_length | sepal_width | petal_length | petal_width | species        |
|-------|--------------|-------------|--------------|-------------|---------------|
| 0     | 5.1          | 3.5         | 1.4          | 0.2         | Iris-setosa   |
| 1     | 4.9          | 3.0         | 1.4          | 0.2         | Iris-setosa   |
| 2     | 4.7          | 3.2         | 1.3          | 0.2         | Iris-setosa   |
| 3     | 4.6          | 3.1         | 1.5          | 0.2         | Iris-setosa   |
| 4     | 5.0          | 3.6         | 1.4          | 0.2         | Iris-setosa   |

Penjelasan kolom:
1. Sepal Length: panjang kelopak (cm)
2. Sepal Width: lebar kelopak (cm)
3. Petal Length: panjang mahkota (cm)
4. Petal Width: lebar mahkota (cm)
5. Species: kategori atau label (“Setosa”, “Versicolor”, “Virginica”)

### 2.5 Statistik Ringkas (Descriptive Statistics)

```
df.describe()
```

Hasil statistik deskriptif dapat berupa:

|         | sepal_length | sepal_width | petal_length | petal_width |
|---------|--------------|-------------|--------------|-------------|
| count   | 150.0        | 150.0       | 150.0        | 150.0       |
| mean    | 5.84         | 3.05        | 3.76         | 1.20        |
| std     | 0.83         | 0.43        | 1.76         | 0.76        |
| min     | 4.3          | 2.0         | 1.0          | 0.1         |
| 25%     | 5.1          | 2.8         | 1.6          | 0.3         |
| 50%     | 5.8          | 3.0         | 4.35         | 1.3         |
| 75%     | 6.4          | 3.3         | 5.1          | 1.8         |
| max     | 7.9          | 4.4         | 6.9          | 2.5         |

Rata-rata sepal dan petal memperlihatkan nilai yang wajar, serta jumlah data tiap atribut konsisten (150 data). Variasi terbesar pada petal length, sepal width paling kecil variasinya.

### 2.6 Deteksi Data Ganda (Duplikat)

```
df.duplicated().sum()
```

Jika hasilnya 3, berarti dari seluruh data, ada 3 yang terduplikasi.

### 2.7 Pengecekan Data Kosong (Null)

```
df.isnull().sum()
```

Hasil tabel:

| kolom         | null_count |
|---------------|------------|
| sepal_length  | 0          |
| sepal_width   | 0          |
| petal_length  | 0          |
| petal_width   | 0          |
| species       | 0          |

Berarti semua kolom terisi penuh, tidak ada missing value.

## 3. Korelasi Antar Fitur

Korelasi digunakan untuk melihat hubungan antara fitur-fitur numerik dalam dataset. Angkanya berkisar antara -1 sampai +1, dengan penjelasan:
- -1 = negatif sempurna
- 0 = tidak berkorelasi
- +1 = positif sempurna

Analisis dilakukan pada sepal_length, sepal_width, petal_length, petal_width.

```
correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix)
```

Contoh matriks korelasi:

|              | sepal_length | sepal_width | petal_length | petal_width |
|--------------|--------------|-------------|--------------|-------------|
| sepal_length | 1.000000     | -0.109369   | 0.871754     | 0.817954    |
| sepal_width  | -0.109369    | 1.000000    | -0.420516    | -0.356544   |
| petal_length | 0.871754     | -0.420516   | 1.000000     | 0.962757    |
| petal_width  | 0.817954     | -0.356544   | 0.962757     | 1.000000    |

**Catatan hasil korelasi:**
- Korelasi paling kuat pada petal_length dan petal_width (positif tinggi)
- Korelasi kuat juga ada pada sepal_length dengan petal_length/petal_width
- Sepal_width cenderung tidak terlalu berhubungan erat dengan fitur lain (bahkan negatif)

## 4. Cek Konsistensi Data

Rekap dari hasil eksplorasi sebelumnya, dataset berisi 150 data tiap atribut numerik. Dari pengecekan null (df.isnull().sum()), semua kolom tidak ada nilai kosong. Namun, terdapat 3 baris duplikat pada hasil df.duplicated().sum(), yang mesti diperhatikan agar tidak mengganggu hasil analisis.

## 5. Visualisasi

### Visualisasi dengan Python

#### Distribusi Label (Jumlah Tiap Species)

![bar species](https://cdn.mathpix.com/snip/images/Gt4lANFXJ1tvApce6vXv0vo4hu08VXJb7_bTkbfgn-M.original.fullsize.png)

Grafik bar menggambarkan jumlah masing-masing species dalam dataset (masing-masing 50 data), artinya kelas seimbang dan baik untuk analisa selanjutnya.

#### Distribusi Fitur-Fitur Numerik

![numeric histogram](https://cdn.mathpix.com/snip/images/6u595jYczu5oHxd1yDMvEGkfvzQYsyvFrhOICQnLnGY.original.fullsize.png)

Pada histogram, masing-masing fitur numerik menampilkan distribusi berbeda, tanpa anomali ekstrim. Fitur petal_length dan petal_width cenderung sangat informatif untuk klasifikasi.

#### Scatter Plot Sepal maupun Petal

![scatter sepal](https://cdn.mathpix.com/snip/images/Y0D4Vc_bKZyv6KgPthBSm8pOPKFWZf-ZcZEN1qaBDpM.original.fullsize.png)

Scatter plot ini membandingkan sepal_length dan sepal_width; hasilnya menunjukkan sebaran data yang tidak benar-benar linear, indikasi korelasi lemah.

![scatter petal](https://cdn.mathpix.com/snip/images/cbXgTI0n7n5F-Hm_QEabnsoY06-1zrCTGJM6BGw0VT4.original.fullsize.png)

Sedangkan scatter plot petal_length dengan petal_width membentuk hubungan hampir linear sekaligus mengelompokkan kelas-kelas species berbeda dengan cukup jelas.

#### Boxplot untuk Deteksi Outlier

![boxplot](https://cdn.mathpix.com/snip/images/YfGQj5-WjGnpPQF6mWpdtoRSXTPrJIw5sXZ6l769Xo0.original.fullsize.png)

Boxplot memperlihatkan persebaran setiap fitur. Terlihat ada beberapa outlier pada sepal_width, sedangkan atribut lain distribusinya relatif normal tanpa outlier signifikan. Petal_length dan petal_width juga memiliki variasi yang tinggi.

### Visualisasi via Orange

- Statistik keseluruhan:  
  ![Orange stats](https://cdn.mathpix.com/snip/images/HDC--owJyN2brHw80LVv73mBWygr0yxk9lX4t96wFvc.original.fullsize.png)
- Bar Chart:
  ![Orange bar 1](https://cdn.mathpix.com/snip/images/0fNG62bZBym-M-49gaNRJ-AExLKqTm4Y6JaYxtxx_L0.original.fullsize.png)
  ![Orange bar 2](https://cdn.mathpix.com/snip/images/5GkmBEAXfm44a2sSxu5ec27ANj-g9KUY8pKwPG6wNAU.original.fullsize.png)
- Scatter Plot:
  ![Orange scatter 1](https://cdn.mathpix.com/snip/images/wdF8Z79VGVflh2mDtT56v_4f5q5SlGbLHtM778Ut6ac.original.fullsize.png)
  ![Orange scatter 2](https://cdn.mathpix.com/snip/images/1K4uhfC_chCDhN56N1oE_UogCOe7cSdr_zdgA9iD8zI.original.fullsize.png)
- Box Plot:
  ![Orange boxplot](https://cdn.mathpix.com/snip/images/0IzZzwtAwUWlcBUmvnfO6rYeciTHx-oNosmnHghUypU.original.fullsize.png)
- Korelasi:
  ![Orange korelasi](https://cdn.mathpix.com/snip/images/5LdarTZMEcRto-WhHPhbKFnrNfQpFu_pgj1JCbCikyQ.original.fullsize.png)

## 6. Perhitungan Jarak Data

### Definisi Similarity & Dissimilarity

#### Similarity

Similarity mengukur tingkat kemiripan dua entitas. Nilai berada antara 0 (tidak mirip) hingga 1 (sangat mirip). Semakin tinggi nilainya, semakin besar tingkat kesamaan.

#### Dissimilarity

Sebaliknya dengan similarity, dissimilarity mengukur tingkat perbedaan dua objek, mulai dari 0 ke atas. Makin besar nilainya, makin tidak mirip.

##### Minkowski Distance (Umum untuk Data Numerik)

$$
d(i, j) = \sqrt[h]{|x_{i1}-x_{j1}|^{h} + |x_{i2}-x_{j2}|^{h} + \cdots + |x_{ip}-x_{jp}|^{h}}
$$

Untuk h=1 → Manhattan, h=2 → Euclidean.

##### Manhattan Distance

$$
d(i,j) = |x_{i1} - x_{j1}| + |x_{i2} - x_{j2}| + \ldots + |x_{ip} - x_{jp}|
$$

Kuat untuk outlier dan data berdimensi tinggi.

##### Euclidean Distance

$$
d(i,j) = \sqrt{(x_{i1} - x_{j1})^2 + (x_{i2} - x_{j2})^2 + \dots + (x_{ip} - x_{jp})^2}
$$

Sering digunakan untuk data numerik, namun perlu normalisasi agar fitur dengan range besar tidak mendominasi.

### Menghitung Dissimilarity pada Dataset

#### a. Dataset Iris

Karena semua fiturnya bertipe numerik, untuk Iris cukup menggunakan Euclidean distance.

##### Contoh Manual (baris 2 vs 1 dan 3 vs 1):

$$
d(2,1) = \sqrt{(4{,}9 - 5{,}1)^2 + (3 - 3{,}5)^2}
= \sqrt{0{,}04 + 0{,}25}
= 0{,}538
$$

$$
d(3,1) = \sqrt{(4{,}7 - 5{,}1)^2 + (3{,}2 - 3{,}5)^2 + (1{,}3 - 1{,}4)^2}
= \sqrt{0{,}16 + 0{,}09 + 0{,}01}
= 0{,}5099
$$

##### Perhitungan dengan Python

```
from scipy.spatial.distance import pdist, squareform

X = df.select_dtypes(include=['float64', 'int64'])
dist = pdist(X, metric='euclidean')
distance_matrix = squareform(dist)

distance_df = pd.DataFrame(distance_matrix)
print(distance_df.iloc[:5, :5])
```

Contoh output (5x5 pertama):

|   | 0        | 1        | 2        | 3        | 4        |
|---|----------|----------|----------|----------|----------|
| 0 | 0.000000 | 0.538516 | 0.509902 | 0.648074 | 0.141421 |
| 1 | 0.538516 | 0.000000 | 0.300000 | 0.331662 | 0.608276 |
| 2 | 0.509902 | 0.300000 | 0.000000 | 0.244949 | 0.509902 |
| 3 | 0.648074 | 0.331662 | 0.244949 | 0.000000 | 0.648074 |
| 4 | 0.141421 | 0.608276 | 0.509902 | 0.648074 | 0.000000 |

##### Hasil dengan Orange

![orange distance matrix](https://cdn.mathpix.com/snip/images/9PGgc7ny0uksXf9OeJWUUqJ1W74_wYqqPmGLGPGhBL8.original.fullsize.png)

Keluaran di atas membuktikan hasil konsisten baik manual, Python, maupun Orange.

#### b. Dataset dengan Fitur Campuran

Dicontohkan menggunakan dataset Student Alcohol Consumption (ada fitur nominal, numerik, ordinal). Jarak dihitung memakai Gower Distance.

##### Perhitungan Manual

Misal membandingkan baris 1 dan 2:
1. Fitur Nominal: nilai yang berbeda = 1, sama = 0. Jumlahkan hasil.
2. Fitur Numerik:  
   $d_{ij}^{(f)} = \frac{|x_{if} - x_{jf}|}{max(x_f) - min(x_f)}$
3. Fitur Ordinal:  
   Normalisasi ranking ke [0,1] lalu selisih nilai.

Gabungkan seluruh hasil lalu rata-rata dengan jumlah fitur.

##### Perhitungan Python

```
import numpy as np

data = df.copy()
n = data.shape[0]

numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(exclude=[np.number]).columns

for col in numeric_cols:
   min_val = data[col].min()
   max_val = data[col].max()
   
   if max_val != min_val:
       data[col] = (data[col] - min_val) / (max_val - min_val)
   else:
       data[col] = 0

dist_matrix = np.zeros((n, n))

for i in range(n):
   for j in range(n):
       total_dist = 0
       valid_features = 0
       
       for col in data.columns:
           xi = data.iloc[i][col]
           xj = data.iloc[j][col]
           
           if pd.isna(xi) or pd.isna(xj):
               continue
           
           if col in numeric_cols:
               d = abs(xi - xj)
           else:
               d = 0 if xi == xj else 1
           
           total_dist += d
           valid_features += 1
       
       dist_matrix[i, j] = total_dist / valid_features

distance_df = pd.DataFrame(dist_matrix)
print(distance_df.iloc[:5, :5])
```

Contoh output 5x5:

|   | 0        | 1        | 2        | 3        | 4        |
|---|----------|----------|----------|----------|----------|
| 0 | 0.000000 | 0.520408 | 0.418367 | 0.561224 | 0.397959 |
| 1 | 0.520408 | 0.000000 | 0.183673 | 0.469388 | 0.163265 |
| 2 | 0.418367 | 0.183673 | 0.000000 | 0.571429 | 0.306122 |
| 3 | 0.561224 | 0.469388 | 0.571429 | 0.000000 | 0.377551 |
| 4 | 0.397959 | 0.163265 | 0.306122 | 0.377551 | 0.000000 |




# 1. Dataset yang Digunakan

Pada analisis ini digunakan **Boston Housing Dataset** yang diperoleh dari Kaggle. Dataset ini memuat berbagai variabel yang mencerminkan kondisi lingkungan, faktor sosial ekonomi, serta karakteristik rumah di wilayah Boston.

Dataset ini sering digunakan dalam machine learning, khususnya untuk kasus prediksi harga rumah berdasarkan beberapa atribut seperti tingkat kriminalitas (*CRIM*), jumlah kamar (*RM*), dan persentase status sosial rendah (*LSTAT*).

---

# 2. Perhitungan WKNN Secara Manual

## 2.1 Identifikasi Missing Value
Ditemukan nilai kosong (*missing value*) pada atribut **LSTAT**. Nilai tersebut akan diestimasi menggunakan metode **Weighted K-Nearest Neighbor (WKNN)**.

---

## 2.2 Normalisasi Data
Data dinormalisasi menggunakan metode **Min-Max Scaling** agar semua variabel berada pada skala yang sama.

Rumus:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

---

## 2.3 Perhitungan Similarity

Nilai kemiripan antar data dihitung dengan rumus:

$$
S_i = \frac{1}{\sum (Y_{ih} - Y_{jh})^2}
$$

Hasil perhitungan:

| No | Si |
|----|----|
| 1 | 0.096432567 |
| 2 | 0.186117946 |
| 3 | 0.251889237 |
| 4 | 1.262310597 |

---

## 2.4 Menghitung Total Similarity

$$
\sum S_i = 1.796750348
$$

---

## 2.5 Menghitung Nilai Tertimbang

| Si | LSTAT (Yjh) | Si × Yjh |
|----|------------|----------|
| 0.096432567 | 0.329032258 | 0.031730 |
| 0.186117946 | 1 | 0.186118 |
| 0.251889237 | 0.175806452 | 0.044277 |
| 1.262310597 | 0 | 0 |

$$
\sum (S_i \times Y_{jh}) = 0.262125
$$

---

## 2.6 Hasil Imputasi

Nilai imputasi dihitung dengan:

$$
\hat{y}_{ih} =
\frac{\sum (S_i \times Y_{jh})}{\sum S_i}
$$

Hasil:

$$
\hat{y}_{ih} = 0.14589179
$$

Nilai ini digunakan untuk menggantikan missing value pada atribut **LSTAT**.

---

# 3. Implementasi Menggunakan Python

## 3.1 Import Library
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

3.2 Membaca Dataset
df = pd.read_excel("data.xlsx")
df = df.replace("?", np.nan)
df = df.apply(pd.to_numeric)

print(df)

3.3 Menentukan Kolom Normalisasi
cols_norm = [
'CRIM','INDUS','NOX','RM','AGE',
'DIS','TAX','PTRATIO','B','LSTAT','MEDV'
]

3.4 Menghitung Nilai Minimum dan Maksimum
min_val = df[cols_norm].min()
max_val = df[cols_norm].max()

print(min_val)
print(max_val)
3.5 Normalisasi Data
df_norm = df.copy()

for col in cols_norm:
    df_norm[col] = (df[col] - min_val[col]) / (max_val[col] - min_val[col])

print(df_norm)
3.6 Menentukan Data Missing
missing_index = df_norm[df_norm['LSTAT'].isna()].index
row_missing = df_norm.loc[missing_index].drop(columns=['LSTAT']).iloc[0]

complete_data = df_norm.dropna()
3.7 Menghitung Similarity
Si = []
Yjh = []

for i in range(len(complete_data)):
    row = complete_data.iloc[i].drop('LSTAT')
    diff = (row_missing - row) ** 2
    similarity = 1 / diff.sum()
    
    Si.append(similarity)
    Yjh.append(complete_data.iloc[i]['LSTAT'])

Si = np.array(Si)
Yjh = np.array(Yjh)

print(Si)

Output:

[0.09643257 0.18611795 0.25188924 1.2623106]
3.8 Menghitung Penyebut
penyebut = np.sum(Si)
print(penyebut)

Output:

1.796750347642724
3.9 Menghitung Pembilang
pembilang = np.sum(Si * Yjh)
print(pembilang)

Output:

0.262125
3.10 Hasil Imputasi
hasil = pembilang / penyebut
print(hasil)

Output:

0.14589179
3.11 Kesimpulan

Hasil perhitungan menggunakan Python menghasilkan nilai yang sama dengan perhitungan manual, yaitu 0.14589179. Hal ini menunjukkan bahwa proses imputasi menggunakan metode WKNN telah dilakukan dengan benar dan konsisten.


## 2. Normalisasi Data

Normalisasi merupakan tahap preprocessing yang bertujuan untuk mengubah skala data ke dalam rentang tertentu sehingga setiap fitur memiliki kontribusi yang seimbang dalam proses analisis.

Langkah ini penting terutama pada algoritma berbasis jarak, karena perbedaan skala dapat menyebabkan fitur dengan nilai besar menjadi lebih dominan dibandingkan fitur lainnya.

---

### 2.1 Jenis-Jenis Normalisasi

#### 1. Min-Max Scaling

Metode ini mentransformasikan nilai data ke dalam rentang tertentu, biasanya antara 0 hingga 1.

Rumus:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

**Contoh Perhitungan:**

Data pada atribut **AGE**:

| AGE |
|-----|
| 65.2 |
| 78.9 |
| 45.8 |

Nilai minimum dan maksimum:

- \(x_{\min} = 45.8\)  
- \(x_{\max} = 78.9\)

Perhitungan:

$$
x' = \frac{65.2 - 45.8}{78.9 - 45.8}
$$

$$
x' = \frac{19.4}{33.1} = 0.586
$$

Hasil normalisasi:

| AGE Awal | AGE Normalisasi |
|----------|----------------|
| 65.2 | 0.586 |
| 78.9 | 1 |
| 45.8 | 0 |

---

#### 2. Z-Score Standardization

Metode ini menggunakan nilai rata-rata (*mean*) dan standar deviasi untuk melakukan normalisasi.

Rumus:

$$
x' = \frac{x - \mu}{\sigma}
$$

**Contoh Perhitungan:**

Data **AGE**:

| AGE |
|-----|
| 65.2 |
| 78.9 |
| 45.8 |

**Menghitung Mean:**

$$
\mu = \frac{65.2 + 78.9 + 45.8}{3} = 63.3
$$

**Menghitung Varians:**

$$
(65.2 - 63.3)^2 = 3.61
$$

$$
(78.9 - 63.3)^2 = 243.36
$$

$$
(45.8 - 63.3)^2 = 306.25
$$

$$
\text{Total} = 553.22
$$

$$
\frac{553.22}{3} = 184.41
$$

**Standar Deviasi:**

$$
\sigma = \sqrt{184.41} = 13.58
$$

**Perhitungan Z-Score:**

$$
x' = \frac{65.2 - 63.3}{13.58} = 0.14
$$

Hasil:

| AGE Awal | Z-Score |
|----------|--------|
| 65.2 | 0.14 |
| 78.9 | 1.15 |
| 45.8 | -1.29 |

---

#### 3. Decimal Scaling

Metode ini dilakukan dengan membagi data menggunakan pangkat 10 berdasarkan jumlah digit terbesar.

Rumus:

$$
x' = \frac{x}{10^j}
$$

**Contoh:**

Data **AGE**:

| AGE |
|-----|
| 65.2 |
| 78.9 |
| 45.8 |

Nilai terbesar adalah **78.9**, sehingga jumlah digit sebelum desimal adalah 2 → \(j = 2\)

Perhitungan:

$$
x' = \frac{65.2}{10^2} = 0.652
$$

Hasil:

| AGE Awal | Decimal Scaling |
|----------|----------------|
| 65.2 | 0.652 |
| 78.9 | 0.789 |
| 45.8 | 0.458 |

---

### 2.2 Implementasi Normalisasi dengan Python (Scikit-Learn)

Selain dilakukan secara manual, normalisasi juga dapat diimplementasikan menggunakan library **scikit-learn** untuk meningkatkan efisiensi.

Data yang digunakan:

| AGE |
|-----|
| 65.2 |
| 78.9 |
| 45.8 |

---

#### 1. Min-Max Scaling

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {'AGE':[65.2,78.9,45.8]}
df = pd.DataFrame(data)

scaler = MinMaxScaler()
df['AGE_MinMax'] = scaler.fit_transform(df[['AGE']])

print(df)

Output:

AGE	AGE_MinMax
65.2	0.586
78.9	1
45.8	0
2. Z-Score Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['AGE_ZScore'] = scaler.fit_transform(df[['AGE']])

print(df)

Output:

AGE	AGE_ZScore
65.2	0.14
78.9	1.15
45.8	-1.29
3. Decimal Scaling (Manual dengan Python)
import numpy as np

data = {'AGE':[65.2,78.9,45.8]}
df = pd.DataFrame(data)

max_value = df['AGE'].max()
j = len(str(int(max_value)))

df['AGE_Decimal'] = df['AGE'] / (10**j)

print(df)

Output:

AGE	AGE_Decimal
65.2	0.652
78.9	0.789
45.8	0.458
Kesimpulan

Berdasarkan hasil perhitungan, nilai normalisasi yang diperoleh melalui Python sesuai dengan hasil perhitungan manual. Hal ini menunjukkan bahwa metode normalisasi telah diterapkan dengan benar, serta penggunaan library seperti scikit-learn dapat membantu mempercepat proses preprocessing data.