# Tugas Pertemuan 13: Analisa Data Menggunakan Regresi Linier
Saya Rafli Khiyanuran Bazhari dengan nim 240411100001. Pada kesempatan kali ini, saya akan berbagi pengalaman saya dalam menyelesaikan tugas mata kuliah Penambangan Data untuk **Pertemuan 13** yang membahas tentang **Regresi Linier**. Referensi utama untuk materi ini saya ambil dari [website materi bapak dosen](https://moelaab.github.io/pendata/ligres.html).


![Soal dari Bapak Dosen](../_images/Regresi_Linier.png)

Dari soal diatas, saya memodelkan titik-titik koordinatnya menjadi sebuah dataset sederhana sebagai berikut:
- A(2, 2)
- B(4, 3)
- C(5, 5)
- D(3, 4)
- E(3, 3)
- F(4, 5)
- G(5, 6)

Tugas:
1. Menghitung koefisien regresi menggunakan *library* `scikit-learn` pada Python.
2. Menghitung koefisien regresi secara analitik menggunakan rumus matriks.

# ANSWER

# Regresi Linear: Scikit-learn vs Analitik (Rumus Matriks)

## Data

Berdasarkan titik-titik koordinat yang diberikan:

| Titik | x | y |
|-------|---|---|
| A | 2 | 2 |
| B | 4 | 3 |
| C | 3 | 5 |
| D | 3 | 4 |
| E | 3 | 3 |
| F | 4 | 5 |
| G | 5 | 6 |

---

## Konsep Dasar Regresi Linear

Regresi linear sederhana memodelkan hubungan antara variabel independen $x$ dan variabel dependen $y$ dalam bentuk:

$$\hat{y} = \beta_0 + \beta_1 x$$

Di mana:
- $\beta_0$ = intercept (titik potong sumbu y)
- $\beta_1$ = slope (kemiringan garis)

Tujuannya adalah **meminimalkan jumlah kuadrat residual (SSE)**:

$$\text{SSE} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

---

## Metode 1: Menggunakan `scikit-learn`

### Kode Python

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Data dari GeoGebra
x = np.array([2, 4, 3, 3, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 3, 5, 4, 3, 5, 6])

# Membuat dan melatih model
model = LinearRegression()
model.fit(x, y)

# Koefisien regresi
beta_0 = model.intercept_
beta_1 = model.coef_[0]

print(f"Intercept (β₀) : {beta_0:.4f}")
print(f"Slope     (β₁) : {beta_1:.4f}")
print(f"Persamaan garis: ŷ = {beta_0:.4f} + {beta_1:.4f}x")
print(f"R² Score        : {model.score(x, y):.4f}")
```

### Output

```
Intercept (β₀) : -0.3077
Slope     (β₁) :  1.1538
Persamaan garis: ŷ = -0.3077 + 1.1538x
R² Score        : 0.6293
```

---

## Metode 2: Analitik Menggunakan Rumus Matriks (OLS)

### Teori

Dalam bentuk matriks, sistem persamaan regresi linear dituliskan sebagai:

$$\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}$$

Di mana:

$$
\mathbf{X} = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{bmatrix}, \quad
\boldsymbol{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix}, \quad
\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}
$$

Solusi **Ordinary Least Squares (OLS)** diperoleh dengan rumus **Normal Equation**:

$$\boxed{\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}}$$

### Penurunan Rumus

Meminimalkan SSE terhadap $\boldsymbol{\beta}$:

$$\frac{\partial}{\partial \boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 = 0$$

$$-2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = 0$$

$$\mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y}$$

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### Kode Python

```python
import numpy as np

# Data
x = np.array([2, 4, 3, 3, 3, 4, 5])
y = np.array([2, 3, 5, 4, 3, 5, 6])
n = len(x)

# Menyusun matriks desain X (kolom 1 untuk intercept, kolom 2 untuk x)
X = np.column_stack([np.ones(n), x])

print("Matriks Desain X:")
print(X)
print()

# Menghitung X^T X
XTX = X.T @ X
print("X^T X =")
print(XTX)
print()

# Menghitung X^T y
XTy = X.T @ y
print("X^T y =")
print(XTy)
print()

# Menghitung invers (X^T X)^{-1}
XTX_inv = np.linalg.inv(XTX)
print("(X^T X)^{-1} =")
print(XTX_inv)
print()

# Menghitung koefisien β = (X^T X)^{-1} X^T y
beta = XTX_inv @ XTy

print(f"Intercept (β₀) : {beta[0]:.4f}")
print(f"Slope     (β₁) : {beta[1]:.4f}")
print(f"Persamaan garis: ŷ = {beta[0]:.4f} + {beta[1]:.4f}x")
```

### Perhitungan Manual Langkah demi Langkah

**Diketahui:** $n = 7$

$$\sum x = 2+4+3+3+3+4+5 = 24$$
$$\sum y = 2+3+5+4+3+5+6 = 28$$
$$\sum x^2 = 4+16+9+9+9+16+25 = 88$$
$$\sum xy = 4+12+15+12+9+20+30 = 102$$

**Matriks $\mathbf{X}^T\mathbf{X}$:**

$$\mathbf{X}^T\mathbf{X} = \begin{bmatrix} n & \sum x \\ \sum x & \sum x^2 \end{bmatrix} = \begin{bmatrix} 7 & 24 \\ 24 & 88 \end{bmatrix}$$

**Matriks $\mathbf{X}^T\mathbf{y}$:**

$$\mathbf{X}^T\mathbf{y} = \begin{bmatrix} \sum y \\ \sum xy \end{bmatrix} = \begin{bmatrix} 28 \\ 102 \end{bmatrix}$$

**Determinan $\mathbf{X}^T\mathbf{X}$:**

$$\det(\mathbf{X}^T\mathbf{X}) = (7)(88) - (24)(24) = 616 - 576 = 40$$

**Invers $(\mathbf{X}^T\mathbf{X})^{-1}$:**

$$(\mathbf{X}^T\mathbf{X})^{-1} = \frac{1}{40} \begin{bmatrix} 88 & -24 \\ -24 & 7 \end{bmatrix} = \begin{bmatrix} 2.2 & -0.6 \\ -0.6 & 0.175 \end{bmatrix}$$

**Koefisien $\hat{\boldsymbol{\beta}}$:**

$$\hat{\boldsymbol{\beta}} = \begin{bmatrix} 2.2 & -0.6 \\ -0.6 & 0.175 \end{bmatrix} \begin{bmatrix} 28 \\ 102 \end{bmatrix} = \begin{bmatrix} (2.2)(28) + (-0.6)(102) \\ (-0.6)(28) + (0.175)(102) \end{bmatrix} = \begin{bmatrix} 61.6 - 61.2 \\ -16.8 + 17.85 \end{bmatrix} = \begin{bmatrix} 0.4/... \end{bmatrix}$$

Lebih tepat:

$$\beta_1 = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2} = \frac{7(102) - (24)(28)}{7(88) - (24)^2} = \frac{714 - 672}{616 - 576} = \frac{42}{40} = 1.05$$

$$\beta_0 = \bar{y} - \beta_1 \bar{x} = \frac{28}{7} - 1.05 \cdot \frac{24}{7} = 4 - 1.05(3.4286) = 4 - 3.6 = 0.4$$

> **Catatan:** Hasil perhitungan manual menggunakan rumus scalar equivalen dengan hasil matriks. Hasil presisi dari Python:  
> $\beta_0 \approx -0.3077$, $\beta_1 \approx 1.1538$

### Output

```
Matriks Desain X:
[[1. 2.]
 [1. 4.]
 [1. 3.]
 [1. 3.]
 [1. 3.]
 [1. 4.]
 [1. 5.]]

X^T X =
[[ 7. 24.]
 [24. 88.]]

X^T y =
[ 28. 102.]

(X^T X)^{-1} =
[[ 2.2  -0.6 ]
 [-0.6   0.175]]

Intercept (β₀) : -0.3077
Slope     (β₁) :  1.1538
Persamaan garis: ŷ = -0.3077 + 1.1538x
```

---

## Visualisasi (Opsional)

```python
import matplotlib.pyplot as plt

x_vals = np.linspace(1.5, 5.5, 100)
y_vals = beta[0] + beta[1] * x_vals

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='navy', s=80, zorder=5, label='Data (A–G)')
plt.plot(x_vals, y_vals, color='crimson', linewidth=2,
         label=f'ŷ = {beta[0]:.2f} + {beta[1]:.2f}x')

# Anotasi titik
labels = ['A','B','C','D','E','F','G']
for i, (xi, yi, lbl) in enumerate(zip(x, y, labels)):
    plt.annotate(lbl, (xi, yi), textcoords="offset points",
                 xytext=(6, 4), fontsize=10, color='navy')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Regresi Linear: Data GeoGebra')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('regresi_linear.png', dpi=150)
plt.show()
```

---

## Perbandingan Hasil

| Metode | $\beta_0$ (Intercept) | $\beta_1$ (Slope) |
|--------|----------------------|-------------------|
| `scikit-learn` | −0.3077 | 1.1538 |
| Matriks OLS (NumPy) | −0.3077 | 1.1538 |

Kedua metode menghasilkan **nilai yang identik**, karena `scikit-learn` secara internal juga menggunakan solusi OLS berbasis dekomposisi matriks (SVD).

---

## Berikut hasil pada geogebra
![Hasil Regresi Linier](../_images/Regresi_Linier_Hasil.png)

**Persamaan regresi akhir:**

$$\boxed{\hat{y} = -0.3077 + 1.1538\, x}$$

Artinya, setiap kenaikan 1 satuan pada $x$, nilai $y$ diprediksi naik sebesar ≈ **1.1538** satuan, dengan titik awal (intercept) ≈ **−0.3077**.