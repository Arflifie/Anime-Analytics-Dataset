# Analisis Popularitas Studio Anime dengan PySpark & Uji Kruskal-Wallis

## Deskripsi
Skrip ini menggabungkan ekosistem **Big Data (PySpark)** dan **statistik non-parametrik (SciPy)** untuk menguji apakah studio produksi memberikan pengaruh nyata terhadap tingkat popularitas anime.

---

## Alur Kerja

```
Data CSV → Spark (Ingestion) → Filter per Studio → Kruskal-Wallis → Kesimpulan
```

---

## Library yang Digunakan

| Library | Fungsi |
|---|---|
| `os`, `sys` | Mengunci jalur Python agar PySpark berjalan di Windows |
| `pyspark.sql.SparkSession` | Pintu masuk utama ke ekosistem Spark |
| `scipy.stats.kruskal` | Fungsi uji statistik Kruskal-Wallis |

> **Catatan Windows:** Variabel `PYSPARK_PYTHON` dan `PYSPARK_DRIVER_PYTHON` diset ke `sys.executable` untuk mencegah error *"Python worker failed to connect back"* akibat App Execution Aliases.

---

## Tahapan Proses

### Tahap 1 — Inisialisasi Spark
```python
spark = SparkSession.builder \
    .appName("AnalisisPopularitasStudio") \
    .master("local[*]") \
    .getOrCreate()
```
- `.master("local[*]")` → menyimulasikan klaster Big Data di lokal, menggunakan seluruh core CPU secara paralel.

---

### Tahap 2 — Data Ingestion
```python
df = spark.read.csv("data/processed/anime_dataset_pre.csv", header=True, inferSchema=True)
```
- `header=True` → baris pertama dibaca sebagai nama kolom.
- `inferSchema=True` → Spark mendeteksi tipe data secara otomatis.
- Menggunakan **Lazy Evaluation**: data dibagi ke partisi, tidak langsung membebani RAM.

---

### Tahap 3 — Ekstraksi Studio Unik
```python
daftar_studio = [row['studios'] for row in df.select('studios').distinct().collect()
                 if row['studios'] is not None]
```
- `distinct()` → menghapus duplikat.
- `collect()` → **Action** yang memicu Spark bekerja dan mengembalikan hasil ke Python.

---

### Tahap 4 — Distributed Data Grouping (Simulasi Map-Reduce)
```python
skor_popularitas_studio = df.filter(df['studios'] == studio) \
                            .select('popularity') \
                            .rdd.flatMap(lambda x: x).collect()
```

| Operasi | Peran | Keterangan |
|---|---|---|
| `filter()` | **MAP** | Worker node mencari baris sesuai studio |
| `select()` | Proyeksi | Hanya mengambil kolom `popularity` |
| `rdd.flatMap(lambda x: x)` | Transformasi | Meleburkan tabel Spark → array 1 dimensi (agar bisa dibaca SciPy) |
| `collect()` | **REDUCE** | Menarik data dari worker ke driver Python |

---

### Tahap 5 — Penghentian Spark
```python
spark.stop()
```
Setelah data diringkas menjadi list Python murni, Spark dihentikan untuk **membebaskan RAM** sebelum komputasi statistik.

---

### Tahap 6 — Uji Statistik Kruskal-Wallis
```python
stat, p_value = kruskal(*kelompok_popularitas)
```
- `*kelompok_popularitas` → **Unpacking Operator**, membongkar list menjadi argumen-argumen terpisah.
- Kruskal-Wallis dipilih karena merupakan alternatif **non-parametrik** dari ANOVA, cocok untuk data popularitas yang distribusinya tidak normal.

---

### Tahap 7 — Interpretasi Hasil
```
H-Statistic : nilai uji statistik
P-Value     : probabilitas hasil terjadi secara kebetulan
```

| Kondisi | Kesimpulan |
|---|---|
| `P-Value < 0.05` | **SIGNIFIKAN** — Studio berpengaruh nyata terhadap popularitas anime |
| `P-Value ≥ 0.05` | **TIDAK SIGNIFIKAN** — Tidak ada perbedaan nyata antar-studio |

---

## Hipotesis

- **H₀ (Nol):** Tidak ada perbedaan tingkat popularitas yang nyata antar-studio produksi.
- **H₁ (Alternatif):** Studio produksi memberikan pengaruh nyata terhadap tingkat popularitas anime.

Tolak H₀ jika `P-Value < α (0.05)`.

---

## Struktur File

```
project/
└── data/
    └── processed/
        └── anime_dataset_pre.csv   ← Dataset input
└── analisis_kruskal.py             ← Skrip utama
```

---

## Cara Menjalankan

```bash
python analisis_kruskal.py
```

Pastikan dependensi berikut sudah terinstall:
```bash
pip install pyspark scipy
```
