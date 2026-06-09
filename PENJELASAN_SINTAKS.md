# 📘 Penjelasan Sintaks Kode — Analisis Popularitas Anime

Dokumen ini menjelaskan setiap baris kode secara rinci: fungsi, tujuan, alasan penggunaan, dan risiko jika tidak digunakan.

---

## 📌 Daftar Isi

1. [Cell 1 — Import Libraries](#-cell-1--import-libraries)
2. [Cell 2 — Inisialisasi Spark](#-cell-2--inisialisasi-spark)
3. [Cell 3 — Baca Data & Ekstraksi PySpark](#-cell-3--baca-data--ekstraksi-pyspark)
4. [Cell 4 — Analisis Pandas & Kruskal-Wallis](#-cell-4--analisis-pandas--kruskal-wallis)

---

## 📦 Cell 1 — Import Libraries

```python
import os
```
| | Keterangan |
|---|---|
| **Fungsi** | Mengakses fitur sistem operasi |
| **Tujuan** | Untuk set environment variable PySpark |
| **Pantangan jika tidak ada** | PySpark tidak bisa diarahkan ke Python yang benar → error saat start |

---

```python
import sys
```
| | Keterangan |
|---|---|
| **Fungsi** | Mengakses info interpreter Python yang sedang berjalan |
| **Tujuan** | Mengambil path Python aktif via `sys.executable` |
| **Pantangan jika tidak ada** | Tidak bisa tahu path Python yang dipakai Jupyter saat ini |

---

```python
import itertools
```
| | Keterangan |
|---|---|
| **Fungsi** | Utilitas untuk operasi iterasi tingkat lanjut |
| **Tujuan** | Meratakan list of list menjadi satu list panjang secara efisien |
| **Pantangan jika tidak ada** | Harus pakai nested loop manual yang jauh lebih lambat |

---

```python
import pandas as pd
```
| | Keterangan |
|---|---|
| **Fungsi** | Library analisis data tabular |
| **Tujuan** | Membuat DataFrame untuk analisis statistik dan persiapan visualisasi |
| **Pantangan jika tidak ada** | Tidak bisa `groupby`, `agg`, `median`, dan semua operasi Pandas di Cell 4 |

---

```python
import matplotlib.pyplot as plt
```
| | Keterangan |
|---|---|
| **Fungsi** | Library dasar untuk membuat grafik |
| **Tujuan** | Membuat canvas, subplot, judul utama, dan merender grafik |
| **Pantangan jika tidak ada** | Seaborn tidak bisa menampilkan grafik karena Seaborn berjalan di atas Matplotlib |

---

```python
import seaborn as sns
```
| | Keterangan |
|---|---|
| **Fungsi** | Library visualisasi statistik berbasis Matplotlib |
| **Tujuan** | Membuat barplot, boxplot, dan stripplot dengan tampilan lebih rapi |
| **Pantangan jika tidak ada** | Harus buat grafik manual dengan Matplotlib yang jauh lebih verbose |

---

```python
from pyspark.sql import SparkSession
```
| | Keterangan |
|---|---|
| **Fungsi** | Kelas utama untuk membuat dan mengelola sesi Spark |
| **Tujuan** | Titik masuk semua operasi PySpark |
| **Pantangan jika tidak ada** | Tidak bisa membuat session Spark sama sekali — seluruh Cell 2 dan 3 tidak jalan |

---

```python
import pyspark.sql.functions as F
```
| | Keterangan |
|---|---|
| **Fungsi** | Kumpulan fungsi built-in Spark SQL |
| **Tujuan** | Mengakses `F.collect_list`, `F.count`, `F.col` untuk operasi agregasi |
| **Pantangan jika tidak ada** | Harus tulis nama lengkap tiap fungsi atau pakai RDD yang jauh lebih rumit |

---

```python
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)
```
| | Keterangan |
|---|---|
| **Fungsi** | Kelas-kelas tipe data Spark |
| **Tujuan** | Mendefinisikan schema manual agar Spark tidak scan file dua kali |
| **Pantangan jika tidak ada** | Harus pakai `inferSchema=True` → Spark scan file dua kali → penyebab buffer error |

---

```python
from scipy.stats import kruskal
```
| | Keterangan |
|---|---|
| **Fungsi** | Fungsi uji statistik Kruskal-Wallis dari SciPy |
| **Tujuan** | Menguji apakah distribusi popularitas antar studio berbeda signifikan |
| **Pantangan jika tidak ada** | Tidak ada uji inferensial → analisis hanya deskriptif, tidak bisa menarik kesimpulan statistik |

---

```python
os.environ['PYSPARK_PYTHON']        = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
```
| | Keterangan |
|---|---|
| **Fungsi** | Set environment variable yang dibaca PySpark saat startup |
| **Tujuan** | Memaksa Spark worker dan driver pakai Python yang sama dengan Jupyter |
| **Pantangan jika tidak ada** | PySpark bisa pakai Python versi berbeda dari environment Jupyter → `Py4JJavaError` saat start session |

---

## ⚡ Cell 2 — Inisialisasi Spark

```python
spark = SparkSession.builder \
    .appName("AnalisisPopularitasStudio") \
```
| | Keterangan |
|---|---|
| **Fungsi** | Memberi nama aplikasi Spark |
| **Tujuan** | Identifikasi di Spark UI jika dibuka di browser `localhost:4040` |
| **Pantangan jika tidak ada** | Session tetap jalan tapi nama tampil sebagai `None` di UI |

---

```python
    .master("local[*]") \
```
| | Keterangan |
|---|---|
| **Fungsi** | Menentukan mode eksekusi Spark |
| **Tujuan** | `local[*]` artinya pakai semua core CPU di mesin lokal secara paralel |
| **Pantangan jika tidak ada** | Spark tidak tahu mau jalan di mana → error saat `.getOrCreate()` |

---

```python
    .config("spark.driver.memory",       "4g") \
    .config("spark.executor.memory",     "4g") \
```
| | Keterangan |
|---|---|
| **Fungsi** | Alokasi RAM untuk driver dan executor |
| **Tujuan** | Mencegah buffer error saat membaca CSV dan melakukan `.collect()` |
| **Pantangan jika tidak ada** | Spark pakai default 1g → sangat mudah crash dengan dataset besar → penyebab `INTERNAL_ERROR buffer limit` |

---

```python
    .config("spark.driver.maxResultSize", "2g") \
```
| | Keterangan |
|---|---|
| **Fungsi** | Batas maksimal ukuran data yang bisa dikembalikan ke driver dari executor |
| **Tujuan** | Memastikan hasil `collect()` ratusan studio tidak melebihi batas |
| **Pantangan jika tidak ada** | Jika hasil `collect()` besar → Spark lempar error `Result size exceeds threshold` |

---

```python
    .config("spark.sql.shuffle.partitions", "8") \
```
| | Keterangan |
|---|---|
| **Fungsi** | Jumlah partisi saat operasi shuffle seperti `groupBy` |
| **Tujuan** | Default Spark adalah 200 partisi — terlalu banyak untuk data lokal. Angka 8 disesuaikan dengan core lokal |
| **Pantangan jika tidak ada** | Spark buat 200 task kecil-kecil untuk `groupBy` → overhead tinggi → lambat di mode lokal |

---

```python
    .getOrCreate()
```
| | Keterangan |
|---|---|
| **Fungsi** | Ambil session yang sudah ada atau buat baru |
| **Tujuan** | Mencegah duplikasi session jika cell dijalankan ulang di Jupyter |
| **Pantangan jika tidak ada** | Tidak ada — ini memang satu-satunya cara membuat SparkSession |

---

```python
spark.sparkContext.setLogLevel("WARN")
```
| | Keterangan |
|---|---|
| **Fungsi** | Set level log Spark |
| **Tujuan** | Menekan output INFO yang sangat verbose agar output Jupyter bersih |
| **Pantangan jika tidak ada** | Jupyter dibanjiri ratusan baris log INFO setiap operasi Spark dijalankan |

---

## 📂 Cell 3 — Baca Data & Ekstraksi PySpark

```python
schema = StructType([...])
```
| | Keterangan |
|---|---|
| **Fungsi** | Mendefinisikan struktur kolom dan tipe datanya secara eksplisit |
| **Tujuan** | Spark langsung tahu tipe tiap kolom tanpa perlu scan file |
| **Pantangan jika tidak ada** | `inferSchema=True` → Spark baca file dua kali → penyebab utama buffer error |

---

```python
StructField("title",        StringType(),  True)
StructField("popularity",   DoubleType(),  True)
StructField("favorites",    DoubleType(),  True)
StructField("studios",      IntegerType(), True)
StructField("studios_name", StringType(),  True)
```
| Kolom | Tipe | Alasan |
|---|---|---|
| `title` | `StringType` | Nama anime berupa teks |
| `popularity` | `DoubleType` | Hasil normalisasi MinMaxScaler → nilai desimal 0.0–1.0 |
| `favorites` | `DoubleType` | Hasil normalisasi MinMaxScaler → nilai desimal 0.0–1.0 |
| `studios` | `IntegerType` | Hasil LabelEncoder → angka bulat |
| `studios_name` | `StringType` | Nama asli studio berupa teks |

> **Pantangan tipe salah:** Operasi aritmetika dan filter bisa gagal atau hasilnya salah

---

```python
df = spark.read.csv(path_data_bersih, header=True, schema=schema)
```
| | Keterangan |
|---|---|
| **Fungsi** | Membaca file CSV menjadi Spark DataFrame |
| **Tujuan** | Memuat data ke dalam Spark untuk diproses secara terdistribusi |
| **Pantangan `header=False`** | Baris pertama (nama kolom) ikut terbaca sebagai data → mengacaukan seluruh analisis |

---

```python
daftar_studio = [
    row['studios'] for row in df.select('studios').distinct().collect()
    if row['studios'] is not None
]
```
| | Keterangan |
|---|---|
| **Fungsi** | List comprehension untuk mengambil nilai unik kolom `studios` |
| **Tujuan** | Mengetahui jumlah total studio unik yang ada di dataset |
| **Alasan `is not None`** | Filter baris null agar tidak masuk ke list |
| **Pantangan jika tidak difilter** | `None` ikut masuk → `groupBy` menghasilkan grup null yang tidak berguna |

---

```python
hasil = (
    df.groupBy('studios', 'studios_name')
```
| | Keterangan |
|---|---|
| **Fungsi** | Mengelompokkan data berdasarkan ID dan nama studio sekaligus |
| **Tujuan** | Agar nama studio ikut terbawa ke hasil tanpa perlu join terpisah |
| **Pantangan jika hanya `groupBy('studios')`** | Nama studio tidak ikut → harus mapping manual lagi setelahnya |

---

```python
      .agg(
          F.collect_list('popularity').alias('skor_popularitas'),
          F.count('popularity').alias('jumlah')
      )
```
| | Keterangan |
|---|---|
| **Fungsi** | `collect_list` mengumpulkan semua nilai popularity per studio menjadi satu list. `count` menghitung jumlah anime |
| **Tujuan** | Menghasilkan struktur data yang langsung siap dipakai Kruskal-Wallis |
| **Pantangan jika pakai `collect_set`** | Nilai duplikat dihapus → distribusi data berubah → hasil statistik tidak akurat |

---

```python
      .filter(F.col('jumlah') >= 10)
```
| | Keterangan |
|---|---|
| **Fungsi** | Menyaring studio yang punya minimal 10 anime |
| **Tujuan** | Menjaga keandalan uji statistik — sampel terlalu kecil membuat uji tidak reliable |
| **Pantangan jika tidak difilter** | Studio dengan 1–2 anime ikut diuji → Kruskal-Wallis tidak valid untuk sampel sangat kecil |

---

```python
kelompok_popularitas_full_population = [row['skor_popularitas'] for row in hasil]
groups_filtered_for_test             = [row['studios']          for row in hasil]
groups_name_for_label                = [row['studios_name']     for row in hasil]
```
| | Keterangan |
|---|---|
| **Fungsi** | Memisahkan hasil `collect()` menjadi tiga list paralel |
| **Tujuan** | Indeks ke-i di ketiga list selalu merujuk studio yang sama — data tidak tertukar |
| **Pantangan jika urutan berbeda** | Data studios dan popularity tidak sinkron → hasil analisis salah total |

---

```python
spark.stop()
```
| | Keterangan |
|---|---|
| **Fungsi** | Menutup dan membebaskan semua resource Spark |
| **Tujuan** | Data sudah di memori Python, Spark tidak dibutuhkan lagi. Membebaskan RAM yang dipakai JVM |
| **Pantangan jika tidak dipanggil** | JVM Spark tetap berjalan di background → memakan RAM → bisa crash di langkah selanjutnya |

---

## 🔬 Cell 4 — Analisis Pandas & Kruskal-Wallis

```python
df_viz_full = pd.DataFrame({
    'studios': list(itertools.chain.from_iterable(
        [sid] * len(scores)
        for sid, scores in zip(groups_filtered_for_test,
                               kelompok_popularitas_full_population)
    )),
    ...
})
```
| | Keterangan |
|---|---|
| **Fungsi** | `chain.from_iterable` meratakan list of list. `[sid] * len(scores)` menduplikasi ID studio sebanyak jumlah anime miliknya. `zip()` menjamin pasangan ID-scores selalu sinkron |
| **Tujuan** | Mengubah struktur data dari "list per studio" menjadi long-format dimana setiap baris adalah satu anime |
| **Pantangan jika tidak pakai `zip`** | Bisa salah pasang ID dengan scores studio yang berbeda |

---

```python
top_10_ids = (
    df_viz_full.groupby('studios')['popularity']
               .median()
               .sort_values(ascending=True)
               .head(10)
               .index.tolist()
)
```
| | Keterangan |
|---|---|
| **Fungsi** | Hitung median popularity per studio, urutkan dari terkecil, ambil 10 teratas |
| **Tujuan** | Mengidentifikasi 10 studio dengan performa popularitas terbaik |
| **Alasan `ascending=True`** | Popularity adalah skor rank — nilai kecil = lebih populer |
| **Alasan pakai median bukan mean** | Median tahan terhadap outlier — satu anime viral tidak mendistorsi gambaran keseluruhan studio |
| **Pantangan jika `ascending=False`** | Mengambil 10 studio paling tidak populer, bukan terpopuler |

---

```python
id_to_name = {
    sid: name for sid, name in
    zip(groups_filtered_for_test, groups_name_for_label)
}
```
| | Keterangan |
|---|---|
| **Fungsi** | Dictionary comprehension membuat peta ID → nama studio |
| **Tujuan** | Mengkonversi ID numerik ke nama asli untuk label grafik |
| **Pantangan jika tidak ada** | Sumbu X grafik menampilkan angka seperti `856`, `234` yang tidak bermakna bagi audiens |

---

```python
top_10_summary_stats['studio_name'] = top_10_summary_stats['studios'].map(id_to_name)
```
| | Keterangan |
|---|---|
| **Fungsi** | `.map()` mengganti setiap nilai ID dengan nama studio dari dictionary |
| **Tujuan** | Menambahkan kolom nama ke tabel statistik untuk label tabel di Grafik 4 |
| **Pantangan jika tidak ada** | Tabel di visualisasi tetap menampilkan ID angka |

---

```python
df_viz_top10['studio_name'] = pd.Categorical(
    df_viz_top10['studio_name'],
    categories=category_order_named,
    ordered=True
)
```
| | Keterangan |
|---|---|
| **Fungsi** | Mengubah kolom menjadi tipe Categorical dengan urutan eksplisit |
| **Tujuan** | Memaksa Seaborn menampilkan studio di sumbu X sesuai urutan median terkecil ke terbesar |
| **Pantangan jika tidak pakai Categorical** | Seaborn mengurutkan berdasarkan alfabet → urutan grafik tidak mencerminkan ranking popularitas |

---

```python
kelompok_top10 = [
    kelompok_popularitas_full_population[i]
    for i, sid in enumerate(groups_filtered_for_test)
    if sid in top_10_ids
]
```
| | Keterangan |
|---|---|
| **Fungsi** | List comprehension memfilter hanya data studio yang masuk Top 10 |
| **Tujuan** | Memastikan Kruskal-Wallis hanya menguji kelompok yang divisualisasikan — konsistensi antara uji dan grafik |
| **Pantangan jika pakai semua studio** | Menguji ratusan studio sekaligus → p-value hampir pasti kecil bukan karena perbedaan nyata tapi karena jumlah grup yang besar |

---

```python
h_stat, p_value = kruskal(*kelompok_top10)
```
| | Keterangan |
|---|---|
| **Fungsi** | Menjalankan uji Kruskal-Wallis. `*` meng-unpack list of list menjadi argumen terpisah per studio |
| **Tujuan** | Mendapatkan H-statistic dan p-value untuk keputusan statistik |
| **Alasan non-parametrik** | Data popularity adalah hasil normalisasi ranking — tidak bisa diasumsikan normal → ANOVA tidak valid |
| **Pantangan jika pakai ANOVA** | Asumsi normalitas dilanggar → hasil uji tidak dapat dipercaya |

---

## 📊 Cara Membaca Hasil

| Nilai | Arti |
|---|---|
| `p_value < 0.05` | **Tolak H0** — ada perbedaan signifikan antar studio |
| `p_value >= 0.05` | **Gagal Tolak H0** — tidak cukup bukti perbedaan signifikan |
| `H-statistic tinggi` | Perbedaan antar grup besar relatif terhadap variasi dalam grup |
| `H-statistic rendah` | Distribusi antar grup relatif serupa |

> ⚠️ **Catatan penting:** Hasil tidak signifikan pada Top 10 **bukan berarti** studio tidak berpengaruh sama sekali. Ini berarti di antara 10 studio terbaik, performa mereka **relatif setara** satu sama lain.

---

*Dokumen ini merupakan bagian dari proyek [Analisis Popularitas Anime](README.md)*
