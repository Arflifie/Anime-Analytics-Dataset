# IMPORT LIBARIES YANG DIBUTUHKAN
import os
import sys
from pyspark.sql import SparkSession
from scipy.stats import kruskal 

# SET ENV VARIABEL PYTHON UNTUK PYSPARK AGAR TIDAK ERROR KETIKA JALANKAN ANALISIS
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def jalankan_analisis_kruskal():
    print("=== Mesin Spark Berhasil Dinyalakan ===")

    # TAHAP 1 INISIALISASI MESIN SPARK
    spark = SparkSession.builder \
        .appName("AnalisisPopularitasStudio") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    print("=== Mesin Spark Berhasil Dinyalakan ===")

    # TAHAP 2 INGESTION (MEMBACA DATA)
    path_data_bersih = "data/processed/anime_dataset_pre.csv"
    print("\n>>> Mengintip 5 data teratas hasil preprocessing:")
    df = spark.read.csv(path_data_bersih, header=True, inferSchema=True)
    df.show(5)

    # TAHAP 3 EKSTRAKSI DATA (MENCARI NAMA STUDIO UNIK)
    daftar_studio = [row['studios'] for row in df.select('studios').distinct().collect() if row['studios'] is not None]
    print(f"Ditemukan {len(daftar_studio)} studio unik untuk dianalisis.")

    # TAHAP 4 DISTRIBUSI DATA (MENGELLOMPOKKAN POPULARITAS BERDASARKAN STUDIO)
    kelompok_popularitas = []
    for studio in daftar_studio:
        skor_popularitas_studio = df.filter(df['studios'] == studio) \
                                    .select('popularity') \
                                    .rdd.flatMap(lambda x: x).collect()
        if len(skor_popularitas_studio) > 0:
            kelompok_popularitas.append(skor_popularitas_studio)

    # TAHAP 5 PENYELESAIAN MESIN SPARK
    spark.stop()
    print("=== Proses PySpark Selesai. Masuk ke Pengujian SciPy ===")

    # TAHAP 6 UJI STATISTIK KRUSKAL-WALLIS
    stat, p_value = kruskal(*kelompok_popularitas)

    # TAHAP 7 INTERPRETASI HASIL
    print("\n================ HASIL UJI STATISTIK KRUSKAL-WALLIS ================")
    print(f"Nilai H-Statistic : {stat:.4f}")
    print(f"Nilai P-Value     : {p_value}")

    if p_value < 0.05:
        print("\nKesimpulan: SIGNIFIKAN! (P-Value < 0.05)")
        print("Studio produksi memberikan pengaruh yang nyata terhadap tingkat popularitas anime.")
    else:
        print("\nKesimpulan: TIDAK SIGNIFIKAN! (P-Value >= 0.05)")
        print("Tidak ada perbedaan tingkat popularitas yang nyata antar-studio produksi.")
    print("====================================================================")

# JALANKAN ANALISIS KRUSKAL-WALLIS KETIKA FILE INI DIEKSEKUSI
if __name__ == "__main__":
    jalankan_analisis_kruskal()