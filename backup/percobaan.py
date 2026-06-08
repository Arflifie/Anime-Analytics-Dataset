import os
import sys

# Mengunci jalur Python yang sedang berjalan di Laragon/VS Code saat ini
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
# pyrefly: ignore [missing-import]
from scipy.stats import kruskal

def jalankan_analisis_kruskal():
    # 1. Bikin Session Spark + TRIK ANTI-CRASH UNTUK JAVA VERSI BARU DI WINDOWS
    spark = SparkSession.builder \
        .appName("AnalisisKruskalStudio") \
        .master("local[*]") \
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/javax.security.auth=ALL-UNNAMED") \
        .config("spark.executor.extraJavaOptions", "--add-opens=java.base/javax.security.auth=ALL-UNNAMED") \
        .getOrCreate()
    
    print("=== Mesin Spark Berhasil Dinyalakan ===")
    
    # 2. Baca data yang sudah kamu cleaning
    path_data_bersih = "data/processed/anime_dataset_pre.csv" 
    df = spark.read.csv(path_data_bersih, header=True, inferSchema=True)
    
    print("\n>>> Mengintip 5 data teratas hasil preprocessing:")
    df.show(5)
    
    # 3. PROSES PYSPARK: Ambil daftar nama studio unik
    daftar_studio = [row['studios'] for row in df.select('studios').distinct().collect()]
    print(f"Ditemukan {len(daftar_studio)} studio unik untuk dianalisis.")
    
    # 4. PROSES PYSPARK: Kelompokkan nilai popularitas berdasarkan studionya
    kelompok_popularitas = []
    
    for studio in daftar_studio:
        # Filter data pakai Spark RDD agar bisa di-collect jadi list Python biasa
        skor_popularitas_studio = df.filter(df['studios'] == studio) \
                                    .select('popularity') \
                                    .rdd.flatMap(lambda x: x).collect()
        
        if len(skor_popularitas_studio) > 0:
            kelompok_popularitas.append(skor_popularitas_studio)
            
    # Matikan mesin Spark jika ekstraksi data ke List sudah selesai
    spark.stop()
    print("=== Proses PySpark Selesai. Masuk ke Pengujian SciPy ===")
    
    # 5. UJI STATISTIK: Kruskal-Wallis dari SciPy
    stat, p_value = kruskal(*kelompok_popularitas)
    
    print("\n================ HASIL UJI STATISTIK KRUSKAL-WALLIS ================")
    print(f"Nilai H-Statistic : {stat:.4f}")
    print(f"Nilai P-Value     : {p_value}")
    
    # 6. KESIMPULAN
    if p_value < 0.05:
        print("\nKesimpulan: SIGNIFIKAN! Studio produksi memberikan pengaruh yang nyata terhadap tingkat popularitas anime.")
    else:
        print("\nKesimpulan: TIDAK SIGNIFIKAN! Tidak ada perbedaan tingkat popularitas yang nyata antar-studio produksi.")
    print("====================================================================")

if __name__ == "__main__":
    jalankan_analisis_kruskal()