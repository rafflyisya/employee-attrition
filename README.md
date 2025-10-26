# Laporan Tugas 1 Statistics Machine Learning  
## Employee Attrition Prediction  
**Rochmat Pornomo Prasetya** — *5003231007*
**Raffly Isya Ramadhan** — *5003231135*
**Andyka Nabil Putra** — *5003231140*

---
## Daftar Isi

- [Domain Proyek: Keuangan](#domain-proyek-keuangan)
  - [Referensi](#referensi)
- [Business Understanding](#business-understanding)
  - [Problem Statements](#problem-statements)
  - [Goals](#goals)
  - [Solution Statements](#solution-statements)
  - [Project Benefits](#project-benefits)
- [Data Understanding](#data-understanding)
  - [Sumber Data](#sumber-data)
  - [Deskripsi Fitur](#deskripsi-fitur)
  - [Penjelasan Kontekstual Fitur](#penjelasan-kontekstual-fitur)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis---deskripsi-variabel)
    - [Missing Value & Outliers](#exploratory-data-analysis---menangani-missing-value-dan-outliers)
    - [Univariate Analysis](#exploratory-data-analysis---univariate-analysis)
    - [Multivariate Analysis](#exploratory-data-analysis---multivariate-analysis)
    - [Kesimpulan EDA](#kesimpulan-eda)
- [Data Preparation](#data-preparation)
  - [Label Encoding dengan Mapping pada Fitur Target](#1-label-encoding-dengan-mapping-pada-fitur-target)
  - [Splitting Dataset](#2-splitting-dataset)
  - [Feature Engineering, Data Cleaning and Preprocessing](#3-feature-engineering-data-cleaning-and-preprocessing)
- [Model Training, Comparison, Selection and Tuning](#model-training-comparison-selection-and-tuning)
  - [Model Selection](#1-model-selection)
  - [Feature Selection](#2-feature-selection)
  - [Hyperparameter Tuning](#3-hyperparameter-tuning)
- [Model Testing and Evaluation](#model-testing-and-evaluation)
  - [Data Test Predict](#1-data-test-predict)
  - [Best Model Evaluation](#2-best-model-evaluation)
    - [Classification Report](#classification-report)
    - [Metode Evaluasi Lanjutan](#metode-evaluasi-lanjutan)
    - [Confusion Matrix](#confusion-matrix)
    - [Plot ROC-AUC Curve](#plot-roc-auc-curve)
    - [Plot PR-AUC Curve](#plot-pr-auc-curve)
- [Save Best Model](#save-best-model)
- [Model Interpretation](#model-interpretation)
  - [Interpretation with SHAP Values](#1-interpretation-with-shap-values)
  - [Feature Importance](#2-feature-importance)
- [Financial Result](#financial-result)
- [Conclusions](#conclusions)
  - [Ringkasan Proyek](#ringkasan-proyek)
  - [Hasil dan Evaluasi Model](#hasil-dan-evaluasi-model)
  - [Penanganan Ketidakseimbangan Data](#penanganan-ketidakseimbangan-data)
  - [Interpretasi dan Validasi Model](#interpretasi-dan-validasi-model)
  - [Estimasi Nilai Finansial](#estimasi-nilai-finansial)
  - [Langkah Selanjutnya](#langkah-selanjutnya)

---
## **Domain Proyek: Keuangan**

<figure>
    <center><img src="[img/credit-card-terminal-payment.jpg](https://whatfix.com/blog/wp-content/uploads/2022/09/employee-churn.png)" alt="Employee Attrition"></center>
</figure>

Manajemen sumber daya manusia merupakan salah satu aspek krusial dalam menjaga keberlanjutan dan daya saing perusahaan di era bisnis modern. Keberhasilan sebuah organisasi tidak hanya ditentukan oleh strategi bisnis dan inovasi produk, tetapi juga oleh kemampuan dalam mempertahankan karyawan yang berkompeten dan berprestasi. Fenomena tingginya tingkat attrition atau employee turnover menjadi tantangan yang signifikan karena dapat menimbulkan berbagai konsekuensi, mulai dari meningkatnya biaya rekrutmen dan pelatihan, terganggunya produktivitas, hingga menurunnya moral kerja tim [[1]](https://www.americanprogress.org/article/there-are-significant-business-costs-to-replacing-employees/?utm_source).
Banyak perusahaan kini berinvestasi dalam strategi retensi untuk menekan laju turnover, seperti peningkatan kepuasan kerja, penyesuaian beban kerja, dan pengembangan jalur karier yang lebih jelas [[2]](https://www.achievers.com/blog/employee-turnover-by-industry/?utm_source). Namun, memahami penyebab karyawan keluar tidaklah sederhana, karena keputusan tersebut dipengaruhi oleh berbagai faktor—baik dari sisi individu, seperti kepuasan dan keseimbangan kerja, maupun dari sisi organisasi, seperti lingkungan kerja dan kebijakan perusahaan [[3]](https://pmc.ncbi.nlm.nih.gov/articles/PMC9309793/?utm_source).
Untuk menghadapi tantangan ini, pendekatan berbasis data menjadi solusi yang menjanjikan. Dengan memanfaatkan data profil karyawan, lingkungan kerja, serta status kepegawaian, perusahaan dapat mengembangkan model prediktif yang mampu mengidentifikasi karyawan berisiko tinggi untuk keluar. Melalui penerapan teknik data science dan machine learning, organisasi tidak hanya dapat memprediksi potensi turnover dengan lebih akurat, tetapi juga memperoleh wawasan strategis mengenai faktor-faktor utama yang memengaruhi retensi. Pendekatan ini memungkinkan manajemen untuk mengambil langkah proaktif dalam menjaga stabilitas tenaga kerja, meningkatkan kepuasan karyawan, dan memperkuat daya saing jangka panjang perusahaan.


---

**Referensi:**

[1] American Progress (2012). *There Are Significant Business Costs to Replacing Employees
*. Retrieved from [https://www.americanprogress.org/](https://www.americanprogress.org/article/there-are-significant-business-costs-to-replacing-employees/?utm_source)  
[2] Achievers (2025). *Employee turnover by industry: The hidden cost of attrition in 2025*. Retrieved from [https://www.achievers.com/sg/](https://www.achievers.com/blog/employee-turnover-by-industry/?utm_source)  
[3] National Library of Medicine (2022). *Factors Affecting Employee’s Retention: Integration of Situational Leadership With Social Exchange Theory*. International Journal of Computer Applications, 166(1), 1–6. [https://pmc.ncbi.nlm.nih.gov/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9309793/?utm_source)

---

## Business Understanding

### Problem Statements  
Dalam dunia bisnis modern, sumber daya manusia menjadi aset strategis yang menentukan keberlanjutan dan daya saing perusahaan. Namun, banyak organisasi menghadapi tantangan serius berupa meningkatnya jumlah karyawan yang meninggalkan perusahaan atau dikenal sebagai employee attrition (turnover). Tingginya tingkat turnover tidak hanya menyebabkan hilangnya talenta dan pengetahuan berharga, tetapi juga menimbulkan biaya besar bagi perusahaan, seperti rekrutmen, pelatihan, serta penurunan produktivitas dan moral kerja tim. Oleh karena itu, penting bagi manajemen untuk memahami faktor-faktor yang memengaruhi keputusan karyawan untuk bertahan atau keluar. Melalui analisis berbasis data yang mencakup profil karyawan, kondisi kerja, tingkat kepuasan, jam lembur, dan keseimbangan kerja–hidup, perusahaan dapat membangun model prediktif menggunakan pendekatan data science dan machine learning untuk memproyeksikan risiko attrition. Dengan demikian, organisasi dapat mengambil langkah preventif yang lebih tepat sasaran dalam meningkatkan retensi, kepuasan kerja, serta mengurangi kerugian finansial akibat tingginya tingkat pergantian karyawan.

Berdasarkan hal tersebut, berikut adalah pernyataan masalah yang diangkat:

- **Pernyataan Masalah 1:** Bagaimana mengidentifikasi faktor-faktor utama yang memengaruhi keputusan karyawan untuk meninggalkan perusahaan (employee attrition)?
- **Pernyataan Masalah 2:** Bagaimana membangun model prediksi yang mampu memperkirakan kemungkinan seorang karyawan akan keluar dengan tingkat akurasi yang tinggi?
- **Pernyataan Masalah 3:** Bagaimana memanfaatkan hasil analisis data untuk merumuskan strategi retensi yang efektif dalam meningkatkan kepuasan, keterikatan, dan loyalitas karyawan terhadap perusahaan?

### Goals  
Untuk menjawab pernyataan masalah di atas, tujuan proyek ini dirumuskan sebagai berikut:

- **Tujuan 1:** Melakukan eksplorasi dan analisis data historis karyawan untuk mengidentifikasi pola dan variabel yang memiliki korelasi tinggi terhadap perilaku employee attrition (keputusan karyawan untuk keluar). 
- **Tujuan 2:** Membangun model prediktif berbasis machine learning yang mampu memperkirakan probabilitas seorang karyawan akan meninggalkan perusahaan.
- **Tujuan 3:** Memberikan rekomendasi dan rencana aksi yang berbasis pada hasil prediksi model untuk meningkatkan retensi pelanggan dan memaksimalkan _Customer Lifetime Value_ (CLV).

### Solution Statements  
Untuk mencapai tujuan-tujuan tersebut, solusi yang akan diimplementasikan meliputi:

- **Eksperimen Berbagai Algoritma Klasifikasi:**  
  Membangun dan membandingkan performa beberapa algoritma seperti:
  - Decision Tree  
  - Random Forest  
  - LightGBM  

- **Optimasi Model dengan Hyperparameter Tuning:**  
  Menggunakan pendekatan seperti Bayesian Optimization dengan optuna untuk mendapatkan konfigurasi model terbaik.

- **Evaluasi Model dengan Metrik yang Relevan:**  
  Menggunakan metrik seperti:
  - Accuracy untuk mengukur prediksi keseluruhan  
  - Precision, Recall, F1-Score untuk menilai performa pada kelas churn  
  - ROC-AUC untuk mengevaluasi kemampuan model dalam membedakan kelas  
  - Confusion Matrix untuk melihat distribusi hasil prediksi

- **Analisis Fitur dan Visualisasi:**  
  Menyajikan visualisasi seperti feature importance dan correlation heatmap untuk menginterpretasikan fitur-fitur utama yang berkontribusi terhadap churn.

### Project Benefits  
Dengan implementasi solusi ini, manfaat utama yang diharapkan antara lain:

- **Penghematan Biaya:** Mengurangi biaya rekrutmen, pelatihan, dan onboarding karyawan baru dengan mempertahankan karyawan yang ada.  
- **Peningkatan Retensi Pelanggan:** Mengidentifikasi karyawan berisiko tinggi untuk keluar dan memungkinkan intervensi retensi yang tepat sasaran dan personal.
- **Perencanaan SDM yang lebih efisien:** Mengalokasikan sumber daya departemen HR untuk program retensi dan pengembangan secara lebih strategis berdasarkan data.
- **Perbaikan budaya perusahaan:** Mengidentifikasi akar penyebab attrition (seperti kepuasan kerja, work-life balance, atau hubungan dengan manajer) untuk membentuk lingkungan kerja yang lebih positif.  
- **Perlindungan Aset Pengetahuan:** Mencegah hilangnya pengetahuan institusional (institutional knowledge) dan keahlian kritis ketika karyawan berpengalaman meninggalkan perusahaan.

---

## Data Understanding

### Sumber Data  
Dataset yang digunakan dalam proyek ini diperoleh dari situs [Kaggle](www.kaggle.com/competitions/tugas-1-sml-a-2025/overview/citation). Dataset ini mencakup informasi tentang **1.467 karyawant**, yang mencatat berbagai aspek kondisi dan identitas yang mempresentasikan kecenderungan karyawan untuk melakukan attrition.

Dataset ini memiliki **35 fitur**, yang mencakup usia, frekuensi perjalanan, gaji harian, jarak tempat tinggal, tingkat pendidikan, tingkat kepuasan pekerjaan, dan lainnya. proyek ini bertujuan memprediksi employee attrition dimana sebesar 83,84% karyawan yang termasuk kategori attrition. Kondisi ketidakseimbangan kelas ini menjadi tantangan utama dalam membangun model prediktif yang akurat untuk mengidentifikasi pola-pola karyawan berisiko keluar dari perusahaan.

### Deskripsi Fitur

| Nama Fitur                         | Deskripsi                                                                 | Tipe Data    |
|------------------------------------|---------------------------------------------------------------------------|--------------|
| `id`                        | ID unik karyawan untuk identifikasi                                                         | `int64`      |
| `Age`   | Usia karyawan                | `int64t`     |
| `BusinessTravel`                     | Usia pelanggan (dalam tahun)                                              | `int64`      |
| `DailyRate`                           | Jenis kelamin pelanggan (`M`/`F`)                                         | `object`     |
| `Department`                  | Jumlah tanggungan                                                         | `int64`      |
| `DistanceFromHome`                  | Tingkat pendidikan: Graduate, High School, College, dll.                  | `object`     |
| `Education`                   | Status pernikahan pelanggan                                               | `object`     |
| `EducationField`                  | Kategori pendapatan tahunan pelanggan                                     | `object`     |
| `EmployeeCount`                    | Jenis kartu kredit                                                        | `object`     |
| `EmployeeNumber`                   | Lama hubungan dengan bank (dalam bulan)                                   | `int64`      |
| `EnvironmentSatisfaction`         | Jumlah produk perbankan yang dimiliki                                     | `int64`      |
| `months_inactive_12_mon`           | Jumlah bulan tidak aktif dalam 12 bulan terakhir                          | `int64`      |
| `contacts_count_12_mon`            | Jumlah interaksi dengan bank dalam 12 bulan terakhir                      | `int64`      |
| `credit_limit`                     | Batas maksimal kredit kartu pelanggan                                     | `float64`    |
| `total_revolving_bal`              | Saldo bergulir (tidak dibayar penuh)                                      | `int64`      |
| `avg_open_to_buy`                  | Rata-rata jumlah kredit yang tersedia untuk digunakan                     | `float64`    |
| `total_amt_chng_q4_q1`             | Perubahan jumlah transaksi dari Q1 ke Q4                                  | `float64`    |
| `total_trans_amt`                  | Total nilai transaksi selama 12 bulan terakhir                            | `int64`      |
| `total_trans_ct`                   | Total jumlah transaksi selama 12 bulan terakhir                           | `int64`      |
| `total_ct_chng_q4_q1`              | Perubahan jumlah transaksi dari Q1 ke Q4                                  | `float64`    |
| `avg_utilization_ratio`            | Rata-rata rasio pemanfaatan limit kredit                                  | `float64`    |


### [Exploratory Data Analysis] - Deskripsi Variabel

| Fitur                       | Count   | Mean         | Std          | Min       | 25%          | 50%          | 75%          | Max          |
|-----------------------------|---------|--------------|--------------|-----------|--------------|--------------|--------------|--------------|
| `clientnum`                 | 10127   | 7.391776e+08 | 3.690378e+07 | 708082083 | 7.130368e+08 | 7.179264e+08 | 7.731435e+08 | 8.283431e+08 |
| `customer_age`              | 10127   | 46.32596     | 8.016814     | 26.0      | 41.0         | 46.0         | 52.0         | 73.0         |
| `dependent_count`           | 10127   | 2.346203     | 1.298908     | 0.0       | 1.0          | 2.0          | 3.0          | 5.0          |
| `months_on_book`            | 10127   | 35.92841     | 7.986416     | 13.0      | 31.0         | 36.0         | 40.0         | 56.0         |
| `total_relationship_count`  | 10127   | 3.812580     | 1.554408     | 1.0       | 3.0          | 4.0          | 5.0          | 6.0          |
| `months_inactive_12_mon`    | 10127   | 2.341167     | 1.010622     | 0.0       | 2.0          | 2.0          | 3.0          | 6.0          |
| `contacts_count_12_mon`     | 10127   | 2.455317     | 1.106225     | 0.0       | 2.0          | 2.0          | 3.0          | 6.0          |
| `credit_limit`              | 10127   | 8631.954     | 9088.777     | 1438.3    | 2555.0       | 4549.0       | 11067.5      | 34516.0      |
| `total_revolving_bal`       | 10127   | 1162.814     | 814.9873     | 0.0       | 359.0        | 1276.0       | 1784.0       | 2517.0       |
| `avg_open_to_buy`           | 10127   | 7469.140     | 9090.685     | 3.0       | 1324.5       | 3474.0       | 9859.0       | 34516.0      |
| `total_amt_chng_q4_q1`      | 10127   | 0.759941     | 0.219207     | 0.0       | 0.631        | 0.736        | 0.859        | 3.397        |
| `total_trans_amt`           | 10127   | 4404.086     | 3397.129     | 510.0     | 2155.5       | 3899.0       | 4741.0       | 18484.0      |
| `total_trans_ct`            | 10127   | 64.85869     | 23.47257     | 10.0      | 45.0         | 67.0         | 81.0         | 139.0        |
| `total_ct_chng_q4_q1`       | 10127   | 0.712222     | 0.238086     | 0.0       | 0.582        | 0.702        | 0.818        | 3.714        |
| `avg_utilization_ratio`     | 10127   | 0.274894     | 0.275692     | 0.0       | 0.023        | 0.176        | 0.503        | 0.999        |
     
Berdasarkan hasil analisis data, profil **1.467** karyawan menunjukkan karakteristik kunci **usia rata-rata 37 tahun** dengan **masa kerja rata-rata 7 tahun** di perusahaan. Dari segi kompensasi, **gaji bulanan rata-rata Rp 6,5 juta** dengan **kenaikan gaji tahunan 15%**. Temuan penting mengungkap bahwa karyawan **hanya mengalami 1 kali promosi dalam 2 tahun terakhir**, dan rata-rata telah **bekerja di 2-3 perusahaan sebelumnya**. Pola ini mengindikasikan bahwa stagnasi karir dan riwayat mobilitas kerja dapat menjadi faktor prediktif yang signifikan untuk analisis employee attrition.

### Rata-rata Fitur per Kategori Attrition

| Fitur                         | Attrited Customer | Existing Customer |
|-------------------------------|-------------------|-------------------|
| `clientnum`                   | 7.352614e+08      | 7.399272e+08      |
| `customer_age`                | 46.66             | 46.26             |
| `dependent_count`             | 2.40              | 2.34              |
| `months_on_book`              | 36.18             | 35.88             |
| `total_relationship_count`    | 3.28              | 3.91              |
| `months_inactive_12_mon`      | 2.69              | 2.27              |
| `contacts_count_12_mon`       | 2.97              | 2.36              |
| `credit_limit`                | 8136.04           | 8726.88           |
| `total_revolving_bal`         | 672.82            | 1256.60           |
| `avg_open_to_buy`             | 7463.22           | 7470.27           |
| `total_amt_chng_q4_q1`        | 0.6943            | 0.7725            |
| `total_trans_amt`             | 3095.03           | 4654.66           |
| `total_trans_ct`              | 44.93             | 68.67             |
| `total_ct_chng_q4_q1`         | 0.5544            | 0.7424            |
| `avg_utilization_ratio`       | 0.1625            | 0.2964            |

Berdasarkan analisis perbandingan antara karyawan yang mengundurkan diri dan yang bertahan, terlihat pola yang signifikan. Karyawan yang keluar cenderung lebih muda dengan masa kerja di perusahaan yang lebih pendek. Mereka juga memiliki pendapatan bulanan yang lebih rendah dan jarak tempuh ke kantor yang lebih jauh. Faktor karir menunjukkan perbedaan mencolok dimana karyawan yang keluar memiliki lebih sedikit pengalaman kerja total, waktu di posisi saat ini yang lebih singkat, dan durasi bekerja dengan manajer yang sama yang lebih pendek. Temuan ini mengindikasikan bahwa karyawan yang lebih muda dengan prospek karir yang terbatas dan keterikatan organisasi yang rendah memiliki kecenderungan lebih besar untuk meninggalkan perusahaan.

### [Exploratory Data Analysis] - Menangani Missing Value dan Outliers

Dalam tahap awal pembersihan data, dilakukan pengecekan terhadap **duplikasi data** dan **missing value**. Hasilnya menunjukkan bahwa **tidak terdapat duplikasi data** maupun **missing value** di seluruh kolom fitur maupun target. Hal ini mengindikasikan bahwa dataset sudah lengkap dan tidak memerlukan teknik imputasi lebih lanjut.

| Variabel                  | Jumlah Missing Value |
|---------------------------|----------------------|
| clientnum                 | 0                    |
| churn_flag                | 0                    |
| customer_age              | 0                    |
| gender                    | 0                    |
| dependent_count           | 0                    |
| education_level           | 0                    |
| marital_status            | 0                    |
| income_category           | 0                    |
| card_category             | 0                    |
| months_on_book            | 0                    |
| total_relationship_count  | 0                    |
| months_inactive_12_mon    | 0                    |
| contacts_count_12_mon     | 0                    |
| credit_limit              | 0                    |
| total_revolving_bal       | 0                    |
| avg_open_to_buy           | 0                    |
| total_amt_chng_q4_q1      | 0                    |
| total_trans_amt           | 0                    |
| total_trans_ct            | 0                    |
| total_ct_chng_q4_q1       | 0                    |
| avg_utilization_ratio     | 0                    |

Berdasarkan hasil deteksi outlier menggunakan metode IQR, teridentifikasi beberapa variabel numerik yang mengandung outlier. Variabel MonthlyIncome memiliki 86 outlier, TrainingTimesLastYear mencatat outlier tertinggi sebanyak 174, dan TotalWorkingYears serta YearsAtCompany masing-masing memiliki 52 outlier. Variabel YearsSinceLastPromotion juga menunjukkan 85 outlier, sementara NumCompaniesWorked memiliki 36 outlier. 

Namun, terdapat beberapa variabel yang bersih dari outlier seperti Age, DailyRate, DistanceFromHome, EmployeeNumber, HourlyRate, MonthlyRate, dan PercentSalaryHike. Hasil ini mengindikasikan bahwa sebagian besar outlier terkonsentrasi pada variabel-variabel yang berkaitan dengan pengalaman kerja, kompensasi, dan perkembangan karir karyawan.

| Variabel                     | Jumlah Outlier |
|------------------------------|----------------|
| `clientnum`                  | 0              |
| `customer_age`               | 22             |
| `dependent_count`            | 0              |
| `months_on_book`             | 3864           |
| `total_relationship_count`   | 5              |
| `months_inactive_12_mon`     | 3316           |
| `contacts_count_12_mon`      | 6297           |
| `credit_limit`               | 9848           |
| `total_revolving_bal`        | 0              |
| `avg_open_to_buy`            | 963            |
| `total_amt_chng_q4_q1`       | 3961           |
| `total_trans_amt`            | 896            |
| `total_trans_ct`             | 2              |
| `total_ct_chng_q4_q1`        | 3941           |
| `avg_utilization_ratio`      | 0              |

<figure>
    <center><img src="img/output_1.png" alt="Box-Plot Outlier"></center>
</figure>

Visualisasi melalui **boxplot** semakin memperjelas sebaran data dan keberadaan outlier di setiap fitur. Fitur seperti **TrainingTimeLastYear**, **MonthlyIncome**, dan beberapa fitur lain tampak memiliki sebaran yang lebar dengan banyak data berada di luar whisker (batas IQR), yang mengindikasikan variasi nilai ekstrim dalam data tersebut.

Meskipun demikian, outlier **tidak dihapus** dari dataset. Hal ini dilakukan untuk menjaga **keutuhan informasi**, mengingat data pencilan tersebut mencerminkan kondisi nyata seperti lonjakan transaksi pelanggan. Menghilangkan outlier justru berisiko menghilangkan pola penting dalam konteks analisis churn pelanggan.


### [Exploratory Data Analysis] - Univariate Analysis

#### Grafik 1: Distribusi Fitur Numerik
<figure>
    <center><img src="img/output_3.png" alt="Distribusi Fitur Numerik"></center>
</figure>

- Sebagian besar nasabah berusia antara 40–50 tahun dan telah menggunakan layanan kartu kredit bank selama 36 bulan. Limit kredit nasabah cenderung rendah, dengan distribusi limit yang miring ke kanan — hanya sekitar 5% yang memiliki limit tinggi (sekitar 35.000). Hal serupa juga terjadi pada avg_open_to_buy dan avg_utilization_ratio, yang juga berpola skewed ke kanan.

- Bank perlu memperhatikan distribusi avg_utilization_ratio, karena sekitar 25% nasabah tidak menggunakan layanan sama sekali (rasio pemanfaatan = 0). Ini menjadi peluang bagi bank untuk meningkatkan pemanfaatan layanan agar pendapatan bertambah.

- Sekitar 25% nasabah juga memiliki total revolving balance sebesar nol, yang bisa menjadi indikator potensi churn. Karena proporsi nol pada variabel ini sama dengan utilization ratio, kemungkinan keduanya berkorelasi — ini akan dianalisis lebih lanjut.

- Distribusi jumlah dan nominal transaksi menunjukkan dua puncak, kemungkinan berbeda berdasarkan status churn. Banyak nasabah melakukan sekitar 40–80 transaksi dalam 12 bulan terakhir, dengan nominal umum sebesar 2.500 atau 5.000.

- Sekitar 80% nasabah memiliki setidaknya tiga produk dari bank.

#### Grafik 3: Distribusi Fitur Kategorik
<figure>
    <center><img src="img/output_4.png" alt="Distribusi Fitur Kategorik"></center>
</figure>

- 53% nasabah adalah perempuan, dan sebagian besar memiliki 2–3 tanggungan. Sebanyak 90% tidak aktif selama 1–3 bulan terakhir, sehingga perlu strategi untuk meningkatkan penggunaan layanan.

- Mayoritas nasabah menghubungi bank 2–3 kali setahun, berstatus menikah/lajang, lulusan sarjana, berpenghasilan < $40K, dan memiliki kartu blue.

- Beberapa kategori seperti 0 bulan tidak aktif dan 6 kali kontak sangat jarang muncul. Selain itu, kategori kartu tidak seimbang—93% nasabah memakai kartu blue. Ketidakseimbangan ini perlu diperhatikan saat pemodelan untuk menghindari overfitting.



## Data Preparation

###1. Feature Engineering, Data Cleaning and Preprocessing¶

Preprocessing untuk Model Berbasis Tree
- **Fitur Numerik**: <br/>
    Akan dilakukan transformasi karena untuk menyeragamkan skala dari berbagai fitur sehingga memiliki rentang nilai yang comparable. Tanpa scaling, fitur dengan skala besar (seperti MonthlyIncome) dapat mendominasi model machine learning dibandingkan fitur berskala kecil (seperti Age), yang menyebabkan model menjadi bias.

- **Fitur Kategorikal**. <br/>
    Akan diterapkan categorical encoding pada variabel dengan tipe **object*. Tanpa encoding, model tidak dapat memproses data kategorikal karena algoritma ML hanya bekerja pada data numerik.

**Variabel yang Akan Dihapus**
'JobLevel', 'YearsInCurrentRole', 'YearsWithCurrManager'

1. `JobLevel`, ‘YearsInCurrentRole’, ‘YearsWithCurrManager’: <br/>
    Akan dihapus karena memiliki korelasi positif tinggi, sehingga informasinya menjadi redundan.
2. 'EmployeeCount', 'Over18', 'StandardHours': <br/>
    Akan dihapus karena memiliki nilai yang sama semua setiap recordnya, sehingga tidak memiliki pengaruh untuk analisis.
2. 'id': <br/>
    Akan dihapus karena memiliki nilai yang unik untuk setiap recordnya, sehingga tidak berguna untuk analisis.

---

## Model Training, Comparison

### 1. Model Selection

Pada tahap pengembangan model prediksi **Employee Attrition**, digunakan 3 pendekatan yaitu pendekatan menggunakan **GridSearchCV**, pendekatan gabungan **XGBoost** dan **RandomizedSearchCV**, dan pendekatan yang terakhir adalah **stacking ensemble** yang mengombinasikan tiga model berbasis gradient boosting yaitu **XGBoost**, **LightGBM**, dan **CatBoost**  dengan model meta-learner **Logistic Regression** sebagai estimator akhir.
Ketiga algoritma dipilih karena masing-masing memiliki keunggulan berbeda dalam menangani data tabular dengan fitur numerik maupun kategorikal, serta dikenal memiliki performa tinggi untuk masalah klasifikasi biner.

#### GridSearchCV
```python
from sklearn.model_selection import GridSearchCV

rf_grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=2,
    n_jobs=-1
)
```
[GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?utm_source) adalah metode dalam scikit-learn yang digunakan untuk melakukan pencarian parameter secara menyeluruh (exhaustive search) pada model pembelajaran mesin. Metode ini menguji semua kombinasi parameter yang ditentukan dalam `param_grid` dan mengevaluasi setiap kombinasi menggunakan cross-validation untuk menentukan kombinasi parameter yang memberikan performa terbaik. Hasilnya adalah model dengan hyperparameter yang dioptimalkan, yang dapat digunakan untuk prediksi lebih lanjut. GridSearchCV juga mendukung evaluasi dengan beberapa metrik dan dapat digunakan bersama dengan pipeline untuk mengoptimalkan parameter pada tahap transformasi dan klasifikasi secara bersamaan. 

#### RandomizedSearchCV
```python
from sklearn.model_selection import RandomizedSearchCV

xgb_random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=30,
    scoring='roc_auc',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
```
[RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) adalah metode di scikit-learn untuk mencari hyperparameter terbaik pada model machine learning dengan cara sampling acak dari ruang parameter yang ditentukan, bukan mengecek semua kombinasi seperti GridSearchCV. Metode ini menggunakan cross-validation untuk menilai performa setiap kombinasi parameter yang dipilih secara acak sehingga proses tuning lebih cepat dan efisien, terutama jika jumlah parameter besar atau ruang parameter sangat luas. RandomizedSearchCV cocok ketika waktu komputasi terbatas atau ketika beberapa parameter memiliki pengaruh kecil terhadap performa model.


#### XGBoost
```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=2,
    reg_lambda=1.2,
    reg_alpha=0.2,
    eval_metric="auc",
    random_state=42
)

```
[XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html) adalah algoritma boosting berbasis pohon yang dibangun untuk efisiensi dan performa tinggi.Model ini mengoptimalkan fungsi kehilangan secara bertahap dengan menambahkan pohon baru yang memperbaiki kesalahan dari pohon sebelumnya. Penggunaan parameter seperti scale_pos_weight dan reg_lambda membantu menangani ketidakseimbangan kelas serta mencegah overfitting. XGBoost dikenal unggul dalam kestabilan prediksi dan kecepatan pelatihan.

#### LightGBM

````python
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=40,
    subsample=0.85,
    colsample_bytree=0.85,
    class_weight='balanced',
    random_state=42,
    metric='auc'
)

````
[LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) merupakan algoritma gradient boosting yang dikembangkan oleh Microsoft, dengan keunggulan pada efisiensi memori dan waktu pelatihan. LightGBM menggunakan pendekatan leaf-wise tree growth, yang memperluas cabang dengan loss reduction tertinggi, menghasilkan model yang lebih akurat pada jumlah pohon yang sama. Parameter class_weight='balanced' membantu mengoreksi bias kelas minoritas, sedangkan num_leaves mengontrol kompleksitas model untuk menjaga keseimbangan antara bias dan varians.

#### CatBoost (Categorical Boosting)


````python
from catboost import CatBoostClassifier

cat = CatBoostClassifier(
    iterations=800,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=2,
    eval_metric='AUC',
    verbose=0,
    random_seed=42
)

````
[CatBoost](https://catboost.ai/docs/en/) adalah algoritma gradient boosting yang dirancang khusus untuk menangani fitur kategorikal tanpa perlu encoding manual.
Dengan teknik ordered boosting dan target statistics, CatBoost mampu mengurangi overfitting serta meningkatkan stabilitas prediksi. Model ini bekerja sangat baik pada data tabular dengan kombinasi fitur numerik dan kategorikal, serta memiliki interpretabilitas yang lebih tinggi dibanding metode boosting lain.

Ketiga model ini digunakan dengan pengaturan parameter awal sebagai percobaan dasar

- Pada langkah ini, membandingkan kinerja model yang berbeda dengan menggunakan **_stratified k-fold cross validation_** untuk melatih masing-masing model dan mengevaluasi skor ROC-AUC. Stratified k-fold cross validation akan mempertahankan proporsi target pada setiap fold, menangani target yang tidak seimbang.

- _k-fold cross validation_ adalah teknik yang digunakan dalam _machine learning_ untuk menilai kinerja model. Teknik ini melibatkan pembagian dataset menjadi K subset, menggunakan K-1 untuk pelatihan dan satu untuk pengujian secara berulang. Hal ini membantu dalam memperkirakan kemampuan generalisasi model dengan mengurangi risiko _overfitting_ dan memberikan metrik kinerja yang lebih andal.

- Tujuan tahap ini adalah untuk memilih model terbaik untuk digunakan dalam _feature selection_, _hyperparameter tuning_, dan evaluasi model akhir. Untuk mendapatkan model terbaik ini, akan dievaluasi skor validasi rata-rata **roc-auc** tertinggi dan melihat trade-off bias-varians.

#### Tabel Perbandingan Performa Model

| Metrik | RandomForest | XGBoost | Stacking Ensemble |
|--------|--------------|---------|------------------|
| ROC AUC (Val) | 0.793493 | 0.808087 | 0.948683 |
| Akurasi (Val) | 0.858844 | 0.857143 | 0.910074 |
| Recall (Val) | 0.157895 | 0.315789 | 0.805274 |

<figure>
    <center><img src="img/output_9.png" alt="Hasil Performa Model"></center>
</figure>

Berdasarkan hasil evaluasi pada data validasi, **Stacking Ensemble** dipilih sebagai model utama karena menunjukkan performa terbaik dibandingkan RandomForest dan XGBoost tunggal. Hal ini terlihat dari skor **ROC-AUC validasi tertinggi (0,9487)**, yang mengindikasikan kemampuan model dalam membedakan kelas secara akurat antara nasabah yang churn dan tidak.
Meskipun RandomForest dan XGBoost memiliki skor akurasi yang cukup tinggi (masing-masing 0,8588 dan 0,8571), nilai **recall mereka relatif rendah (0,158 dan 0,316)**. Ini menandakan bahwa kedua model tunggal ini kurang sensitif dalam mendeteksi nasabah yang churn. Sebaliknya, Stacking Ensemble berhasil meningkatkan recall hingga 0,8053, sehingga lebih mampu menangkap kasus churn, sambil tetap mempertahankan spesifisitas yang tinggi (0,97), sehingga jumlah false positive tetap rendah.
Skor akurasi validasi Stacking Ensemble (0,9101) dan spesifisitas yang tinggi menunjukkan bahwa model ini tidak hanya unggul dalam mendeteksi churn, tetapi juga tetap akurat dalam mengklasifikasikan nasabah non-churn, sehingga keseimbangan antara sensitifitas dan spesifisitas terjaga dengan baik.
Performa ini mencerminkan kualitas data yang baik dan efektivitas pendekatan ensemble, di mana kombinasi beberapa model (XGBoost, LGBM, dan CatBoost) mampu menangkap pola yang tidak tertangkap oleh model tunggal. Meskipun potensi peningkatan performa lebih lanjut melalui hyperparameter tuning relatif kecil, langkah tersebut tetap direkomendasikan untuk menyempurnakan model.



### 3. Hyperparameter Tuning

- Dilakukan _hyperparameter tuning_ LightGBM menggunakan **Bayesian optimization** melalui library **optuna**.  

- Bayesian optimization melakukan pencarian cerdas dalam ruang hyperparameter model, dengan menyeimbangkan trade-off antara eksplorasi dan eksploitasi.

- **Grid search** tidak efisien karena melakukan pencarian secara menyeluruh dengan menguji semua kombinasi parameter yang mungkin tanpa mempertimbangkan efek interaksi antar-parameter. Hal ini akan merugikan dalam kasus ini, karena dataset pelatihan cukup besar, dan parameter seperti jumlah pohon (number of trees) dan laju pembelajaran (learning rate) saling terkait dengan parameter lainnya.

- Pada setiap iterasi, akan dilatih model menggunakan parameter yang diuji pada dataset train dan mengevaluasinya dengan **stratified k-fold cross-validation** untuk menghindari overfitting akibat _hyperparameter tuning_ yang berlebihan pada dataset train.  

- Penting untuk mendefinisikan hyperparameter **scale_pos_weight/class_weight** untuk menangani ketidakseimbangan data. Hyperparameter ini memungkinkan model untuk lebih baik mempelajari pola dari kelas minoritas (absent (1)) dengan memberikan bobot lebih besar pada instance tersebut. Bobot ini akan meningkatkan biaya log-loss ketika salah mengklasifikasikan kelas ini, sehingga menghasilkan pembelajaran yang lebih baik pada kelas tersebut.  

- Tuning hyperparameter adalah langkah penyempurnaan. Yang secara signifikan meningkatkan performa model adalah langkah rekayasa fitur (feature engineering). Seperti yang sudah disebutkan sebelumnya, mengingat performa luar biasa model kita pada data berkualitas tinggi.

#### Best Parameter

```python
final_best_params = {
    'objective': 'binary',
    'metric': 'roc_auc',
    'n_estimators': 1000,
    'verbosity': -1,
    'bagging_freq': 1,
    'class_weight': 'balanced', 
    'learning_rate': best_params['learning_rate'],
    'num_leaves': best_params['num_leaves'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'min_data_in_leaf': best_params['min_data_in_leaf']
}

lgb_clf = LGBMClassifier(**final_best_params)
lgb_clf.fit(X_train_selected, y_train)
```
---

## Model Testing and Evaluation

### 1. Data Test Predict

Dilakukan transform untuk preprocessing dan feature selection pada data test. Kemudian, dilakukan predict dan mendapatkan estimasi probabilitas churn dari model untuk evaluasi.

### 2. Best Model Evaluation

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.98      | 0.98   | 0.98     | 1701    |
| 1     | 0.90      | 0.88   | 0.89     | 325     |

- **Accuracy**: 0.96 (2026 data)
- **Macro Average**:
  - Precision: 0.94
  - Recall: 0.93
  - F1-Score: 0.93
- **Weighted Average**:
  - Precision: 0.96
  - Recall: 0.96
  - F1-Score: 0.96

#### Metode Evaluasi Lanjutan

- **Brier Score**: 0.03  
- **Gini Coefficient**: 0.98  
- **Kolmogorov-Smirnov (KS) Statistic**: 0.89

#### Confusion Matrix
<figure>
    <center><img src="img/output_10.png" alt="Confusion Matrix"></center>
</figure>

#### Plot ROC-AUC Curve
<figure>
    <center><img src="img/output_11.png" alt="ROC-AUC Curve"></center>
</figure>

#### Plot PR-AUC Curve
<figure>
    <center><img src="img/output_12.png" alt="PR-AUC Curve"></center>
</figure>

| No | Metrik     | Nilai     |
|----|------------|-----------|
| 0  | Accuracy   | 0.964956  |
| 1  | Precision  | 0.901899  |
| 2  | Recall     | 0.876923  |
| 3  | F1-Score   | 0.889236  |
| 4  | ROC-AUC    | 0.990192  |
| 5  | KS         | 0.894844  |
| 6  | Gini       | 0.980384  |
| 7  | PR-AUC     | 0.960791  |
| 8  | Brier      | 0.028807  |

Hasil akhir menunjukkan bahwa model memiliki performa yang sangat baik

- **Recall (0,89):** Model mampu mengidentifikasi 89% pelanggan yang berhenti berlangganan (*churners*). Berdasarkan confusion matrix, sebanyak **290 dari 325** churners berhasil diprediksi dengan benar.
  
- **Precision (0,90):** Dari seluruh pelanggan yang diprediksi akan churn, **90%** di antaranya benar-benar churners. Dalam confusion matrix, dari **324** prediksi churn, **290** adalah churners sesungguhnya.

- **ROC-AUC (0,99):** Skor ROC-AUC sebesar 0,99 mengindikasikan kemampuan luar biasa model dalam membedakan antara pelanggan yang churn dan yang tidak. Secara praktis, apabila dipilih satu pelanggan churn dan satu non-churn secara acak, model akan memberikan probabilitas churn yang lebih tinggi kepada pelanggan yang benar-benar churn dalam **99% kasus**.

Performa tinggi ini **bukan berasal dari kebocoran data**, karena semua fitur yang digunakan tersedia saat prediksi dilakukan, dan data dibagi sebelum proses pemodelan. Hasil ini mencerminkan **kualitas data yang sangat baik**, di mana variabel-variabel independen mampu secara jelas membedakan antara churners dan non-churners.

Selain itu, **kesamaan skor antara data train, test, dan validasi** menunjukkan bahwa model memiliki **kemampuan generalisasi yang andal**.

**Langkah Selanjutnya:**

Melakukan analisis terhadap distribusi probabilitas churn yang diprediksi oleh model.

<figure>
    <center><img src="img/output_13.png" alt="Probability Plot"></center>
</figure>

Dapat dilihat bahwa ada pemisahan yang jelas antara distribusi probabilitas yang diprediksi untuk churner dan non-churner.

<figure>
    <center><img src="img/output_14.png" alt="Probability Score Ordering"></center>
</figure>

Urutan skor probabilitas juga terlihat bagus. Semua churner berada di antara desil ke-7 dan ke-10.

---

## Save Best Model
```python
filename = '../model/LightGBM_model__v1.pkl'
```

---

## Model Interpretation

#### 1. Interpretation with SHAP Values

- Untuk menginterpretasikan hasil LightGBM, akan dianalisis nilai **SHAP**.  

- SHAP adalah library yang memungkinkan interpretasi hasil algoritma machine learning. Dengan SHAP, dapat dipahami dampak masing-masing fitur terhadap prediksi model individu, di mana **$ f(x) = E(f(x)) + SHAP $**.  

- Secara sederhana, nilai SHAP dari sebuah fitur (seberapa besar pengaruhnya terhadap prediksi individu) adalah penjumlahan berbobot kontribusi marjinal dengan mempertimbangkan semua kemungkinan kombinasi fitur (**feature coalitions**).  

- **Feature coalition** adalah kelompok fitur, dan nilainya merupakan prediksi model individu yang hanya menggunakan fitur-fitur dalam kelompok tersebut. Kontribusi marjinal dari sebuah fitur adalah perbedaan antara nilai prediksi untuk kombinasi fitur dengan dan tanpa fitur tersebut. Nilai kontribusi marjinal dijumlahkan untuk semua kemungkinan kombinasi dengan dan tanpa fitur tersebut. Bobotnya didasarkan pada probabilitas fitur yang sedang dihitung nilai SHAP-nya untuk berada dalam kombinasi tersebut.

```python
import shap

explainer = shap.Explainer(lgb_clf)
shap_values = explainer(X_test_selected)
```

- Terdapat **25 variabel** dalam model, dan untuk setiap observasi, masing-masing memiliki **nilai SHAP** yang menggambarkan kontribusinya terhadap prediksi.

- Dalam kasus klasifikasi biner, hasil prediksi dinyatakan dalam bentuk **log-odds**. Pada visualisasi berikut, $E(f(X))$ merepresentasikan **nilai rata-rata prediksi dalam skala log-odds**.

- **Log-odds** sendiri merupakan logaritma dari *odds*, yaitu rasio antara probabilitas suatu kejadian terjadi dengan tidak terjadi. Penggunaan logaritma ini menjadikan skala prediksi lebih linear dan stabil.

- **Nilai SHAP positif** menunjukkan bahwa suatu fitur meningkatkan nilai log-odds, yang berarti juga meningkatkan **kemungkinan pelanggan untuk churn**, sedangkan **nilai negatif** menurunkan probabilitas tersebut.

- Untuk mengubah log-odds menjadi **probabilitas churn**, digunakan fungsi logistik (sigmoid) sebagai berikut:

<center>

$$
\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}
$$

</center>

- Visualisasi **waterfall plot** digunakan untuk menunjukkan **kontribusi masing-masing fitur terhadap prediksi churn** baik untuk observasi positif (churner) maupun negatif (non-churner).

#### Sampel salah satu pelanggan
<figure>
    <center><img src="img/output_15.png" alt="Probability Sample Pelanggan"></center>
</figure>

- Pelanggan ini diprediksi **tidak churn**, dengan **probabilitas churn sangat mendekati nol** berdasarkan hasil transformasi log-odds melalui fungsi logistik.

- Salah satu faktor utama adalah **jumlah transaksi** dalam 12 bulan terakhir yang mencapai **105 transaksi**, yang **menurunkan log-odds churn sebesar 3,07**. Artinya, semakin sering pelanggan bertransaksi, semakin kecil kemungkinan ia untuk churn.

- Sebaliknya, fitur **total revolving balance** yang bernilai **nol** justru **meningkatkan log-odds churn sebesar 1,41**. Ini sejalan dengan temuan sebelumnya bahwa banyak pelanggan churn memiliki nilai nol pada fitur ini maupun pada **average utilization ratio**.

- Selain itu, fitur hasil dari proses **feature engineering**, seperti `proportion_inactive_months` dan `trans_ct_per_inactivity`, juga memberikan pengaruh signifikan terhadap log-odds, baik yang mengarah pada penurunan maupun peningkatan risiko churn.

### 2. Feature Importance
<figure>
    <center><img src="img/output_16.png" alt="Feature Importance"></center>
</figure>

- Seperti yang diharapkan, jumlah transaksi dan jumlah total transaksi dalam 12 bulan terakhir merupakan fitur yang paling penting. Hal ini sangat masuk akal, dan seperti yang telah kita lihat pada analisis eksplorasi data (EDA), variabel-variabel ini menunjukkan perbedaan yang jelas antara churners dan non-churners.  

- Selain itu, fitur-fitur yang kita buat pada tahap **feature engineering** juga termasuk dalam daftar fitur paling penting, yang menggambarkan betapa pentingnya langkah ini dalam meningkatkan performa model pembelajaran mesin. Contohnya adalah **trans_ct_per_inactivity** dan **avg_trans_amt**.  

- Sekarang, melalui **beeswarm plot**, kita dapat mengamati hubungan antara fitur-fitur dan prediksi model.

<figure>
    <center><img src="img/output_17.png" alt="SHAP Value Impact"></center>
</figure>

- Terlihat bahwa nilai rendah pada jumlah transaksi dalam 12 bulan terakhir memiliki dampak positif terhadap log-odds churn, dan, akibatnya, terhadap probabilitas churn, sedangkan nilai tinggi memiliki dampak negatif.

- Secara mengejutkan, nilai tinggi pada rata-rata jumlah transaksi justru cenderung memberikan dampak positif terhadap log-odds churn, dan, akibatnya, probabilitas churn, sedangkan nilai rendah cenderung memiliki dampak negatif.

---

## Financial Result

**Estimasi Dampak Finansial Model terhadap Bank**

Untuk menunjukkan nilai tambah dari analisis ini, akan disajikan performa model dalam bentuk estimasi keuntungan finansial bagi pihak bank. Analisis ini didasarkan pada **confusion matrix** dan data yang tersedia saat ini.

**Asumsi Dasar:**
Karena tidak tersedia data spesifik mengenai keuntungan aktual, digunakan asumsi umum bahwa salah satu sumber pendapatan utama bank dari kartu kredit berasal dari **biaya atas saldo kredit yang belum dibayar (total revolving balance)**. Rata-rata biaya tersebut di pasar adalah **18% per tahun**.

**Komponen Biaya dan Manfaat yang Diperhitungkan:**

- **Biaya Retensi untuk False Positive (FP):**  
  Pelanggan yang salah diprediksi akan churn namun sebenarnya tidak. Bank akan mengeluarkan biaya retensi yang tidak perlu.  
  *Asumsi:* Bank memberikan diskon biaya dari 18% menjadi 10%, sehingga kehilangan pendapatan sebesar **8%** dari total revolving balance.

- **Kehilangan Pendapatan dari False Negative (FN):**  
  Pelanggan yang benar-benar churn namun gagal terdeteksi oleh model. Bank kehilangan seluruh pendapatan sebesar **18%** dari saldo mereka.

- **Pendapatan dari True Positive (TP):**  
  Pelanggan yang diprediksi churn dan berhasil dipertahankan. Bank tetap memperoleh pendapatan sebesar **10%** dari saldo mereka melalui strategi retensi.

**Langkah Selanjutnya:**
Dilakukan perhitungan proyeksi keuntungan/kerugian berdasarkan nilai-nilai di atas menggunakan **dataset aktual untuk hasil finansial**, dengan mempertimbangkan jumlah pelanggan pada setiap kategori confusion matrix (`TP`, `FP`, `FN`) dan total revolving balance mereka.

Model menghasilkan estimasi hasil finansial sekitar $175,587. Jumlah sebenarnya akan bergantung pada kebijakan manajemen bank saat mengimplementasikan strategi retensi untuk pelanggan berdasarkan probabilitas churn yang diprediksi.

Sebagai contoh, jika bank ingin bersikap lebih konservatif dengan mengurangi pengeluaran yang terkait dengan **false positive**, bank dapat menargetkan pelanggan dengan probabilitas churn yang lebih tinggi, sehingga memengaruhi potensi keuntungan.  

Namun demikian, untuk tujuan estimasi dan sebagai dasar pengambilan keputusan, kita telah memastikan bahwa proyek ini sangat layak untuk dilakukan.  

## Conclusions

### Ringkasan Proyek
Dalam proyek ini, telah dikembangkan sebuah model klasifikasi berbasis LightGBM untuk memprediksi probabilitas pelanggan yang akan melakukan churn pada layanan kartu kredit sebuah bank. Tujuan utama dari pengembangan model ini adalah untuk menghasilkan prediksi yang akurat terhadap potensi churn, mengidentifikasi faktor-faktor utama yang memengaruhi keputusan pelanggan untuk berhenti menggunakan layanan, serta menyusun rekomendasi aksi yang dapat diimplementasikan guna meminimalkan tingkat churn. Dengan demikian, pihak bank dapat menyusun strategi retensi yang lebih efektif, mengingat bahwa mempertahankan pelanggan yang sudah ada umumnya lebih ekonomis dibandingkan dengan mengakuisisi pelanggan baru.

### Hasil dan Evaluasi Model
Permasalahan bisnis yang diangkat telah berhasil diselesaikan dengan baik. Model yang dibangun mampu mengidentifikasi 89% pelanggan yang churn dengan tingkat akurasi AUC sebesar 0,99. Nilai AUC yang tinggi ini menunjukkan bahwa model memiliki kemampuan klasifikasi yang sangat baik, yakni dalam 99% kasus, model memberikan probabilitas churn yang lebih tinggi kepada pelanggan yang benar-benar churn dibandingkan yang tidak.

Selama tahap _Exploratory Data Analysis (EDA)_, telah berhasil diidentifikasi beberapa faktor utama yang menjadi penyebab churn, dan temuan ini kemudian digunakan untuk memberikan rekomendasi awal kepada pihak bank terkait pola attrisi pelanggan.

### Penanganan Ketidakseimbangan Data
Dalam menghadapi ketidakseimbangan pada variabel target, telah diterapkan beberapa strategi, antara lain: Stratified hold-out split, k-fold cross-validation, serta penggunaan hyperparameter _class_weight_. Pendekatan SMOTE secara eksplisit tidak digunakan karena dinilai kurang merepresentasikan kondisi realistis dalam dunia industri. Pendekatan yang digunakan difokuskan pada simulasi solusi data science yang praktis dan realistis.

### Interpretasi dan Validasi Model
Model telah diinterpretasikan dengan menggunakan teknik SHAP (SHapley Additive exPlanations) untuk mengevaluasi kontribusi setiap fitur terhadap prediksi model, baik secara global maupun individual. Hasil interpretasi ini selaras dengan temuan selama tahap EDA, yang mengindikasikan bahwa fitur-fitur yang paling berpengaruh memang sudah teridentifikasi sebelumnya. Selain itu, nilai probabilitas yang dihasilkan oleh model dinilai masuk akal dan konsisten, memperkuat kepercayaan terhadap keandalan model.

### Estimasi Nilai Finansial
Berdasarkan estimasi awal, proyek ini memiliki potensi memberikan dampak finansial sebesar $171.477. Besarnya nilai manfaat aktual tentu akan sangat tergantung pada struktur biaya yang ditetapkan bank serta sejauh mana strategi retensi berbasis model ini diimplementasikan oleh manajemen. Meskipun demikian, estimasi ini memberikan dasar yang kuat untuk pengambilan keputusan bisnis.

### Langkah Selanjutnya
Tahap selanjutnya dari proyek ini adalah deploy model ke dalam lingkungan produksi dengan menerapkan prinsip _Continuous Integration/Continuous Deployment_ (CI/CD). Langkah ini bertujuan untuk memastikan proses otomatisasi yang berkelanjutan serta pemeliharaan model yang efisien dan dapat diandalkan di lingkungan operasional.


