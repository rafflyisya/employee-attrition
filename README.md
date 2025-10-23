# Laporan Proyek Machine Learning  
## Credit Card Users Churn Prediction  
**Muhammad Tsaqif** — *MC004D5Y2062*

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
    <center><img src="img/credit-card-terminal-payment.jpg" alt="Credit Card"></center>
</figure>

Penggunaan kartu kredit merupakan salah satu layanan keuangan yang memberikan kontribusi signifikan terhadap pendapatan sebuah bank. Melalui berbagai jenis biaya seperti biaya tahunan, biaya keterlambatan pembayaran, biaya penarikan tunai, serta biaya transaksi internasional, kartu kredit menjadi salah satu pilar utama dalam model bisnis perbankan ritel [[1]](https://www.bostonfed.org/publications/research-department-working-paper/2010/the-2009-survey-of-consumer-payment-choice.aspx). Namun, belakangan ini, terjadi peningkatan jumlah pelanggan yang menghentikan penggunaan layanan kartu kredit, yang dikenal sebagai fenomena *credit card churn*. Hal ini menimbulkan kekhawatiran bagi manajemen bank karena dapat berdampak langsung terhadap pendapatan dan stabilitas bisnis jangka panjang [[2]](https://www2.deloitte.com/us/en/insights/industry/financial-services/credit-card-customer-churn.html).

Fenomena churn dapat dipengaruhi oleh berbagai faktor, seperti ketidakpuasan pelanggan terhadap layanan, persaingan dari institusi keuangan lain, atau perubahan dalam perilaku keuangan individu [[3]](https://doi.org/10.5120/ijca2017914142). Oleh karena itu, penting bagi institusi keuangan untuk tidak hanya memahami siapa saja pelanggan yang berisiko tinggi melakukan churn, tetapi juga mengidentifikasi alasan-alasan yang mendasarinya. Dengan pendekatan yang tepat, bank dapat mengambil langkah preventif, seperti meningkatkan layanan pelanggan, menawarkan program loyalitas, atau menyesuaikan produk agar lebih sesuai dengan kebutuhan pengguna.

Proyek ini berada dalam domain analisis perilaku pelanggan dan manajemen risiko keuangan, dengan fokus pada pengembangan model prediksi berbasis data untuk mengidentifikasi pelanggan yang berpotensi meninggalkan layanan kartu kredit. Dengan memanfaatkan teknik data science dan machine learning, bank dapat memprediksi perilaku churn secara lebih akurat dan melakukan intervensi yang bersifat proaktif untuk mempertahankan pelanggan dan meminimalisir kerugian finansial yang mungkin timbul.

---

**Referensi:**

[1] Federal Reserve Bank of Boston (2010). *The 2009 Survey of Consumer Payment Choice*. Retrieved from [https://www.bostonfed.org](https://www.bostonfed.org/publications/research-department-working-paper/2010/the-2009-survey-of-consumer-payment-choice.aspx)  
[2] Deloitte (2018). *The changing face of credit card churn*. Retrieved from [https://www2.deloitte.com](https://www2.deloitte.com/us/en/insights/industry/financial-services/credit-card-customer-churn.html)  
[3] Zhang, J., & Feng, X. (2017). *Customer Churn Prediction in Credit Card Industry Using Data Mining Techniques*. International Journal of Computer Applications, 166(1), 1–6. [https://doi.org/10.5120/ijca2017914142](https://doi.org/10.5120/ijca2017914142)

---

## Business Understanding

### Problem Statements  
Dalam industri perbankan, layanan kartu kredit merupakan salah satu sumber utama pendapatan. Namun, bank menghadapi tantangan serius karena meningkatnya jumlah pelanggan yang berhenti menggunakan layanan ini (*credit card churn*). Tingginya angka churn tidak hanya berdampak pada hilangnya pendapatan langsung, tetapi juga meningkatkan beban biaya untuk akuisisi pelanggan baru.

Berdasarkan hal tersebut, berikut adalah pernyataan masalah yang diangkat:

- **Pernyataan Masalah 1:** Bagaimana mengidentifikasi faktor-faktor penting yang memengaruhi keputusan pelanggan untuk berhenti menggunakan layanan kartu kredit?
- **Pernyataan Masalah 2:** Bagaimana membangun model prediksi yang mampu memperkirakan kemungkinan seorang pelanggan akan melakukan churn dengan tingkat akurasi tinggi?  
- **Pernyataan Masalah 3:** Bagaimana menyusun strategi berbasis data untuk menurunkan tingkat churn, serta meningkatkan retensi dan pengalaman pelanggan?

### Goals  
Untuk menjawab pernyataan masalah di atas, tujuan proyek ini dirumuskan sebagai berikut:

- **Tujuan 1:** Melakukan eksplorasi dan analisis data historis pelanggan untuk mengidentifikasi pola dan fitur yang berkorelasi tinggi terhadap perilaku churn.  
- **Tujuan 2:** Membangun model prediktif berbasis machine learning yang mampu menghitung probabilitas churn dari masing-masing pelanggan.  
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

- **Penghematan Biaya:** Mengurangi kebutuhan akuisisi pelanggan baru dengan mempertahankan pelanggan lama.  
- **Peningkatan Retensi Pelanggan:** Menyasar pelanggan berisiko tinggi dengan pendekatan personalisasi.  
- **Peningkatan Pengalaman Pelanggan:** Mengidentifikasi titik-titik frustasi dalam layanan dan melakukan perbaikan.  
- **Pemasaran yang Lebih Efisien:** Mengalokasikan sumber daya pemasaran pada segmen pelanggan yang paling membutuhkan.  
- **Perlindungan Pendapatan:** Menjaga stabilitas pendapatan dari segmen kartu kredit jangka panjang.

---

## Data Understanding

### Sumber Data  
Dataset yang digunakan dalam proyek ini diperoleh dari situs [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/data). Dataset ini mencakup informasi tentang **10.127 pelanggan** layanan kartu kredit, yang mencatat berbagai aspek demografis dan perilaku transaksi pelanggan.

Dataset ini memiliki **21 fitur**, yang mencakup usia, gaji, status pernikahan, limit kartu kredit, jenis kartu, aktivitas akun, dan lainnya. Di antara seluruh pelanggan, hanya sekitar **16,07%** yang termasuk dalam kategori churn (berhenti menggunakan layanan). Ketidakseimbangan kelas ini menjadikan proses pelatihan model prediktif sebagai tantangan tersendiri.

### Deskripsi Fitur

| Nama Fitur                         | Deskripsi                                                                 | Tipe Data    |
|------------------------------------|---------------------------------------------------------------------------|--------------|
| `CLIENTNUM`                        | ID unik pelanggan                                                         | `int64`      |
| `Attrition_Flag` => `churn_flag`   | Status churn: `Attrited Customer` atau `Existing Customer`                | `object`     |
| `customer_age`                     | Usia pelanggan (dalam tahun)                                              | `int64`      |
| `gender`                           | Jenis kelamin pelanggan (`M`/`F`)                                         | `object`     |
| `dependent_count`                  | Jumlah tanggungan                                                         | `int64`      |
| `education_level`                  | Tingkat pendidikan: Graduate, High School, College, dll.                  | `object`     |
| `marital_status`                   | Status pernikahan pelanggan                                               | `object`     |
| `income_category`                  | Kategori pendapatan tahunan pelanggan                                     | `object`     |
| `card_category`                    | Jenis kartu kredit                                                        | `object`     |
| `months_on_book`                   | Lama hubungan dengan bank (dalam bulan)                                   | `int64`      |
| `total_relationship_count`         | Jumlah produk perbankan yang dimiliki                                     | `int64`      |
| `months_inactive_12_mon`           | Jumlah bulan tidak aktif dalam 12 bulan terakhir                          | `int64`      |
| `contacts_count_12_mon`            | Jumlah interaksi dengan bank dalam 12 bulan terakhir                      | `int64`      |
| `credit_limit`                     | Batas maksimal kredit kartu pelanggan                                     | `float64`    |
| `total_revolving_bal`              | Saldo bergulir (tidak dibayar penuh)                                      | `int64`      |
| `avg_open_to_buy`                  | Rata-rata jumlah kredit yang tersedia untuk digunakan                     | `float64`    |
| `total_amt_chng_q4_q1`             | Perubahan jumlah transaksi dari Q1 ke Q4                                  | `float64`    |
| `total_trans_amt`                  | Total nilai transaksi selama 12 bulan terakhir                            | `int64`      |
| `total_trans_ct`                   | Total jumlah transaksi selama 12 bulan terakhir                           | `int64`      |
| `total_ct_chng_q4_q1`              | Perubahan jumlah transaksi dari Q1 ke Q4                                  | `float64`    |
| `avg_utilization_ratio`            | Rata-rata rasio pemanfaatan limit kredit                                  | `float64`    |

### Penjelasan Kontekstual Fitur

- **Revolving Balance**  
  Saldo bergulir adalah jumlah saldo yang tidak dibayar penuh setiap bulan dan dibawa ke bulan berikutnya. Kolom `Total_Revolving_Bal` merepresentasikan nilai ini.

- **Average Open to Buy**  
  `Avg_Open_To_Buy` adalah rata-rata jumlah kredit yang masih tersedia untuk digunakan selama 12 bulan terakhir.

- **Average Utilization Ratio**  
  `Avg_Utilization_Ratio` menunjukkan persentase dari limit kredit yang digunakan oleh pelanggan. Rasio ini berkaitan erat dengan skor kredit dan stabilitas finansial pelanggan.

- **Hubungan antara `Avg_Open_To_Buy`, `Credit_Limit`, dan `Avg_Utilization_Ratio`**  
  Nilai-nilai ini secara matematis berkaitan sebagai berikut:
  $$
    \frac{Avg \; Open \; To \; Buy}{Credit \; Limit} + Avg \; Utilization \; Ratio = 1
  $$

### [Exploratory Data Analysis] - Deskripsi Variabel

| Fitur                       | Count   | Mean         | Std          | Min       | 25%          | 50%          | 75%          | Max          |
|-----------------------------|---------|--------------|--------------|-----------|--------------|--------------|--------------|--------------|
| `clientnum`                 | 10127   | 7.391776e+08 | 3.690378e+07 | 708082083 | 7.130368e+08 | 7.179264e+08 | 7.731435e+08 | 8.283431e+08 |
| `customer_age`              | 10127   | 46.32596     | 8.016814     | 26.0      | 41.0         | 46.0         | 52.0         | 73.0         |
| `dependent_count`           | 10127   | 2.346203     | 1.298908     | 0.0       | 1.0          | 2.0          | 3.0          | 5.0          |
| `months_on_book`            | 10127   | 35.92841     | 7.986416     | 13.0      | 31.0         | 36.0         | 40.0         | 56.0         |
| `total_relationship_count`  | 10127   | 3.812580     | 1.554408     | 1.0       | 3.0          | 4.0          | 5.0          | 6.0          |
| `months_inactive_12_mon`    | 10127   | 2.341167     | 1.010622     | 0.0       | 2.0          | 2.0          | 3.0          | 6.0          |
| `contacts_count_12_mon`     | 10127   | 2.455317     | 1.106225     | 0.0       | 2.0          | 2.0          | 3.0          | 6.0          |
| `credit_limit`              | 10127   | 8631.954     | 9088.777     | 1438.3    | 2555.0       | 4549.0       | 11067.5      | 34516.0      |
| `total_revolving_bal`       | 10127   | 1162.814     | 814.9873     | 0.0       | 359.0        | 1276.0       | 1784.0       | 2517.0       |
| `avg_open_to_buy`           | 10127   | 7469.140     | 9090.685     | 3.0       | 1324.5       | 3474.0       | 9859.0       | 34516.0      |
| `total_amt_chng_q4_q1`      | 10127   | 0.759941     | 0.219207     | 0.0       | 0.631        | 0.736        | 0.859        | 3.397        |
| `total_trans_amt`           | 10127   | 4404.086     | 3397.129     | 510.0     | 2155.5       | 3899.0       | 4741.0       | 18484.0      |
| `total_trans_ct`            | 10127   | 64.85869     | 23.47257     | 10.0      | 45.0         | 67.0         | 81.0         | 139.0        |
| `total_ct_chng_q4_q1`       | 10127   | 0.712222     | 0.238086     | 0.0       | 0.582        | 0.702        | 0.818        | 3.714        |
| `avg_utilization_ratio`     | 10127   | 0.274894     | 0.275692     | 0.0       | 0.023        | 0.176        | 0.503        | 0.999        |
     
Dataset ini mencerminkan profil **10.127 pelanggan bank** dengan **usia rata-rata 46 tahun** dan sebagian besar memiliki **2–3 tanggungan**. Sebagian besar telah menjadi nasabah selama sekitar **3 tahun** dan memiliki **3–5 produk perbankan**. Aktivitas transaksi bervariasi, dengan rata-rata **65 transaksi** dan nilai transaksi tahunan sekitar **4.400**. Meskipun ***sebagian besar hanya tidak aktif selama 2 bulan dalam setahun**, ada juga yang hingga **6 bulan tidak aktif**. Pelanggan menggunakan sekitar **27% dari batas kredit mereka**, namun ada yang hampir mencapai **100%**, menunjukkan potensi risiko kredit. Perubahan transaksi antar kuartal menunjukkan adanya penurunan aktivitas secara rata-rata. Temuan ini menunjukkan adanya variasi perilaku nasabah yang dapat dimanfaatkan untuk prediksi churn pelanggan.

### Rata-rata Fitur per Kategori Churn Flag

| Fitur                         | Attrited Customer | Existing Customer |
|-------------------------------|-------------------|-------------------|
| `clientnum`                   | 7.352614e+08      | 7.399272e+08      |
| `customer_age`                | 46.66             | 46.26             |
| `dependent_count`             | 2.40              | 2.34              |
| `months_on_book`              | 36.18             | 35.88             |
| `total_relationship_count`    | 3.28              | 3.91              |
| `months_inactive_12_mon`      | 2.69              | 2.27              |
| `contacts_count_12_mon`       | 2.97              | 2.36              |
| `credit_limit`                | 8136.04           | 8726.88           |
| `total_revolving_bal`         | 672.82            | 1256.60           |
| `avg_open_to_buy`             | 7463.22           | 7470.27           |
| `total_amt_chng_q4_q1`        | 0.6943            | 0.7725            |
| `total_trans_amt`             | 3095.03           | 4654.66           |
| `total_trans_ct`              | 44.93             | 68.67             |
| `total_ct_chng_q4_q1`         | 0.5544            | 0.7424            |
| `avg_utilization_ratio`       | 0.1625            | 0.2964            |

Pelanggan yang churn (Attrited Customer) cenderung memiliki usia sedikit lebih tinggi dan jumlah tanggungan yang sedikit lebih banyak dibanding pelanggan yang tetap (Existing Customer). Mereka juga memiliki hubungan yang lebih singkat dengan bank, lebih sedikit produk perbankan, dan lebih jarang berinteraksi dalam 12 bulan terakhir. Nilai transaksi dan frekuensi transaksi mereka jauh lebih rendah, serta perubahan aktivitas transaksi kuartalan mereka juga lebih kecil. Selain itu, mereka memiliki saldo bergulir yang lebih tinggi dan pemanfaatan kredit yang lebih rendah, menunjukkan potensi kurangnya aktivitas finansial aktif. Secara umum, pelanggan yang churn menunjukkan pola hubungan dan transaksi yang lebih pasif dibanding pelanggan yang tetap.

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

Selanjutnya, dilakukan deteksi **outlier** menggunakan metode **Interquartile Range (IQR)** untuk setiap fitur numerik. Hasil analisis menunjukkan bahwa beberapa variabel memiliki jumlah outlier yang cukup signifikan, seperti **credit_limit (9.848 outlier)**, **contacts_count_12_mon (6.297 outlier)**, **months_on_book (3.864 outlier)**, **months_inactive_12_mon (3.316 outlier)**, **total_amt_chng_q4_q1 (3.961 outlier)**, dan **total_ct_chng_q4_q1 (3.941 outlier)**. Keberadaan outlier yang tinggi pada fitur-fitur tersebut mengindikasikan adanya variasi ekstrem dalam perilaku atau karakteristik nasabah, seperti frekuensi transaksi, perubahan aktivitas, dan batas kredit yang tidak biasa. Beberapa fitur lain seperti **avg_open_to_buy (963 outlier)**, **total_trans_amt (896 outlier)**, dan **customer_age (22 outlier)** juga menunjukkan adanya pencilan, meskipun dalam jumlah yang lebih rendah. Sementara itu, variabel seperti **clientnum**, **dependent_count**, **total_revolving_bal**, dan **avg_utilization_ratio** tidak mengandung outlier sama sekali, menandakan distribusi nilai yang relatif stabil untuk fitur-fitur tersebut.

| Variabel                     | Jumlah Outlier |
|------------------------------|----------------|
| `clientnum`                  | 0              |
| `customer_age`               | 22             |
| `dependent_count`            | 0              |
| `months_on_book`             | 3864           |
| `total_relationship_count`   | 5              |
| `months_inactive_12_mon`     | 3316           |
| `contacts_count_12_mon`      | 6297           |
| `credit_limit`               | 9848           |
| `total_revolving_bal`        | 0              |
| `avg_open_to_buy`            | 963            |
| `total_amt_chng_q4_q1`       | 3961           |
| `total_trans_amt`            | 896            |
| `total_trans_ct`             | 2              |
| `total_ct_chng_q4_q1`        | 3941           |
| `avg_utilization_ratio`      | 0              |

<figure>
    <center><img src="img/output_1.png" alt="Box-Plot Outlier"></center>
</figure>

Visualisasi melalui **boxplot** semakin memperjelas sebaran data dan keberadaan outlier di setiap fitur. Fitur seperti **months_on_book**, **credit_limit**, dan beberapa fitur lain tampak memiliki sebaran yang lebar dengan banyak data berada di luar whisker (batas IQR), yang mengindikasikan variasi nilai ekstrim dalam data tersebut.

Meskipun demikian, outlier **tidak dihapus** dari dataset. Hal ini dilakukan untuk menjaga **keutuhan informasi**, mengingat data pencilan tersebut mencerminkan kondisi nyata seperti lonjakan transaksi pelanggan. Menghilangkan outlier justru berisiko menghilangkan pola penting dalam konteks analisis churn pelanggan.

Sebagai langkah mitigasi terhadap pengaruh outlier, model _machine learning_ yang digunakan adalah model berbasis tree, sehingga lebih _robust_ terhadap outlier.

### [Exploratory Data Analysis] - Univariate Analysis

#### Grafik 1: Distribusi Kategori Churn Pelanggan
<figure>
    <center><img src="img/output_2.png" alt="Distribusi Target"></center>
</figure>

Proporsi pelanggan churn dalam dataset hanya sebesar 16,1%, menunjukkan **ketidakseimbangan kelas yang signifikan**. Untuk mengatasi hal ini, diterapkan beberapa strategi berikut:

1. **Stratified Hold-Out dan K-Fold Cross Validation** <br>
   Digunakan untuk memastikan proporsi churn tetap seimbang pada data pelatihan, validasi, dan pengujian, termasuk saat penyetelan hiperparameter.

2. **Penyesuaian Bobot Kelas** <br>
   Diberikan bobot lebih tinggi pada kelas minoritas (churner) agar model lebih sensitif terhadap kesalahan klasifikasi pada kelas ini.

3. **Prediksi Probabilistik & Precision-Recall Trade-off** <br>
   Fokus diarahkan pada prediksi probabilitas churn, bukan hanya klasifikasi biner, agar keputusan bisnis lebih fleksibel. Analisis precision-recall digunakan untuk menjaga keseimbangan performa model.

#### Grafik 2: Distribusi Fitur Numerik
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

### [Exploratory Data Analysis] - Multivariate Analysis

#### Grafik 1: Matriks Korelasi Fitur Numerik
<figure>
    <center><img src="img/output_5.png" alt="Matriks Korelasi"></center>
</figure>

- Jumlah transaksi, saldo revolving, dan perubahan jumlah transaksi berkorelasi negatif dengan churn — makin rendah, makin besar risiko churn.
- Umur dan lama menjadi nasabah berkorelasi positif — nasabah tua cenderung lebih loyal.
- Credit limit dan avg utilization ratio berkorelasi negatif; sedangkan revolving balance dan avg utilization ratio berkorelasi positif.
- Pria cenderung punya credit limit lebih tinggi.
- Avg open to buy dan credit limit berkorelasi sempurna, sehingga avg open to buy akan dihapus karena redundan.

Berikut distribusi beberapa hubungan di bawah ini.

<figure>
    <center><img src="img/output_6.png" alt="Scatter Plot Hubungan"></center>
</figure>

Menariknya, terdapat hubungan eksponensial menurun antara credit limit dan average utilization ratio. Artinya, semakin tinggi limit kredit, semakin rendah tingkat pemanfaatannya — dan penurunannya terjadi secara eksponensial. Dengan kata lain, nasabah dengan limit kredit tinggi cenderung lebih jarang menggunakan kartu kreditnya.

#### Grafik 2: Distribusi Fitur Numerik Berdasarkan Churn Flag
<figure>
    <center><img src="img/output_7.png" alt="Distribusi Fitur Numerik by churn_flag"></center>
</figure>

- Nasabah yang churn cenderung memiliki 1–3 produk, limit kredit lebih rendah, dan saldo revolving lebih kecil—bahkan banyak yang nol. Mereka juga melakukan lebih sedikit transaksi, baik dari sisi jumlah maupun nominal, serta perubahan transaksi yang kecil. Hal ini sejalan dengan perilaku tidak aktif sebelum berhenti menggunakan layanan.

- Rata-rata pemanfaatan kartu mereka juga rendah, bahkan banyak yang nol.

- Meskipun diskretisasi fitur kontinu bisa membantu analisis, untuk keperluan prediksi menggunakan model seperti LightGBM, hal itu bisa menambah kompleksitas dan justru merugikan. Karena pola perbedaan churn dan non-churn sudah terlihat jelas, diskretisasi tidak dilakukan.

#### Grafik 3: Distribusi Fitur Kategorik Berdasarkan Churn Rate
<figure>
    <center><img src="img/output_8.png" alt="Distribusi Kategorik Numerik by churn_rate"></center>
</figure>

- Tingkat churn meningkat seiring frekuensi kontak dengan bank; seluruh nasabah yang menghubungi sebanyak 6 kali tercatat churn.
- Kategori *unknown* pada status pernikahan memiliki tingkat churn tertinggi; pada tingkat pendidikan dan pendapatan, kategori ini menempati posisi kedua tertinggi. Oleh karena itu, kategori *unknown* sebaiknya tetap disertakan dalam tahap prapemrosesan karena memiliki daya diskriminasi.
- Meskipun distribusi jenis kartu sangat tidak seimbang (didominasi tipe *blue*), tingkat churn bervariasi antar kategori. Tipe *platinum* menunjukkan tingkat churn tertinggi. Variabel ini tetap digunakan untuk dianalisis lebih lanjut melalui pemilihan fitur atau penilaian pentingnya fitur dalam model prediktif seperti LightGBM.
- Tingkat churn lebih tinggi pada nasabah perempuan.
- Secara mengejutkan, nasabah yang tidak pernah tidak aktif selama 12 bulan terakhir justru memiliki tingkat churn tertinggi.
- Nasabah dengan tingkat pendidikan doktoral memiliki tingkat churn tertinggi dibandingkan kelompok pendidikan lainnya.

### [Kesimpulan EDA]

- Distribusi rasio pemanfaatan kartu kredit cenderung miring kanan, dengan sekitar 25% nasabah tidak menggunakan layanan sama sekali. Oleh karena itu, bank perlu mengembangkan strategi untuk meningkatkan tingkat pemanfaatan guna mendongkrak pendapatan.

- Sebanyak 90% nasabah tidak aktif selama 1–3 bulan terakhir. Bank harus menyiapkan langkah-langkah untuk mengurangi tingkat ketidakaktifan tersebut.

- Sekitar 75% nasabah menghubungi bank minimal dua kali dalam setahun. Bank perlu menggali lebih dalam alasan di balik panggilan tersebut, terutama apakah terdapat indikasi ketidakpuasan terhadap layanan.

- Terdapat korelasi positif yang kuat antara usia nasabah dan lama menjadi nasabah. Bank sebaiknya fokus pada upaya mempertahankan nasabah lama sekaligus meningkatkan masa aktif nasabah muda.

- Hubungan antara limit kredit dan tingkat pemanfaatan bersifat menurun secara eksponensial; nasabah dengan limit tinggi cenderung menggunakan kartu mereka dengan tingkat pemanfaatan yang lebih rendah.

- Nasabah yang churn umumnya memiliki limit kredit, saldo revolving, jumlah transaksi, serta tingkat pemanfaatan yang lebih rendah, bahkan banyak di antaranya tidak menggunakan kartu sama sekali.

- Tingkat churn meningkat seiring dengan banyaknya kontak nasabah ke bank; khususnya, seluruh nasabah yang melakukan enam kali kontak dilaporkan churn. Selain itu, nasabah dengan tingkat pendidikan doktoral menunjukkan tingkat churn tertinggi dibandingkan kelompok lainnya.

---

## Data Preparation

### 1. Label Encoding dengan Mapping pada Fitur Target

Proses encoding dilakukan secara manual untuk fitur target **churn_flag**. Mapping digunakan sebagai berikut:

| Kategori Churn Flag             | Label |
|---------------------------------|-------|
| `Existing Customer` => No Churn | 0     |
| `Attrited Customer` => Churn    | 1     |

### 2. Splitting Dataset

- Menetapkan `stratify = y` sehingga fungsi train_test_split memastikan bahwa proses pemisahan mempertahankan persentase yang sama dari setiap kelas target di set train dan test.

Dataset yang digunakan dalam analisis ini terdiri dari data pelatihan (train) dan data pengujian (test) dengan rincian sebagai berikut:

- **Ukuran data fitur (train)**: 8.101 observasi dengan 20 fitur.
- **Ukuran data target (train)**: 8.101 observasi.
- **Ukuran data fitur (test)**: 2.026 observasi dengan 20 fitur.
- **Ukuran data target (test)**: 2.026 observasi.

#### Proporsi Kelas pada Variabel Target

Distribusi proporsi kelas pada variabel target `churn_flag` untuk masing-masing data adalah sebagai berikut:

- **Data Pelatihan (Train)**:
  - Kelas 0 (tidak churn): 83,93%
  - Kelas 1 (churn): 16,07%

- **Data Pengujian (Test)**:
  - Kelas 0 (tidak churn): 83,96%
  - Kelas 1 (churn): 16,04%

Distribusi kelas yang relatif seimbang antara data pelatihan dan pengujian menunjukkan bahwa proses pembagian data telah mempertahankan proporsi kelas, sehingga model dapat dilatih dan dievaluasi secara konsisten terhadap fenomena *churn*.

### 3. Feature Engineering, Data Cleaning and Preprocessing¶

Preprocessing untuk Model Berbasis Tree
- **Fitur Numerik**: <br/>
    Tidak akan dilakukan transformasi apa pun karena model berbasis tree tidak memerlukan _feature scaling_.

- **Fitur Kategorikal** (Ordinal => education level, income category, dan card category): <br/>
    Akan diterapkan ordinal encoding untuk mempertahankan karakteristik ordinal.

- **Fitur Kategorikal** (Nominal => marital status): <br/>
    Akan diterapkan target encoding karena penggunaan one-hot encoding dapat merugikan model berbasis tree akibat _sparse representation_ dan meningkatnya dimensi.

- **Fitur Gender**:
    Akan diterapkan one-hot encoding karena fitur ini akan diubah menjadi variabel biner unik, sehingga tidak meningkatkan dimensi.

**Feature Engineering**

Untuk mendapatkan informasi maksimal dari fitur yang tersedia, dilakukan _feature engineering_ yang sudah terintegrasi dalam preprocessing dengan membuat fitur-fitur berikut:

Fitur Rasio: <br/>
<pre>
1. `products_per_dependent`      = total_relationship_count / dependent_count
2. `trans_amt_per_dependent`     = total_trans_amt / dependent_count
3. `trans_ct_per_dependent`      = total_trans_ct / dependent_count
4. `trans_amt_per_products`      = total_trans_amt / total_relationship_count
5. `trans_ct_per_products`       = total_trans_ct / total_relationship_count
6. `avg_trans_amt`               = total_trans_amt / total_trans_ct
7. `credit_util_rate`            = total_revolving_bal / credit_limit
8. `proportion_inactive_months`  = months_inactive_12_mon / months_on_book
9. `products_per_tenure`         = total_relationship_count / months_on_book
10. `products_per_contacts`      = total_relationship_count / contacts_count_12_mon
11. `dependents_per_contacts`    = dependent_count / contacts_count_12_mon
12. `trans_ct_per_contacts`      = total_trans_ct / contacts_count_12_mon
13. `products_per_inactivity`    = total_relationship_count / months_inactive_12_mon
14. `dependents_per_inactivity`  = dependent_count / months_inactive_12_mon
15. `trans_ct_per_inactivity`    = total_trans_ct / months_inactive_12_mon
16. `trans_amt_per_credit_limit` = total_trans_amt / credit_limit
17. `age_per_tenure`             = customer_age / months_on_book
18. `trans_ct_per_tenure`        = total_trans_ct / months_on_book
19. `trans_amt_per_tenure`       = total_trans_amt / months_on_book
</pre>

Fitur Penjumlahan: <br/>
<pre>
1. `total_spending`              = total_trans_amt + total_revolving_bal
2. `education_income_levels`     = education_level + income_category (ordinal)
3. `inactivity_contacts`         = contacts_count_12_mon + months_inactive_12_mon
</pre>

Fitur-fitur di atas dapat menangkap hubungan dan pola tersembunyi, serta relevan dalam konteks bisnis. Hal ini sangat penting untuk diperhatikan saat melakukan _feature engineering_.

**Penanganan Kategori `Unknown`**

income_category, marital_status, dan education_level:

Mempertahankan kategori `unknown` sebagai salah satu kategori dalam variabel-variabel tersebut. Hal ini dikarenakan hasil EDA menunjukkan bahwa kategori ini memberikan diskriminasi antara _churner_ dan _non-churner_.

- Pada `marital_status`, kategori `unknown` memiliki tingkat _churn_ tertinggi.
- Pada `education_level` dan `income_category`, kategori `unknown` memiliki tingkat _churn_ tertinggi kedua.

Dengan demikian, tidak dilakukan imputasi karena hal tersebut dapat memperkenalkan bias dan menghilangkan informasi penting.

**Penanganan Variabel `card_category`**

Mempertahankan variabel `card_category` dalam data meskipun distribusinya sangat tidak seimbang (_imbalanced_). Hasil EDA menunjukkan bahwa kategori Gold dan Silver memiliki tingkat churn yang lebih tinggi.

**Variabel yang Akan Dihapus**

1. `avg_open_to_buy`: <br/>
    Akan dihapus karena memiliki korelasi positif sempurna dengan credit_limit, sehingga informasinya menjadi redundan.
2. `CLIENTNUM`: <br/>
    Akan dihapus karena memiliki nilai unik untuk setiap _record_, sehingga tidak berguna untuk analisis.

---

## Model Training, Comparison, Selection and Tuning

### 1. Model Selection

Pada tahap pengembangan model, digunakan tiga algoritma klasifikasi berbasis tree yang umum dan efektif, yaitu Decision Tree (DT), Random Forest (RF), dan LightGBM (LGBM). Pemilihan ketiga model tersebut didasarkan pada karakteristik masing-masing serta tujuan untuk membandingkan performa secara empiris.

#### Decision Tree (DT)

```python
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_scaled, y_train)
```
[Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) merupakan model klasifikasi yang membagi data berdasarkan fitur yang paling informatif. Model ini bekerja dengan membuat pohon keputusan dari akar hingga daun berdasarkan aturan if-else yang memaksimalkan pemisahan kelas. Sebagai model yang mudah dipahami dan divisualisasikan, Decision Tree cocok digunakan sebagai baseline yang interpretatif.

#### Random Forest (RF)

````python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)
````
[Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) adalah metode ensemble yang menggabungkan banyak pohon keputusan untuk meningkatkan akurasi dan stabilitas prediksi. Dengan membangun pohon pada subset data secara acak dan menggabungkan hasilnya, model ini mampu mengurangi overfitting dan bekerja baik pada data dengan banyak fitur.

#### LightGBM (LGBM)

````python
from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train_scaled, y_train)
````
[LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) adalah algoritma boosting berbasis pohon yang dikembangkan untuk efisiensi tinggi dan kecepatan pelatihan. Model ini bekerja dengan pendekatan leaf-wise dan mampu menangani dataset besar dengan performa tinggi.

Ketiga model ini digunakan dengan pengaturan parameter awal sebagai percobaan dasar

- Pada langkah ini, membandingkan kinerja model yang berbeda dengan menggunakan **_stratified k-fold cross validation_** untuk melatih masing-masing model dan mengevaluasi skor ROC-AUC. Stratified k-fold cross validation akan mempertahankan proporsi target pada setiap fold, menangani target yang tidak seimbang.

- _k-fold cross validation_ adalah teknik yang digunakan dalam _machine learning_ untuk menilai kinerja model. Teknik ini melibatkan pembagian dataset menjadi K subset, menggunakan K-1 untuk pelatihan dan satu untuk pengujian secara berulang. Hal ini membantu dalam memperkirakan kemampuan generalisasi model dengan mengurangi risiko _overfitting_ dan memberikan metrik kinerja yang lebih andal.

- Tujuan tahap ini adalah untuk memilih model terbaik untuk digunakan dalam _feature selection_, _hyperparameter tuning_, dan evaluasi model akhir. Untuk mendapatkan model terbaik ini, akan dievaluasi skor validasi rata-rata **roc-auc** tertinggi dan melihat trade-off bias-varians.

#### Tabel Perbandingan Performa Model

| Metrik               | LightGBM | Random Forest  | Decision Tree  |
|----------------------|----------|----------------|----------------|
| ROC AUC (Val)        | 0.992292 | 0.988307       | 0.876184       |
| Akurasi (Val)        | 0.969263 | 0.958400       | 0.933094       |
| Recall (Val)         | 0.879434 | 0.805721       | 0.780342       |
| Spesifisitas (Val)   | 0.986468 | 0.989410       | 0.960729       |
| ROC AUC (Train)      | 1.0      | 1.0            | 1.0            |
| Akurasi (Train)      | 1.0      | 1.0            | 1.0            |
| Recall (Train)       | 1.0      | 1.0            | 1.0            |
| Spesifisitas (Train) | 1.0      | 1.0            | 1.0            |
| Waktu Latih (detik)  | 0.296113 | 3.120936       | 0.369970       |

<figure>
    <center><img src="img/output_9.png" alt="Hasil Performa Model"></center>
</figure>

Model **LightGBM** dipilih untuk proses _feature selection_, _hyperparameter tuning_, dan evaluasi akhir karena menunjukkan performa terbaik dengan rata-rata skor **ROC-AUC validasi tertinggi**. Meskipun model mengalami indikasi _overfitting_ (skor ROC-AUC sebesar 1 pada data pelatihan), hasil validasi tetap sangat tinggi (**0,99**), menunjukkan generalisasi yang kuat.

Performa luar biasa ini **bukan disebabkan oleh kebocoran data**, karena seluruh fitur yang digunakan tersedia pada saat prediksi dan pembagian data dilakukan secara tepat sebelum proses modeling. Hasil ini mencerminkan **kualitas data yang baik**, di mana variabel independen secara jelas mampu membedakan antara nasabah yang churn dan tidak.

Meskipun potensi peningkatan performa lebih lanjut melalui _hyperparameter tuning_ relatif kecil, langkah tersebut **tetap akan dilakukan** sebagai bagian dari proses penyempurnaan model.

### 2. Feature Selection

- Langkah seleksi fitur sangat penting untuk meningkatkan kemampuan generalisasi model dan membuatnya lebih sederhana, sehingga menambah efisiensi komputasi. Mengingat terdapat 40 fitur, menyederhanakan model tanpa kehilangan performa adalah keuntungan yang sangat besar. Digunakan metode _Recursive Feature Elimination_ (RFE) untuk seleksi fitur.

- _Recursive Feature Elimination_ (RFE) adalah metode seleksi fitur yang secara sistematis menghilangkan fitur yang tidak relevan atau kurang penting dari model prediktif. Metode ini bekerja dengan melatih model secara berulang pada subset fitur, memberikan peringkat berdasarkan pentingnya fitur, dan mengeliminasi fitur yang paling tidak penting hingga jumlah fitur yang diinginkan tercapai.

- Salah satu hyperparameter penting dalam RFE adalah jumlah akhir fitur yang diinginkan. Untuk menentukan nilai ini secara otomatis, akan digunakan kelas `RFECV` dari sklearn. `RFECV` akan menerapkan _stratified k-fold cross-validation_ untuk menemukan nilai terbaik dari hyperparameter tersebut.

#### Daftar Fitur Terpilih (Feature Selection)

| No | Nama Fitur                    |
|----|-------------------------------|
| 0  | customer_age                  |
| 1  | contacts_count_12_mon         |
| 2  | credit_limit                  |
| 3  | total_revolving_bal           |
| 4  | total_amt_chng_q4_q1          |
| 5  | total_trans_amt               |
| 6  | total_trans_ct                |
| 7  | total_ct_chng_q4_q1           |
| 8  | avg_utilization_ratio         |
| 9  | trans_amt_per_dependent       |
| 10 | trans_ct_per_dependent        |
| 11 | trans_amt_per_products        |
| 12 | trans_ct_per_products         |
| 13 | avg_trans_amt                 |
| 14 | proportion_inactive_months    |
| 15 | products_per_tenure           |
| 16 | trans_ct_per_contacts         |
| 17 | products_per_inactivity       |
| 18 | trans_ct_per_inactivity       |
| 19 | trans_amt_per_credit_limit    |
| 20 | age_per_tenure                |
| 21 | trans_ct_per_tenure           |
| 22 | trans_amt_per_tenure          |
| 23 | total_spending                |
| 24 | education_income_levels       |

- Seperti yang dapat dilihat, fitur-fitur yang menunjukkan diskriminasi yang jelas antara _churners_ dan _non-churners_ tetap dipertahankan, seperti **total_trans_ct** dan **total_trans_amt**. Selain itu, banyak fitur yang dihasilkan dari proses rekayasa fitur juga tetap dipertahankan, menunjukkan pentingnya tahap ini terhadap performa model.

- Sebagai hasilnya, jumlah fitur berhasil dikurangi dari 40 menjadi 25 fitur paling penting, yang secara signifikan mengurangi kompleksitas komputasi.

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
