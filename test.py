from flask import Flask, redirect, render_template, request, url_for
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
# Inisialisasi dataset
inputData = 'static/dataset/Data Kuliner.xlsx'
df = pd.read_excel(inputData)

# Cleaning Data
df.columns = ['no', 'Reff_OSS', 'NIK', 'Nama_Lengkap', 'TGL', 'Usia', 'JK', 'Pendidikan', 'No_Telp', 'Email', 'Provinsi', 'Kabupaten', 'Kecamatan', 'Desa', 'Nama_Jln', 'Nama_Usaha', 'NIB', 'Tgl_Terbit_NIB', 'Tgl_Pendirian_Usaha', 'Koordinat', 'Bidang_Usaha', 'Sektor_Usaha',
              'Kegiatan_Usaha', 'Produk_Komoditas_Ekspor', 'Tujuan_Pemasaran', 'Status_Kepemilikan_Tanah', 'Sarana_Media_Elektronik', 'Modal_Bantuan_Pemerintah', 'Pinjaman', 'Omset_Pertahun', 'Kepemilikan_Asuransi_Kesehatan', 'Tenaga_Kerja_Laki', 'Tenaga_Kerja_Perempuan', 'Rerata_Usia_Pekerja', 'Status_Formulir']

data = df
data_cleaning = data.dropna()

# Selection Data
data_selection = data_cleaning.loc[:, ['Pendidikan', 'Tgl_Pendirian_Usaha', 'Kegiatan_Usaha', 'Tujuan_Pemasaran', 'Status_Kepemilikan_Tanah',
                                       'Sarana_Media_Elektronik', 'Modal_Bantuan_Pemerintah', 'Pinjaman', 'Omset_Pertahun', 'Kepemilikan_Asuransi_Kesehatan', 'Tenaga_Kerja_Laki', 'Tenaga_Kerja_Perempuan']]

# Transformation Data
data2 = data_selection

# Tranformasi kolom Pendidikan
t_pendidikan = pd.get_dummies(data2.Pendidikan)

# Penghitungan Umur Usaha
for index, row in data2.iterrows():
    data2.loc[index, 'Umur_Usaha'] = datetime.now(
    ).year - int(row['Tgl_Pendirian_Usaha'][-4:])
t_umur_usaha = data2['Umur_Usaha']

# Tranformasi kolom kegiatan usaha
t_kegiatan_usaha = data2['Kegiatan_Usaha'].str.get_dummies(sep=', ')

# Tranformasi kolom tujuan pemasaran
t_tujuan_pemasaran = data2['Tujuan_Pemasaran'].str.get_dummies(sep=', ')

# transformasi kolom kepemilikan tanah
t_kepemilikan_tanah = data2['Status_Kepemilikan_Tanah'].str.get_dummies(
    sep=', ')

# transformasi kolom sarana media elektronik
t_sarana_media_elektronik = data2['Sarana_Media_Elektronik'].str.get_dummies(
    sep=', ')

# transformasi kolom modal bantuan pemerintah
t_modal_bantuan_pemerintah = pd.get_dummies(data2.Modal_Bantuan_Pemerintah)

# transformasi kolom pinjaman
t_pinjaman = data2['Pinjaman'].str.get_dummies(sep=', ')

# transformasi kolom omset pertahun
t_omset_pertahun = pd.get_dummies(data2.Omset_Pertahun)

# transformasi kolom asuransi
t_asuransi = data2['Kepemilikan_Asuransi_Kesehatan'].str.get_dummies(sep=', ')

# memasukkan kolom ke variabel untuk digabungkan
t_tenaga_kerja_laki = data2['Tenaga_Kerja_Laki']
t_tenaga_kerja_perempuan = data2['Tenaga_Kerja_Perempuan']

# proses penyatuan hasil transformasi untuk di transformasi
data_transformasi = pd.concat([t_pendidikan, t_umur_usaha, t_kegiatan_usaha, t_tujuan_pemasaran, t_kepemilikan_tanah, t_sarana_media_elektronik,
                               t_modal_bantuan_pemerintah, t_pinjaman, t_omset_pertahun, t_asuransi, t_tenaga_kerja_laki, t_tenaga_kerja_perempuan], axis='columns')


# dendogram
plt.title("Dendograms")
Z = shc.linkage(data_transformasi, method='single')
dend = shc.dendrogram(Z)

# CLUSTERING
clustering = AgglomerativeClustering(
    n_clusters=2, affinity='euclidean', linkage='single')
cluster_result = clustering.fit_predict(data_transformasi)

# PENGUJIAN SILHOUETTE
silh_avg_score_ = silhouette_score(data_transformasi, cluster_result)

# PENAMBAHAN CLUSTER KE TABEL
data_cleaning['Cluster'] = cluster_result

##########################################

ALLOWED_EXTENSION = set(['xlsx'])
app.config['UPLOAD_FOLDER'] = 'static/dataset'

# menguji upload file


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION


@app.route('/', methods=['GET', 'POST'])
def index():
    # return render_template('index.html')
    if request.method == 'POST':
        # filedata merupakan nama variabel yang terdapat pada html
        file = request.files['filedata']
        if 'filedata' not in request.files:
            return redirect(request.url)

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return 'file berhasil disave di ' + filename

    return render_template('index.html')


@app.route('/cleaning')
def cleaning():
    return render_template('cleaning.html', data_tabel=[data_selection.to_html(classes="table table-bordered", table_id="clean")])


@app.route('/transformation')
def transformation():
    return render_template('transformation.html', data_tabel=[data_transformasi.to_html(classes="table table-bordered", table_id="transformation")])


@app.route('/cluster')
def cluster():
    return render_template('cluster.html', data_tabel=[data_cleaning.to_html(classes="table table-bordered", table_id="cluster")],  score=silh_avg_score_, dendo=shc.dendrogram(Z))


if __name__ == '__main__':
    app.run(debug=True)
