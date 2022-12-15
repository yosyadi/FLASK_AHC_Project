#from crypt import methods
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

loadData = ''
# df = pd.read_excel('static/dataset/Data Kuliner.xlsx')
# df = pd.read_excel (r'data\default.xlsx', sheet_name='UKM Jasa')
# df = pd.DataFrame()
df = pd.read_excel('static/dataset/Data_Kerajianan.xlsx')
df_cleaning = df
df_selection = df_cleaning
df_transformation = df_selection
# df_cleaning = df
# data = df_cleaning.iloc[:, [7, 18, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37]]
# df_cluster = data

# df_transformasi = df_cleaning
# df_transformation = df_cleaning

ALLOWED_EXTENSION = set(['xlsx'])
app.config['UPLOAD_FOLDER'] = 'static/dataset'

# menguji upload file


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/load_data', methods=['GET', 'POST'])
def load_data():
    global df
    global loadData
    # return render_template('index.html')
    if request.method == 'POST':
        # filedata merupakan nama variabel yang terdapat pada html
        file = request.files['filedata']
        if 'filedata' not in request.files:
            return redirect(request.url)

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            loadData = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], loadData))
            df = pd.read_excel('static/dataset/'+loadData)
            print(loadData, df)
            # return loadData
        return render_template('load_data.html', data_tabel=[df.to_html(classes="table table-bordered", table_id="asli")])
    return render_template('load_data.html')


@app.route('/preprocessing')
def cleaning_selection():
    global df
    global df_cleaning
    global df_selection
    df_cleaning = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]]
    # df_cleaning = df_cleaning.drop(labels=[1])
    # Cleaning Data
    df_cleaning = df_cleaning.dropna()
    df_cleaning.columns = ['no', 'Reff_OSS', 'NIK', 'Nama_Lengkap', 'Tanggal_Lahir', 'Usia', 'Jenis_Kelamin', 'Pendidikan', 'No_Telp', 'Email', 'Provinsi', 'Kabupaten', 'Kecamatan', 'Desa', 'Nama_Jln', 'Nama_Usaha', 'NIB', 'Tgl_Terbit_NIB', 'Tgl_Pendirian_Usaha', 'Koordinat', 'Bidang_Usaha', 'Sektor_Usaha',
                           'Kegiatan_Usaha', 'Produk_Komoditas_Ekspor', 'Tujuan_Pemasaran', 'Status_Kepemilikan_Tanah', 'Sarana_Media_Elektronik', 'Modal_Bantuan_Pemerintah', 'Pinjaman', 'Omset_Pertahun', 'Kepemilikan_Asuransi_Kesehatan', 'Tenaga_Kerja_Laki', 'Tenaga_Kerja_Perempuan', 'Rerata_Usia_Pekerja', 'Status_Formulir']

    # Selection Data
    df_selection = df_cleaning.loc[:, ['Pendidikan', 'Tgl_Pendirian_Usaha', 'Kegiatan_Usaha', 'Tujuan_Pemasaran', 'Status_Kepemilikan_Tanah',
                                       'Sarana_Media_Elektronik', 'Modal_Bantuan_Pemerintah', 'Pinjaman', 'Omset_Pertahun', 'Kepemilikan_Asuransi_Kesehatan', 'Tenaga_Kerja_Laki', 'Tenaga_Kerja_Perempuan']]

    return render_template('preprocessing.html', data_clean=[df_cleaning.to_html(classes="table table-bordered", table_id="clean")], data_select=[df_selection.to_html(classes="table table-bordered", table_id="select")])


@ app.route('/transformation')
def transformation():
    global df_transformation
    data = df_selection
    # Tranformasi kolom Pendidikan
    t_pendidikan = pd.get_dummies(data.Pendidikan)

    # Penghitungan Umur Usaha
    for index, row in data.iterrows():
        data.loc[index, 'Umur_Usaha'] = datetime.now(
        ).year - int(row['Tgl_Pendirian_Usaha'][-4:])
    t_umur_usaha = data['Umur_Usaha']

    # Tranformasi kolom kegiatan usaha
    t_kegiatan_usaha = data['Kegiatan_Usaha'].str.get_dummies(
        sep=', ')

    # Tranformasi kolom tujuan pemasaran
    t_tujuan_pemasaran = data['Tujuan_Pemasaran'].str.get_dummies(
        sep=', ')

    # transformasi kolom kepemilikan tanah
    t_kepemilikan_tanah = data['Status_Kepemilikan_Tanah'].str.get_dummies(
        sep=', ')

    # transformasi kolom sarana media elektronik
    t_sarana_media_elektronik = data['Sarana_Media_Elektronik'].str.get_dummies(
        sep=', ')

    # transformasi kolom modal bantuan pemerintah
    t_modal_bantuan_pemerintah = pd.get_dummies(
        data.Modal_Bantuan_Pemerintah)

    # transformasi kolom pinjaman
    t_pinjaman = data['Pinjaman'].str.get_dummies(sep=', ')

    # transformasi kolom omset pertahun
    t_omset_pertahun = pd.get_dummies(data.Omset_Pertahun)

    # transformasi kolom asuransi
    t_asuransi = data['Kepemilikan_Asuransi_Kesehatan'].str.get_dummies(
        sep=', ')

    # memasukkan kolom ke variabel untuk digabungkan
    t_tenaga_kerja_laki = data['Tenaga_Kerja_Laki']
    t_tenaga_kerja_perempuan = data['Tenaga_Kerja_Perempuan']

    # proses penyatuan hasil transformasi untuk di transformasi
    df_transformation = pd.concat([t_pendidikan, t_umur_usaha, t_kegiatan_usaha, t_tujuan_pemasaran, t_kepemilikan_tanah, t_sarana_media_elektronik,
                                  t_modal_bantuan_pemerintah, t_pinjaman, t_omset_pertahun, t_asuransi, t_tenaga_kerja_laki, t_tenaga_kerja_perempuan], axis='columns')

    return render_template('transformation.html', data_transform=[df_transformation.to_html(classes="table table-bordered", table_id="transform")])


@app.route('/cluster', methods=['GET', 'POST'])
def cluster():
    global df_cluster
    df_cluster = df_selection
    v_cluster = request.form.get('center_point')
    # v_cluster = 3
    # dendogram
    plt.figure(figsize=(150, 70))
    plt.title("Dendograms")
    dend = shc.dendrogram(shc.linkage(
        df_transformation, method='complete', metric='euclidean'))
    plt.savefig('static/img/dendogram.png', format='png', bbox_inches='tight')
    # CLUSTERING
    clustering = AgglomerativeClustering(n_clusters=int(
        v_cluster), affinity='euclidean', linkage='single')
    cluster_result = clustering.fit_predict(df_transformation)

    # PENGUJIAN SILHOUETTE
    silh_avg_score_ = silhouette_score(df_transformation, cluster_result)

    data_print_cluster = df_cluster

    # PENAMBAHAN CLUSTER KE TABEL
    data_print_cluster['cluster'] = cluster_result

    new_X = data_print_cluster

    new_Y = pd.DataFrame(new_X)

    clust0 = new_Y.apply(
        lambda x: True if x['cluster'] == 0 else False, axis=1)
    clust1 = new_Y.apply(
        lambda x: True if x['cluster'] == 1 else False, axis=1)
    clust2 = new_Y.apply(
        lambda x: True if x['cluster'] == 2 else False, axis=1)
    clust3 = new_Y.apply(
        lambda x: True if x['cluster'] == 3 else False, axis=1)
    clust4 = new_Y.apply(
        lambda x: True if x['cluster'] == 4 else False, axis=1)
    clust5 = new_Y.apply(
        lambda x: True if x['cluster'] == 5 else False, axis=1)
    jumlah0 = len(clust0[clust0 == True].index)
    jumlah1 = len(clust1[clust1 == True].index)
    jumlah2 = len(clust2[clust2 == True].index)
    jumlah3 = len(clust3[clust3 == True].index)
    jumlah4 = len(clust4[clust4 == True].index)
    jumlah5 = len(clust5[clust5 == True].index)

    Data = {'Chart': [jumlah0, jumlah1, jumlah2, jumlah3, jumlah4, jumlah5]}
    diagram_pie = pd.DataFrame(Data, columns=['Chart'], index=[
                               'Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'])

    diagram_pie.plot.pie(y='Chart', figsize=(
        8, 8), autopct='%1.2f%%', startangle=70)
    plt.savefig('static/img/chart.png', format='png', bbox_inches='tight')

    return render_template('cluster.html', data_hasil=[data_print_cluster.to_html(classes="table table-bordered", table_id="data")], cluster_count=v_cluster, slh=silh_avg_score_)


if __name__ == '__main__':
    app.run(debug=True)
