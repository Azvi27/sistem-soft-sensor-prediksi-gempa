# Impor pustaka yang diperlukan
import pandas as pd
import numpy as np
import datetime
import joblib # Untuk menyimpan dan memuat model
import os
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder # Untuk binning
import folium # Untuk visualisasi peta
import webbrowser # Untuk membuka file HTML secara otomatis
import matplotlib.pyplot as plt # Untuk plot statis
import seaborn as sns # Untuk plot statis yang lebih baik
# ... impor pustaka lain seperti folium, webbrowser ...

import matplotlib
matplotlib.use('Agg') # <-- TAMBAHKAN BARIS INI
import matplotlib.pyplot as plt # Baris ini sudah ada sebelumnya
import seaborn as sns # Baris ini sudah ada sebelumnya

# ... sisa kode Anda ...

# --- Konfigurasi Global ---
DATA_FILE = 'gempa_data_cleaned.csv' 
MAX_DIST_PREV_KM_THRESHOLD = 270 # Tetap menggunakan threshold dari v9/v10

# Path untuk menyimpan/memuat model Random Forest
MODEL_RF_LAT_PATH = f'combined_rf_model_latitude_v15_dist{MAX_DIST_PREV_KM_THRESHOLD}_binned_time.pkl'
MODEL_RF_LON_PATH = f'combined_rf_model_longitude_v15_dist{MAX_DIST_PREV_KM_THRESHOLD}_binned_time.pkl'
MODEL_RF_MAG_PATH = f'combined_rf_model_mag_v15_dist{MAX_DIST_PREV_KM_THRESHOLD}_binned_time.pkl' 
MODEL_RF_DEP_PATH = f'combined_rf_model_depth_v15_dist{MAX_DIST_PREV_KM_THRESHOLD}_binned_time.pkl' 
FEATURE_STATS_PATH = f'feature_stats_v15_dist{MAX_DIST_PREV_KM_THRESHOLD}_binned_time.pkl' # Path unik untuk statistik v15
WEB_TIME_ENCODER_PATH = 'web_time_log_encoder.pkl' # <-- TAMBAHKAN BARIS INI

# Definisi bin untuk time_diff_hours_log
# Nilai log1p(jam): log1p(0.1 jam) ~ 0.095, log1p(1 jam) ~ 0.69, log1p(6 jam) ~ 1.94, log1p(24 jam) ~ 3.2
# Batas ini bisa disesuaikan berdasarkan distribusi data Anda
TIME_LOG_BINS = [-np.inf, 0.1, 0.7, 2.0, np.inf] 
TIME_LOG_BIN_LABELS = ['time_log_very_short', 'time_log_short', 'time_log_medium', 'time_log_long']

# Fitur yang akan digunakan untuk melatih model Random Forest
FEATURES_RF_BASE = [ 
    'year', 'month_sin', 'month_cos', 
    'lat_prev', 'lon_prev', 'depth_prev', 'mag_prev',
    'dist_to_prev_km', # Kembali menggunakan versi kontinu
    'abs_depth_change', 
    'mag_change'
    # 'time_diff_hours_log', # Akan diganti dengan bin
]
FEATURES_RF = FEATURES_RF_BASE.copy() 
# Penambahan label OHE untuk waktu akan dilakukan di load_and_preprocess_data_all_targets

TARGET_LAT_RF = 'lat_target' 
TARGET_LON_RF = 'lon_target' 
TARGET_MAG_RF = 'mag_target' 
TARGET_DEP_RF = 'depth_target' 

REQUIRED_COLUMNS = ['tgl', 'ot', 'lat', 'lon', 'depth', 'mag', 'month']

# Global OneHotEncoder untuk konsistensi waktu
time_log_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
time_encoder_fitted = False 

# --- Fungsi Utilitas ---
def haversine_distance_func(lat1, lon1, lat2, lon2):
    R = 6371
    lat1_rad, lon1_rad = np.radians(np.asarray(lat1, dtype=float)), np.radians(np.asarray(lon1, dtype=float))
    lat2_rad, lon2_rad = np.radians(np.asarray(lat2, dtype=float)), np.radians(np.asarray(lon2, dtype=float))
    delta_phi, delta_lambda = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = np.sin(delta_phi / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# --- 1. Fungsi Pemuatan dan Pra-pemrosesan Data (dengan binning waktu) ---
def load_and_preprocess_data_all_targets(file_path=DATA_FILE, dist_threshold=MAX_DIST_PREV_KM_THRESHOLD):
    global time_encoder_fitted, time_log_encoder, FEATURES_RF 
    print(f"INFO: Memulai pemuatan dan pra-pemrosesan data dari {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"KRITIS: File data '{file_path}' tidak ditemukan. Proses dihentikan.")
        raise FileNotFoundError(f"File data '{file_path}' tidak ditemukan.")
    df = pd.read_csv(file_path)
    print(f"INFO: Data awal dimuat. Jumlah baris: {len(df)}, Jumlah kolom: {len(df.columns)}")
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"KRITIS: Kolom berikut tidak ditemukan di CSV: {', '.join(missing_cols)}. Proses dihentikan.")
        raise ValueError(f"Kolom penting tidak ditemukan: {', '.join(missing_cols)}")
    try:
        df['datetime'] = pd.to_datetime(df['tgl'] + ' ' + df['ot'], errors='coerce')
    except Exception:
        df['datetime'] = pd.to_datetime(df['tgl'], errors='coerce')
    initial_rows = len(df)
    df.dropna(subset=['datetime'], inplace=True)
    rows_after_dt_dropna = len(df)
    print(f"INFO: Baris setelah dropna 'datetime': {rows_after_dt_dropna} (hilang {initial_rows - rows_after_dt_dropna} baris)")
    if df.empty: raise ValueError("Data kosong setelah validasi datetime.")
    df.sort_values('datetime', inplace=True)
    df['year'] = df['datetime'].dt.year
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['lat_prev'] = df['lat'].shift(1)
    df['lon_prev'] = df['lon'].shift(1)
    df['depth_prev'] = df['depth'].shift(1)
    df['mag_prev'] = df['mag'].shift(1)
    df['datetime_prev'] = df['datetime'].shift(1)
    df['time_diff_hours'] = (df['datetime'] - df['datetime_prev']).dt.total_seconds() / 3600.0
    df['time_diff_hours_log'] = np.log1p(df['time_diff_hours']) # Tetap hitung versi log untuk binning
    df['dist_to_prev_km'] = np.nan
    valid_prev_indices = df['lat_prev'].notna() & df['lon_prev'].notna() & df['lat'].notna() & df['lon'].notna()
    df.loc[valid_prev_indices, 'dist_to_prev_km'] = df[valid_prev_indices].apply(
        lambda row: haversine_distance_func(row['lat'], row['lon'], row['lat_prev'], row['lon_prev']), axis=1
    )
    df['depth_change'] = df['depth'] - df['depth_prev']
    df['abs_depth_change'] = df['depth_change'].abs() 
    df['mag_change'] = df['mag'] - df['mag_prev']
    df[TARGET_LAT_RF] = df['lat']
    df[TARGET_LON_RF] = df['lon']
    df[TARGET_MAG_RF] = df['mag']
    df[TARGET_DEP_RF] = df['depth']

    pre_bin_features = FEATURES_RF_BASE + ['time_diff_hours_log'] + [TARGET_LAT_RF, TARGET_LON_RF, TARGET_MAG_RF, TARGET_DEP_RF]
    cols_to_check_nan_pre_bin = [col for col in pre_bin_features if col in df.columns]
    
    initial_rows_before_rf_dropna = len(df)
    df_rf_ready = df.dropna(subset=cols_to_check_nan_pre_bin).copy()
    rows_after_rf_dropna = len(df_rf_ready)
    print(f"INFO: Baris data RF setelah dropna fitur dasar: {rows_after_rf_dropna} (hilang {initial_rows_before_rf_dropna - rows_after_rf_dropna} baris)")

    if dist_threshold is not None:
        print(f"INFO: Menerapkan filter jarak: dist_to_prev_km < {dist_threshold} km")
        original_count_before_dist_filter = len(df_rf_ready)
        df_rf_ready = df_rf_ready[df_rf_ready['dist_to_prev_km'] < dist_threshold].copy()
        print(f"INFO: Baris data RF setelah filter jarak: {len(df_rf_ready)} (hilang {original_count_before_dist_filter - len(df_rf_ready)} baris)")
        if df_rf_ready.empty:
            print(f"PERINGATAN: Tidak ada data RF tersisa setelah filter jarak dengan threshold {dist_threshold} km.")
    
    if df_rf_ready.empty: 
        print("KRITIS: Tidak ada data valid untuk RF setelah pra-pemrosesan akhir (filter/dropna).")
        return pd.DataFrame(columns=FEATURES_RF), pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64'), None

    print(f"INFO: Melakukan binning pada 'time_diff_hours_log' dengan batas: {TIME_LOG_BINS}")
    df_rf_ready['time_log_bin'] = pd.cut(df_rf_ready['time_diff_hours_log'], 
                                         bins=TIME_LOG_BINS, 
                                         labels=TIME_LOG_BIN_LABELS, 
                                         right=False, 
                                         include_lowest=True) 
    
    if not time_encoder_fitted:
        time_log_encoder.fit(df_rf_ready[['time_log_bin']])
        time_encoder_fitted = True
        try:
            # Simpan encoder yang sudah di-fit ke file
            joblib.dump(time_log_encoder, WEB_TIME_ENCODER_PATH)
            print(f"INFO: Time encoder yang sudah dilatih disimpan ke {WEB_TIME_ENCODER_PATH}")
        except Exception as e:
            print(f"ERROR: Gagal menyimpan time encoder: {e}")
        
    time_log_encoded_features = time_log_encoder.transform(df_rf_ready[['time_log_bin']])
    ohe_time_column_names = time_log_encoder.get_feature_names_out(['time_log_bin'])
    time_log_encoded_df = pd.DataFrame(time_log_encoded_features, 
                                       columns=ohe_time_column_names,
                                       index=df_rf_ready.index)
    
    df_rf_ready = pd.concat([df_rf_ready, time_log_encoded_df], axis=1)
    columns_to_drop_after_time_bin = ['time_diff_hours_log', 'time_log_bin']
    if 'time_diff_hours' in df_rf_ready.columns: 
        columns_to_drop_after_time_bin.append('time_diff_hours')
    df_rf_ready.drop(columns=columns_to_drop_after_time_bin, axis=1, inplace=True)
    
    FEATURES_RF = FEATURES_RF_BASE.copy() 
    if 'time_diff_hours_log' in FEATURES_RF: FEATURES_RF.remove('time_diff_hours_log') 
    
    for col_name in ohe_time_column_names:
        if col_name not in FEATURES_RF:
             FEATURES_RF.append(col_name)
    print(f"INFO: Fitur RF akhir setelah one-hot encoding waktu: {FEATURES_RF}")
    
    actual_features_in_df = [f for f in FEATURES_RF if f in df_rf_ready.columns]
    if len(actual_features_in_df) != len(FEATURES_RF):
        missing_from_df = set(FEATURES_RF) - set(df_rf_ready.columns)
        print(f"WARNING: Fitur berikut ada di FEATURES_RF tapi tidak di DataFrame: {missing_from_df}")
        X_rf = df_rf_ready[actual_features_in_df]
    else:
        X_rf = df_rf_ready[FEATURES_RF]

    y_rf_lat = df_rf_ready[TARGET_LAT_RF]
    y_rf_lon = df_rf_ready[TARGET_LON_RF]
    y_rf_mag = df_rf_ready[TARGET_MAG_RF]
    y_rf_dep = df_rf_ready[TARGET_DEP_RF]

    feature_stats = {
        'median_abs_depth_change': X_rf['abs_depth_change'].median() if 'abs_depth_change' in X_rf.columns else 0.0,
        'median_mag_change': X_rf['mag_change'].median() if 'mag_change' in X_rf.columns else 0.0,
        'median_dist_to_prev_km': X_rf['dist_to_prev_km'].median() if 'dist_to_prev_km' in X_rf.columns else 0.0
    }
    try:
        joblib.dump(feature_stats, FEATURE_STATS_PATH)
        print(f"INFO: Statistik fitur untuk inisialisasi disimpan ke {FEATURE_STATS_PATH}")
        print(f"INFO: Statistik yang disimpan: {feature_stats}")
    except Exception as e:
        print(f"ERROR: Gagal menyimpan statistik fitur: {e}")

    print("INFO: Pra-pemrosesan data untuk RF selesai.")
    return X_rf, y_rf_lat, y_rf_lon, y_rf_mag, y_rf_dep, feature_stats

# --- 2. Fungsi Pelatihan Model Random Forest (Sama) ---
def train_random_forest_model_tuned(X_train, y_train, model_save_path, model_name="RF Model"):
    if X_train.empty or y_train.empty: print(f"ERROR: Data latih {model_name} kosong."); return None
    print(f"INFO: Memulai pelatihan (TUNING MENDALAM) untuk {model_name} dengan {len(X_train)} sampel...")
    param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [20, 30, None],
                  'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    rf = RandomForestRegressor(random_state=42)
    cv_folds = 5 
    if len(X_train) < cv_folds * 2 :
        needed_samples = cv_folds * 2
        print(f"WARNING: Sampel latih ({len(X_train)}) < kebutuhan CV ({needed_samples}).")
        if len(X_train) >= 4 : cv_folds = max(2, len(X_train) // 2); print(f"INFO: CV folds {model_name} jadi {cv_folds}.")
        else: print(f"ERROR: Sampel ({len(X_train)}) terlalu sedikit untuk GridSearchCV."); return None 
    best_model = None
    if cv_folds <= 1 and len(X_train) > 0 :
        print(f"INFO: Melatih {model_name} tanpa GridSearchCV (sampel tidak cukup).")
        best_model = rf.fit(X_train, y_train)
    elif len(X_train) == 0: print(f"ERROR: Data latih {model_name} kosong."); return None
    else:
        print(f"INFO: Memulai GridSearchCV cv={cv_folds} untuk {model_name}. Ini akan lama...")
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv_folds, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
        try:
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"INFO: Pelatihan {model_name} selesai. Parameter terbaik: {grid_search.best_params_}")
        except ValueError as e:
            print(f"ERROR GridSearchCV {model_name}: {e}. Fallback tanpa GridSearchCV.")
            if not X_train.empty and not y_train.empty: best_model = rf.fit(X_train, y_train)
            else: print(f"ERROR: Data latih {model_name} kosong saat fallback."); return None
    if best_model:
        try: joblib.dump(best_model, model_save_path); print(f"INFO: {model_name} disimpan ke {model_save_path}")
        except Exception as e: print(f"ERROR: Gagal menyimpan {model_name} ke {model_save_path}: {e}")
    return best_model

# --- 3. Fungsi Pemuatan Model (Sama) ---
def load_model(model_path, model_name="Model"):
    if not os.path.exists(model_path): print(f"INFO: File model '{model_path}' tidak ditemukan."); return None
    try: model = joblib.load(model_path); print(f"INFO: {model_name} berhasil dimuat."); return model
    except Exception as e: print(f"ERROR saat memuat {model_name}: {e}"); return None

# --- 4. Fungsi Evaluasi Model Random Forest (Sama) ---
def evaluate_rf_models_all(model_lat, model_lon, model_mag, model_dep, 
                           X_test_rf, y_test_lat_rf, y_test_lon_rf, y_test_mag_rf, y_test_dep_rf, 
                           num_worst_errors=10, map_filename_base="evaluasi_peta", plot_filename_base="evaluasi_plot"):
    if not all([model_lat, model_lon, model_mag, model_dep]): print("INFO: Satu atau lebih model RF tidak tersedia untuk evaluasi."); return
    if X_test_rf.empty: print("INFO: Data uji RF kosong, evaluasi tidak dapat dilakukan."); return
    print("\n--- Evaluasi Model Random Forest pada Data Uji ---")
    current_features_for_eval = [f for f in FEATURES_RF if f in X_test_rf.columns]
    if len(current_features_for_eval) != len(FEATURES_RF):
        print(f"WARNING: Fitur di X_test_rf tidak cocok dengan FEATURES_RF global. Menggunakan fitur yang ada di X_test_rf.")
    y_pred_lat_rf = model_lat.predict(X_test_rf[current_features_for_eval])
    y_pred_lon_rf = model_lon.predict(X_test_rf[current_features_for_eval])
    y_pred_mag_rf = model_mag.predict(X_test_rf[current_features_for_eval])
    y_pred_dep_rf = model_dep.predict(X_test_rf[current_features_for_eval])
    mae_lat = mean_absolute_error(y_test_lat_rf, y_pred_lat_rf); rmse_lat = np.sqrt(mean_squared_error(y_test_lat_rf, y_pred_lat_rf))
    print(f"RF - MAE Latitude: {mae_lat:.4f}, RMSE Latitude: {rmse_lat:.4f}")
    mae_lon = mean_absolute_error(y_test_lon_rf, y_pred_lon_rf); rmse_lon = np.sqrt(mean_squared_error(y_test_lon_rf, y_pred_lon_rf))
    print(f"RF - MAE Longitude: {mae_lon:.4f}, RMSE Longitude: {rmse_lon:.4f}")
    haversine_errors = haversine_distance_func(y_test_lat_rf.values, y_test_lon_rf.values, y_pred_lat_rf, y_pred_lon_rf)
    print(f"RF - Rata-rata Haversine Distance: {np.mean(haversine_errors):.4f} km")
    print(f"RF - Median Haversine Distance: {np.median(haversine_errors):.4f} km")
    print(f"RF - Max Haversine Distance: {np.max(haversine_errors):.4f} km")
    mae_mag = mean_absolute_error(y_test_mag_rf, y_pred_mag_rf); rmse_mag = np.sqrt(mean_squared_error(y_test_mag_rf, y_pred_mag_rf))
    print(f"RF - MAE Magnitudo: {mae_mag:.4f}, RMSE Magnitudo: {rmse_mag:.4f}")
    mae_dep = mean_absolute_error(y_test_dep_rf, y_pred_dep_rf); rmse_dep = np.sqrt(mean_squared_error(y_test_dep_rf, y_pred_dep_rf))
    print(f"RF - MAE Kedalaman: {mae_dep:.2f} km, RMSE Kedalaman: {rmse_dep:.2f} km")
    error_analysis_df = X_test_rf.copy() 
    error_analysis_df['actual_lat'] = y_test_lat_rf; error_analysis_df['actual_lon'] = y_test_lon_rf
    error_analysis_df['predicted_lat'] = y_pred_lat_rf; error_analysis_df['predicted_lon'] = y_pred_lon_rf
    error_analysis_df['actual_mag'] = y_test_mag_rf; error_analysis_df['predicted_mag'] = y_pred_mag_rf
    error_analysis_df['actual_depth'] = y_test_dep_rf; error_analysis_df['predicted_depth'] = y_pred_dep_rf
    error_analysis_df['haversine_error_km'] = haversine_errors
    worst_predictions_df = error_analysis_df.sort_values(by='haversine_error_km', ascending=False)
    print(f"\n--- Analisis Detail untuk {num_worst_errors} Prediksi RF dengan Error Haversine Terbesar ---")
    print(f"Menampilkan {num_worst_errors} prediksi dengan error Haversine terbesar:")
    pd.set_option('display.max_columns', None); pd.set_option('display.width', 1000) 
    print(worst_predictions_df.head(num_worst_errors))
    pd.reset_option('display.max_columns'); pd.reset_option('display.width')
    map_center_lat = y_test_lat_rf.mean() if not y_test_lat_rf.empty else -2.5
    map_center_lon = y_test_lon_rf.mean() if not y_test_lon_rf.empty else 118.0
    eval_map = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=4)
    for idx, row in worst_predictions_df.head(num_worst_errors).iterrows():
        folium.CircleMarker(location=[row['actual_lat'], row['actual_lon']], radius=max(3, row['actual_mag'] * 1.5), color='blue', fill=True, fill_color='blue', fill_opacity=0.6, tooltip=f"Aktual (Terburuk {idx+1}): Lat={row['actual_lat']:.2f}, Lon={row['actual_lon']:.2f}, Mag={row['actual_mag']:.1f}, Dep={row['actual_depth']:.1f}km. Error: {row['haversine_error_km']:.2f}km").add_to(eval_map)
        folium.CircleMarker(location=[row['predicted_lat'], row['predicted_lon']], radius=max(3, row['predicted_mag'] * 1.5), color='red', fill=True, fill_color='red', fill_opacity=0.6, tooltip=f"Prediksi (Terburuk {idx+1}): Lat={row['predicted_lat']:.2f}, Lon={row['predicted_lon']:.2f}, Mag={row['predicted_mag']:.1f}, Dep={row['predicted_depth']:.1f}km").add_to(eval_map)
        folium.PolyLine([(row['actual_lat'], row['actual_lon']), (row['predicted_lat'], row['predicted_lon'])], color="black", weight=1, opacity=0.5).add_to(eval_map)
    best_predictions_df = error_analysis_df.sort_values(by='haversine_error_km', ascending=True)
    for idx, row in best_predictions_df.head(num_worst_errors).iterrows(): 
        folium.CircleMarker(location=[row['actual_lat'], row['actual_lon']], radius=max(3, row['actual_mag'] * 1.5), color='green', fill=True, fill_color='green', fill_opacity=0.6, tooltip=f"Aktual (Terbaik {idx+1}): Lat={row['actual_lat']:.2f}, Lon={row['actual_lon']:.2f}, Mag={row['actual_mag']:.1f}, Dep={row['actual_depth']:.1f}km. Error: {row['haversine_error_km']:.2f}km").add_to(eval_map)
        folium.CircleMarker(location=[row['predicted_lat'], row['predicted_lon']], radius=max(3, row['predicted_mag'] * 1.5), color='orange', fill=True, fill_color='orange', fill_opacity=0.6, tooltip=f"Prediksi (Terbaik {idx+1}): Lat={row['predicted_lat']:.2f}, Lon={row['predicted_lon']:.2f}, Mag={row['predicted_mag']:.1f}, Dep={row['predicted_depth']:.1f}km").add_to(eval_map)
        folium.PolyLine([(row['actual_lat'], row['actual_lon']), (row['predicted_lat'], row['predicted_lon'])], color="gray", weight=1, opacity=0.5).add_to(eval_map)
    map_file = f"{map_filename_base}_test_data.html"
    eval_map.save(map_file); print(f"INFO: Peta evaluasi data uji disimpan sebagai {map_file}")
    try: webbrowser.open(f"file://{os.path.realpath(map_file)}", new=2)
    except Exception as e: print(f"INFO: Tidak bisa membuka peta di browser: {e}")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    sns.histplot(haversine_errors, kde=True, bins=30, color='skyblue')
    plt.title(f'Distribusi Error Haversine\nRata-rata: {np.mean(haversine_errors):.2f} km, Median: {np.median(haversine_errors):.2f} km', fontsize=14)
    plt.xlabel('Haversine Error (km)', fontsize=12); plt.ylabel('Frekuensi', fontsize=12)
    hist_file = f"{plot_filename_base}_haversine_error_distribution.png"
    plt.savefig(hist_file); print(f"INFO: Histogram error Haversine disimpan sebagai {hist_file}"); plt.close()
    plt.figure(figsize=(8, 8))
    sns.regplot(x=y_test_mag_rf, y=y_pred_mag_rf, scatter_kws={'alpha':0.3, 's':50, 'color':'green'}, line_kws={'color':'red', 'linestyle':'--'})
    plt.plot([y_test_mag_rf.min(), y_test_mag_rf.max()], [y_test_mag_rf.min(), y_test_mag_rf.max()], 'k--', lw=2, label='Ideal (y=x)') 
    plt.title(f'Magnitudo: Aktual vs. Prediksi (MAE: {mae_mag:.3f})', fontsize=14)
    plt.xlabel('Magnitudo Aktual', fontsize=12); plt.ylabel('Magnitudo Prediksi', fontsize=12); plt.legend()
    scatter_mag_file = f"{plot_filename_base}_mag_actual_vs_predicted.png"
    plt.savefig(scatter_mag_file); print(f"INFO: Scatter plot magnitudo disimpan sebagai {scatter_mag_file}"); plt.close()
    plt.figure(figsize=(8, 8))
    sns.regplot(x=y_test_dep_rf, y=y_pred_dep_rf, scatter_kws={'alpha':0.3, 's':50, 'color':'purple'}, line_kws={'color':'red', 'linestyle':'--'})
    plt.plot([y_test_dep_rf.min(), y_test_dep_rf.max()], [y_test_dep_rf.min(), y_test_dep_rf.max()], 'k--', lw=2, label='Ideal (y=x)')
    plt.title(f'Kedalaman: Aktual vs. Prediksi (MAE: {mae_dep:.2f} km)', fontsize=14)
    plt.xlabel('Kedalaman Aktual (km)', fontsize=12); plt.ylabel('Kedalaman Prediksi (km)', fontsize=12); plt.legend()
    scatter_dep_file = f"{plot_filename_base}_depth_actual_vs_predicted.png"
    plt.savefig(scatter_dep_file); print(f"INFO: Scatter plot kedalaman disimpan sebagai {scatter_dep_file}"); plt.close()

# --- 5. Fungsi Prediksi Iteratif dengan Random Forest (Disesuaikan untuk Fitur v12/v15) ---
def prepare_rf_input_for_prediction_v15(prev_lon, prev_lat, prev_depth, prev_mag, 
                                        time_log_binned_features, 
                                        dist_km_val, 
                                        abs_depth_chg, mag_chg, 
                                        prediction_dt):
    global FEATURES_RF 
    year = prediction_dt.year; month = prediction_dt.month
    month_sin, month_cos = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
    
    data = {'year': [year], 'month_sin': [month_sin], 'month_cos': [month_cos],
            'lat_prev': [prev_lat], 'lon_prev': [prev_lon], 
            'depth_prev': [prev_depth], 'mag_prev': [prev_mag],
            'dist_to_prev_km': [dist_km_val], 
            'abs_depth_change': [abs_depth_chg], 
            'mag_change': [mag_chg]}
    input_df = pd.DataFrame(data)
    for col in time_log_binned_features.columns:
        input_df[col] = time_log_binned_features[col].values[0]
    for col in FEATURES_RF: 
        if col not in input_df.columns: input_df[col] = 0.0                  
    return input_df[FEATURES_RF] 

def predict_rf_next_n_events_v15(start_lon, start_lat, start_depth, start_mag, 
                                n_predictions, 
                                model_rf_lat, model_rf_lon, model_rf_mag, model_rf_dep, 
                                initial_dt_prediction, feature_stats,
                                map_filename_base="prediksi_interaktif_peta"): 
    try:
        time_log_encoder = joblib.load(WEB_TIME_ENCODER_PATH)
        print("INFO: Time encoder berhasil dimuat untuk prediksi.")
    except FileNotFoundError:
        print(f"KRITIS: File encoder '{WEB_TIME_ENCODER_PATH}' tidak ditemukan. Latih model terlebih dahulu.")
        # Kembalikan list kosong karena prediksi tidak bisa dilanjutkan
        return []
    except Exception as e:
        print(f"KRITIS: Gagal memuat time encoder: {e}")
        return []
     
    if not all([model_rf_lat, model_rf_lon, model_rf_mag, model_rf_dep]):
        print("ERROR: Satu atau lebih model RF tidak valid. Prediksi iteratif dibatalkan.")
        return []
    if feature_stats is None:
        print("ERROR: Statistik fitur tidak tersedia. Prediksi iteratif mungkin kurang akurat untuk langkah pertama.")
        feature_stats = {'median_abs_depth_change': 0.0, 'median_mag_change': 0.0, 'median_dist_to_prev_km': 0.0}
        
    predicted_events = []
    current_lat, current_lon = start_lat, start_lat
    current_depth, current_mag = start_depth, start_mag
    current_datetime = initial_dt_prediction
    
    interactive_map_center_lat = start_lat; interactive_map_center_lon = start_lon
    pred_map = folium.Map(location=[interactive_map_center_lat, interactive_map_center_lon], zoom_start=6)
    folium.Marker([start_lat, start_lon], popup=f"Input Awal: Lat={start_lat:.2f}, Lon={start_lon:.2f}, Mag={start_mag:.1f}, Dep={start_depth:.1f}km", icon=folium.Icon(color='purple', icon='star')).add_to(pred_map)
    print(f"\nINFO: Memprediksi {n_predictions} kejadian berikutnya dari (Lat:{start_lat:.2f}, Lon:{start_lon:.2f}, Dep:{start_depth:.1f}, Mag:{start_mag:.1f}):")
    path_coordinates = [[start_lat, start_lon]]
    
    iter_prev_lat, iter_prev_lon = start_lat, start_lat
    iter_prev_depth, iter_prev_mag = start_depth, start_mag
    iter_prev_datetime = initial_dt_prediction

    for i in range(n_predictions):
        prediction_dt_for_time_features = current_datetime 
        time_diff_h = (current_datetime - iter_prev_datetime).total_seconds() / 3600.0 if i > 0 else 1.0 
        time_log_val = np.log1p(max(0.000001, time_diff_h))
        
       # Buat bin untuk nilai waktu prediksi
        time_log_bin_pred_val = pd.cut(pd.Series([time_log_val]), bins=TIME_LOG_BINS, labels=TIME_LOG_BIN_LABELS, right=False, include_lowest=True)
        
        # Periksa apakah nilai berada di luar jangkauan bin
        if pd.isna(time_log_bin_pred_val.iloc[0]):
             print(f"WARNING: time_log_val ({time_log_val}) di luar rentang bin waktu. Fitur waktu OHE akan diisi nol.")
             # Buat array nol jika di luar jangkauan
             time_log_encoded_array_pred = np.zeros((1, len(time_log_encoder.get_feature_names_out(['time_log_bin']))))
        else:
            # Gunakan encoder yang sudah dimuat untuk mengubah data
            time_log_encoded_array_pred = time_log_encoder.transform(pd.DataFrame({'time_log_bin': time_log_bin_pred_val}))
            
        time_log_binned_input_features = pd.DataFrame(time_log_encoded_array_pred, columns=time_log_encoder.get_feature_names_out(['time_log_bin']))
        
        dist_km_val = haversine_distance_func(current_lat, current_lon, iter_prev_lat, iter_prev_lon) if i > 0 else feature_stats.get('median_dist_to_prev_km', 0.0)
        dist_km_val = min(dist_km_val, MAX_DIST_PREV_KM_THRESHOLD - 0.01 if MAX_DIST_PREV_KM_THRESHOLD is not None else dist_km_val )
        
        abs_depth_chg_val = abs(current_depth - iter_prev_depth) if i > 0 else feature_stats.get('median_abs_depth_change', 0.0)
        mag_chg_val = current_mag - iter_prev_mag if i > 0 else feature_stats.get('median_mag_change', 0.0)
        
        prev_lon_input = current_lon if i > 0 else start_lon 
        prev_lat_input = current_lat if i > 0 else start_lat
        prev_depth_input = current_depth if i > 0 else start_depth
        prev_mag_input = current_mag if i > 0 else start_mag
        
        input_features_df = prepare_rf_input_for_prediction_v15( # Menggunakan fungsi prepare yang benar
            prev_lon_input, prev_lat_input, prev_depth_input, prev_mag_input,
            time_log_binned_input_features, 
            dist_km_val, abs_depth_chg_val, mag_chg_val,
            prediction_dt_for_time_features
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                next_lat_pred = model_rf_lat.predict(input_features_df)[0]
                next_lon_pred = model_rf_lon.predict(input_features_df)[0]
                next_mag_pred = model_rf_mag.predict(input_features_df)[0] 
                next_dep_pred = model_rf_dep.predict(input_features_df)[0] 
        except Exception as e: print(f"ERROR prediksi RF iteratif ke-{i+1}: {e}"); return predicted_events
        
        event_data = {'lat': next_lat_pred, 'lon': next_lon_pred, 'depth': next_dep_pred, 'mag': next_mag_pred}
        predicted_events.append(event_data)
        print(f"  RF Prediksi {i+1}: Lat={next_lat_pred:.4f}, Lon={next_lon_pred:.4f}, Dep={next_dep_pred:.1f}, Mag={next_mag_pred:.1f}")
        
        folium.CircleMarker(location=[next_lat_pred, next_lon_pred], radius=max(3, next_mag_pred * 1.5), color='blue' if i % 2 == 0 else 'red', fill=True, fill_color='blue' if i % 2 == 0 else 'red', fill_opacity=0.7, tooltip=f"Prediksi {i+1}: Lat={next_lat_pred:.2f}, Lon={next_lon_pred:.2f}, Mag={next_mag_pred:.1f}, Dep={next_dep_pred:.1f}km").add_to(pred_map)
        path_coordinates.append([next_lat_pred, next_lon_pred])
        
        iter_prev_lat, iter_prev_lon = current_lat, current_lon
        iter_prev_depth, iter_prev_mag = current_depth, current_mag
        iter_prev_datetime = current_datetime
        
        current_lat, current_lon = next_lat_pred, next_lon_pred
        current_depth, current_mag = next_dep_pred, next_mag_pred
    
    if len(path_coordinates) > 1:
        folium.PolyLine(path_coordinates, color="purple", weight=2.5, opacity=1).add_to(pred_map)
    map_file_interactive = f"{map_filename_base}_langkah_prediksi.html"
    pred_map.save(map_file_interactive)
    print(f"INFO: Peta prediksi interaktif disimpan sebagai {map_file_interactive}")
    try: webbrowser.open(f"file://{os.path.realpath(map_file_interactive)}", new=2)
    except Exception as e: print(f"INFO: Tidak bisa membuka peta prediksi interaktif di browser: {e}")
    return predicted_events
    
# --- 6. Fungsi Utama (Main) ---
def main_combined_v15_binned_time(): # Nama fungsi diubah
    global time_encoder_fitted, FEATURES_RF 
    time_encoder_fitted = False 
    FEATURES_RF = FEATURES_RF_BASE.copy() 

    print(f"--- Sistem Soft Sensor Gempa v15 (Binning Waktu, Filter < {MAX_DIST_PREV_KM_THRESHOLD}km) ---") # Nama diubah
    feature_stats = None 
    # PERBAIKAN: force_retrain_rf didefinisikan sebelum digunakan dalam blok try
    force_retrain_rf = input(f"Latih ulang SEMUA model RF (4 model, binning waktu, filter <{MAX_DIST_PREV_KM_THRESHOLD}km, SANGAT LAMA)? (y/n, default n): ").lower() == 'y'

    if os.path.exists(FEATURE_STATS_PATH) and not force_retrain_rf : # Hanya muat statistik jika tidak force retrain
        try:
            feature_stats = joblib.load(FEATURE_STATS_PATH)
            print(f"INFO: Statistik fitur berhasil dimuat dari {FEATURE_STATS_PATH}")
        except Exception as e:
            print(f"WARNING: Gagal memuat statistik fitur dari {FEATURE_STATS_PATH}: {e}.")
            feature_stats = None # Pastikan None jika gagal muat

    try:
        X_rf_all, y_rf_lat_all, y_rf_lon_all, y_rf_mag_all, y_rf_dep_all, calculated_feature_stats = load_and_preprocess_data_all_targets(
            dist_threshold=MAX_DIST_PREV_KM_THRESHOLD
        )
        # Update feature_stats jika baru dihitung (calculated_feature_stats tidak None)
        # atau jika kita memaksa retrain (karena statistik mungkin perlu diupdate juga)
        if calculated_feature_stats and (feature_stats is None or force_retrain_rf): 
             feature_stats = calculated_feature_stats
    except (FileNotFoundError, ValueError) as e:
        print(f"KRITIS: Gagal memuat/memproses data awal: {e}. Program berhenti.")
        return

    if X_rf_all.empty:
        print("KRITIS: Tidak ada data valid untuk RF setelah pra-pemrosesan. Program berhenti.")
        return
    
    actual_features_in_X_all = [f for f in FEATURES_RF if f in X_rf_all.columns]
    if len(actual_features_in_X_all) != len(FEATURES_RF):
        missing_from_X_all = set(FEATURES_RF) - set(X_rf_all.columns)
        print(f"KRITIS: Fitur berikut hilang dari X_rf_all setelah OHE (sebelum split): {missing_from_X_all}")
        return
    
    X_train_rf, X_test_rf, \
    y_train_lat_rf, y_test_lat_rf, \
    y_train_lon_rf, y_test_lon_rf, \
    y_train_mag_rf, y_test_mag_rf, \
    y_train_dep_rf, y_test_dep_rf = train_test_split(
        X_rf_all[FEATURES_RF], y_rf_lat_all, y_rf_lon_all, y_rf_mag_all, y_rf_dep_all,
        test_size=0.2, random_state=42, shuffle=False)
    
    print(f"INFO: Ukuran data latih RF: {len(X_train_rf)}, data uji RF: {len(X_test_rf)}")
    if len(X_train_rf) == 0:
        print("KRITIS: Data latih RF kosong setelah split. Tidak bisa melanjutkan.")
        return
    
    model_rf_lat = load_model(MODEL_RF_LAT_PATH, f"Model RF Lat (v15_dist<{MAX_DIST_PREV_KM_THRESHOLD}km_binned_time)")
    if model_rf_lat is None or force_retrain_rf:
        model_rf_lat = train_random_forest_model_tuned(X_train_rf, y_train_lat_rf, MODEL_RF_LAT_PATH, f"Model RF Lat (v15_dist<{MAX_DIST_PREV_KM_THRESHOLD}km_binned_time)")

    model_rf_lon = load_model(MODEL_RF_LON_PATH, f"Model RF Lon (v15_dist<{MAX_DIST_PREV_KM_THRESHOLD}km_binned_time)")
    if model_rf_lon is None or force_retrain_rf:
        model_rf_lon = train_random_forest_model_tuned(X_train_rf, y_train_lon_rf, MODEL_RF_LON_PATH, f"Model RF Lon (v15_dist<{MAX_DIST_PREV_KM_THRESHOLD}km_binned_time)")

    model_rf_mag = load_model(MODEL_RF_MAG_PATH, f"Model RF Mag (v15_dist<{MAX_DIST_PREV_KM_THRESHOLD}km_binned_time)")
    if model_rf_mag is None or force_retrain_rf:
        model_rf_mag = train_random_forest_model_tuned(X_train_rf, y_train_mag_rf, MODEL_RF_MAG_PATH, f"Model RF Mag (v15_dist<{MAX_DIST_PREV_KM_THRESHOLD}km_binned_time)")

    model_rf_dep = load_model(MODEL_RF_DEP_PATH, f"Model RF Dep (v15_dist<{MAX_DIST_PREV_KM_THRESHOLD}km_binned_time)")
    if model_rf_dep is None or force_retrain_rf:
        model_rf_dep = train_random_forest_model_tuned(X_train_rf, y_train_dep_rf, MODEL_RF_DEP_PATH, f"Model RF Dep (v15_dist<{MAX_DIST_PREV_KM_THRESHOLD}km_binned_time)")

    if all([model_rf_lat, model_rf_lon, model_rf_mag, model_rf_dep]):
        evaluate_rf_models_all(model_rf_lat, model_rf_lon, model_rf_mag, model_rf_dep,
                               X_test_rf, y_test_lat_rf, y_test_lon_rf, y_test_mag_rf, y_test_dep_rf, 
                               num_worst_errors=10, 
                               map_filename_base=f"evaluasi_peta_v15_dist{MAX_DIST_PREV_KM_THRESHOLD}_binned_time",
                               plot_filename_base=f"evaluasi_plot_v15_dist{MAX_DIST_PREV_KM_THRESHOLD}_binned_time") 
    else:
        print("KRITIS: Satu atau lebih model RF tidak berhasil dimuat atau dilatih.")

    print("\n--- Interaksi Pengguna ---")
    if all([model_rf_lat, model_rf_lon, model_rf_mag, model_rf_dep]):
        
        ## PERUBAHAN DIMULAI: Membungkus logika prediksi dalam loop while True ##
        while True: 
            print("\nOpsi: Prediksi 5 Kejadian Gempa Berikutnya (Lokasi, Mag, Dep)")
            if input("Lakukan prediksi RF? (y/n, default y): ").lower() != 'n':
                if feature_stats is None: 
                    print("PERINGATAN: Statistik fitur tidak tersedia untuk inisialisasi prediksi pertama yang lebih baik.")
                    feature_stats_for_pred = {'median_abs_depth_change': 0.0, 'median_mag_change': 0.0, 'median_dist_to_prev_km': 0.0}
                else:
                    feature_stats_for_pred = feature_stats
                
                # Loop untuk memastikan input pengguna valid sebelum melanjutkan
                while True:
                    try:
                        in_lat = float(input("  Latitude gempa terakhir (mis: -6.82): "))
                        in_lon = float(input("  Longitude gempa terakhir (mis: 107.60): "))
                        in_depth = float(input("  Kedalaman gempa terakhir (mis: 10.0): "))
                        in_mag = float(input("  Magnitudo gempa terakhir (mis: 4.5): "))
                        in_datetime_str = input("  Tanggal & Waktu gempa terakhir (YYYY-MM-DD HH:MM:SS, mis: 2023-01-01 10:00:00): ")
                        in_datetime = pd.to_datetime(in_datetime_str)
                        break # Keluar dari loop input jika semua valid
                    except ValueError: 
                        print("  Input tidak valid. Masukkan angka atau format tanggal yang benar.")
                
                predict_rf_next_n_events_v15(in_lon, in_lat, in_depth, in_mag, 5, 
                                            model_rf_lat, model_rf_lon, model_rf_mag, model_rf_dep, 
                                            in_datetime, feature_stats_for_pred, 
                                            map_filename_base=f"prediksi_interaktif_v15_dist{MAX_DIST_PREV_KM_THRESHOLD}_binned_time")
            else:
                # Jika pengguna memilih 'n' untuk prediksi, keluar dari loop utama
                break

            # Tanyakan apakah pengguna ingin melakukan prediksi lagi
            if input("\nLakukan prediksi lagi? (y/n, default n): ").lower() != 'y':
                break # Keluar dari loop while True jika jawaban bukan 'y'
        ## PERUBAHAN SELESAI ##

    else:
        print("INFO: Model RF tidak tersedia sepenuhnya, prediksi tidak dapat dilakukan.")
            
    print("\n--- Selesai ---")

if __name__ == "__main__":
    main_combined_v15_binned_time()
