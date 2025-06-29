import os
import warnings
import pandas as pd
import numpy as np
from flask_cors import CORS
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, url_for

# Impor skrip utama Anda sebagai modul
# Pastikan nama file "sistem_gempa_gabungan_v15_binned_time.py" sudah benar
try:
    import sistem_gempa_gabungan_v15_binned_time as gempa_system
except ImportError:
    print("KRITIS: Gagal mengimpor 'sistem_gempa_gabungan_v15_binned_time.py'. Pastikan file ada di direktori yang sama dan tidak ada error sintaks.")
    exit()


# Inisialisasi Aplikasi Flask
app = Flask(__name__, static_folder='static')
CORS(app)

# --- Fungsi Helper ---

def check_model_files_exist():
    """Mengecek apakah semua file model .pkl yang dibutuhkan ada."""
    model_paths = [
        gempa_system.MODEL_RF_LAT_PATH,
        gempa_system.MODEL_RF_LON_PATH,
        gempa_system.MODEL_RF_MAG_PATH,
        gempa_system.MODEL_RF_DEP_PATH
    ]
    return all(os.path.exists(p) for p in model_paths)

def get_feature_stats():
    """Memuat statistik fitur yang dibutuhkan untuk prediksi."""
    if os.path.exists(gempa_system.FEATURE_STATS_PATH):
        return gempa_system.joblib.load(gempa_system.FEATURE_STATS_PATH)
    return None

def _run_evaluation(models, test_data):
    """Menjalankan evaluasi, membuat visualisasi, dan memformat hasilnya untuk frontend."""
    model_lat, model_lon, model_mag, model_dep = models
    X_test_rf, y_test_lat_rf, y_test_lon_rf, y_test_mag_rf, y_test_dep_rf = test_data

    if X_test_rf.empty:
        return None

    # Hasilkan prediksi
    y_pred_lat_rf = model_lat.predict(X_test_rf)
    y_pred_lon_rf = model_lon.predict(X_test_rf)
    y_pred_mag_rf = model_mag.predict(X_test_rf)
    y_pred_dep_rf = model_dep.predict(X_test_rf)
    haversine_errors = gempa_system.haversine_distance_func(y_test_lat_rf.values, y_test_lon_rf.values, y_pred_lat_rf, y_pred_lon_rf)
    
    # 1. Hitung Metrik
    metrics = {
        "mae_lat": f"{gempa_system.mean_absolute_error(y_test_lat_rf, y_pred_lat_rf):.4f}",
        "rmse_lat": f"{np.sqrt(gempa_system.mean_squared_error(y_test_lat_rf, y_pred_lat_rf)):.4f}",
        "mae_lon": f"{gempa_system.mean_absolute_error(y_test_lon_rf, y_pred_lon_rf):.4f}",
        "rmse_lon": f"{np.sqrt(gempa_system.mean_squared_error(y_test_lon_rf, y_pred_lon_rf)):.4f}",
        "mae_mag": f"{gempa_system.mean_absolute_error(y_test_mag_rf, y_pred_mag_rf):.4f}",
        "rmse_mag": f"{np.sqrt(gempa_system.mean_squared_error(y_test_mag_rf, y_pred_mag_rf)):.4f}",
        "mae_dep": f"{gempa_system.mean_absolute_error(y_test_dep_rf, y_pred_dep_rf):.2f}",
        "rmse_dep": f"{np.sqrt(gempa_system.mean_squared_error(y_test_dep_rf, y_pred_dep_rf)):.2f}",
        "mean_haversine_error_km": f"{np.mean(haversine_errors):.2f} km",
        "median_haversine_error_km": f"{np.median(haversine_errors):.2f} km",
        "max_haversine_error_km": f"{np.max(haversine_errors):.2f} km",
    }

    # 2. Siapkan DataFrame untuk analisis
    error_analysis_df = X_test_rf.copy()
    for col, data in {'actual_lat': y_test_lat_rf, 'actual_lon': y_test_lon_rf, 'predicted_lat': y_pred_lat_rf, 'predicted_lon': y_pred_lon_rf, 'actual_mag': y_test_mag_rf, 'predicted_mag': y_pred_mag_rf, 'actual_depth': y_test_dep_rf, 'predicted_depth': y_pred_dep_rf, 'haversine_error_km': haversine_errors}.items():
        error_analysis_df[col] = data
    worst_predictions_df = error_analysis_df.sort_values(by='haversine_error_km', ascending=False)
    best_predictions_df = error_analysis_df.sort_values(by='haversine_error_km', ascending=True)

    # 3. Buat dan Simpan Visualisasi
    os.makedirs(app.static_folder, exist_ok=True)
    timestamp = int(datetime.now().timestamp())
    map_filename = f"evaluation_map_{timestamp}.html"
    haversine_plot_filename = f"plot_haversine_{timestamp}.png"
    magnitude_plot_filename = f"plot_magnitude_{timestamp}.png"
    depth_plot_filename = f"plot_depth_{timestamp}.png"

    # Buat peta evaluasi dengan titik-titiknya
    map_center_lat = y_test_lat_rf.mean() if not y_test_lat_rf.empty else -2.5
    map_center_lon = y_test_lon_rf.mean() if not y_test_lon_rf.empty else 118.0
    eval_map = gempa_system.folium.Map(location=[map_center_lat, map_center_lon], zoom_start=4)
    
    for _, row in worst_predictions_df.head(10).iterrows():
        gempa_system.folium.CircleMarker(location=[row['actual_lat'], row['actual_lon']], radius=max(3, row['actual_mag'] * 1.5), color='blue', fill=True, fill_opacity=0.6, tooltip=f"Aktual (Error Terburuk): {row['haversine_error_km']:.2f}km").add_to(eval_map)
        gempa_system.folium.CircleMarker(location=[row['predicted_lat'], row['predicted_lon']], radius=max(3, row['predicted_mag'] * 1.5), color='red', fill=True, fill_opacity=0.6, tooltip=f"Prediksi (Error Terburuk): {row['haversine_error_km']:.2f}km").add_to(eval_map)
        gempa_system.folium.PolyLine([(row['actual_lat'], row['actual_lon']), (row['predicted_lat'], row['predicted_lon'])], color="black", weight=1).add_to(eval_map)
    
    for _, row in best_predictions_df.head(10).iterrows(): 
        gempa_system.folium.CircleMarker(location=[row['actual_lat'], row['actual_lon']], radius=max(3, row['actual_mag'] * 1.5), color='green', fill=True, fill_opacity=0.6, tooltip=f"Aktual (Error Terbaik): {row['haversine_error_km']:.2f}km").add_to(eval_map)
    
    eval_map.save(os.path.join(app.static_folder, map_filename))

    # Buat plot
    gempa_system.plt.style.use('seaborn-v0_8-whitegrid')
    gempa_system.plt.figure(figsize=(10, 6)); gempa_system.sns.histplot(haversine_errors, kde=True, bins=30); gempa_system.plt.title('Distribusi Error Haversine'); gempa_system.plt.savefig(os.path.join(app.static_folder, haversine_plot_filename)); gempa_system.plt.close()
    gempa_system.plt.figure(figsize=(8, 8)); gempa_system.sns.regplot(x=y_test_mag_rf, y=y_pred_mag_rf); gempa_system.plt.title('Magnitudo: Aktual vs. Prediksi'); gempa_system.plt.savefig(os.path.join(app.static_folder, magnitude_plot_filename)); gempa_system.plt.close()
    gempa_system.plt.figure(figsize=(8, 8)); gempa_system.sns.regplot(x=y_test_dep_rf, y=y_pred_dep_rf); gempa_system.plt.title('Kedalaman: Aktual vs. Prediksi'); gempa_system.plt.savefig(os.path.join(app.static_folder, depth_plot_filename)); gempa_system.plt.close()
    
    plots = {
        "evaluation_map": url_for('static', filename=map_filename),
        "haversine_distribution": url_for('static', filename=haversine_plot_filename),
        "magnitude_scatter": url_for('static', filename=magnitude_plot_filename),
        "depth_scatter": url_for('static', filename=depth_plot_filename),
    }
    
    # 4. Format tabel analisis error
    error_analysis_list = worst_predictions_df.head(10)[['lat_prev', 'lon_prev', 'mag_prev', 'actual_lat', 'actual_lon', 'predicted_lat', 'predicted_lon', 'haversine_error_km']].round(4).to_dict(orient='records')
    
    return {"metrics": metrics, "plots": plots, "error_analysis": error_analysis_list}


# --- Endpoint API ---

@app.route('/')
def index():
    """Menyajikan file index.html utama."""
    return send_from_directory('.', 'index.html')

@app.route('/status')
def status():
    """Mengecek apakah model sudah dilatih dan siap digunakan."""
    if check_model_files_exist():
        return jsonify({'ready': True, 'message': 'Model Siap'})
    else:
        return jsonify({'ready': False, 'message': 'Model Belum Dilatih'})

@app.route('/train', methods=['POST'])
def train_model():
    """Endpoint untuk memicu proses pelatihan model."""
    try:
        X_rf_all, y_rf_lat_all, y_rf_lon_all, y_rf_mag_all, y_rf_dep_all, _ = \
            gempa_system.load_and_preprocess_data_all_targets()
        
        X_train_rf, X_test_rf, y_train_lat_rf, y_test_lat_rf, y_train_lon_rf, y_test_lon_rf, \
        y_train_mag_rf, y_test_mag_rf, y_train_dep_rf, y_test_dep_rf = gempa_system.train_test_split(
            X_rf_all, y_rf_lat_all, y_rf_lon_all, y_rf_mag_all, y_rf_dep_all,
            test_size=0.2, random_state=42, shuffle=False)
        
        models = [
            gempa_system.train_random_forest_model_tuned(X_train_rf, y_train_lat_rf, gempa_system.MODEL_RF_LAT_PATH, "Model RF Lat"),
            gempa_system.train_random_forest_model_tuned(X_train_rf, y_train_lon_rf, gempa_system.MODEL_RF_LON_PATH, "Model RF Lon"),
            gempa_system.train_random_forest_model_tuned(X_train_rf, y_train_mag_rf, gempa_system.MODEL_RF_MAG_PATH, "Model RF Mag"),
            gempa_system.train_random_forest_model_tuned(X_train_rf, y_train_dep_rf, gempa_system.MODEL_RF_DEP_PATH, "Model RF Dep")
        ]

        if not all(models):
            return jsonify({'message': 'Pelatihan gagal, satu atau lebih model tidak dapat dibuat.'}), 500

        evaluation_results = _run_evaluation(models, (X_test_rf, y_test_lat_rf, y_test_lon_rf, y_test_mag_rf, y_test_dep_rf))
        return jsonify({'message': 'Pelatihan dan evaluasi selesai!', 'evaluation': evaluation_results})

    except Exception as e:
        return jsonify({'message': f'Terjadi error saat pelatihan: {str(e)}'}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    """Endpoint untuk mengevaluasi model yang sudah ada."""
    if not check_model_files_exist():
        return jsonify({'message': 'Model tidak ditemukan. Latih model terlebih dahulu.'}), 404
    try:
        models = [
            gempa_system.load_model(gempa_system.MODEL_RF_LAT_PATH),
            gempa_system.load_model(gempa_system.MODEL_RF_LON_PATH),
            gempa_system.load_model(gempa_system.MODEL_RF_MAG_PATH),
            gempa_system.load_model(gempa_system.MODEL_RF_DEP_PATH)
        ]
        
        X_rf_all, y_rf_lat_all, y_rf_lon_all, y_rf_mag_all, y_rf_dep_all, _ = \
            gempa_system.load_and_preprocess_data_all_targets()
        
        _, X_test_rf, _, y_test_lat_rf, _, y_test_lon_rf, _, y_test_mag_rf, _, y_test_dep_rf = \
            gempa_system.train_test_split(X_rf_all, y_rf_lat_all, y_rf_lon_all, y_rf_mag_all, y_rf_dep_all, test_size=0.2, random_state=42, shuffle=False)
        
        evaluation_results = _run_evaluation(models, (X_test_rf, y_test_lat_rf, y_test_lon_rf, y_test_mag_rf, y_test_dep_rf))
        return jsonify({'message': 'Evaluasi Selesai', 'evaluation': evaluation_results})

    except Exception as e:
        return jsonify({'message': f'Terjadi error saat evaluasi: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict_events():
    """Endpoint untuk memprediksi kejadian gempa berikutnya."""
    if not check_model_files_exist():
        return jsonify({'message': 'Model tidak ditemukan. Latih model terlebih dahulu.'}), 404
    try:
        data = request.get_json()
        in_lat, in_lon, in_depth, in_mag = float(data['lat']), float(data['lon']), float(data['depth']), float(data['mag'])
        in_datetime = pd.to_datetime(data['datetime'])

        models = [gempa_system.load_model(p) for p in [gempa_system.MODEL_RF_LAT_PATH, gempa_system.MODEL_RF_LON_PATH, gempa_system.MODEL_RF_MAG_PATH, gempa_system.MODEL_RF_DEP_PATH]]
        feature_stats = get_feature_stats()

        if not all(models) or not feature_stats:
            return jsonify({'message': 'Gagal memuat semua komponen model atau statistik fitur.'}), 500

        os.makedirs(app.static_folder, exist_ok=True)
        map_filename = f"prediction_map_{int(datetime.now().timestamp())}.html"
        # Nama dasar untuk disimpan, tanpa ekstensi
        map_save_path_base = os.path.join(app.static_folder, os.path.splitext(map_filename)[0])

        predictions = gempa_system.predict_rf_next_n_events_v15(
            start_lon=in_lon, start_lat=in_lat, start_depth=in_depth, start_mag=in_mag, n_predictions=5,
            model_rf_lat=models[0], model_rf_lon=models[1], model_rf_mag=models[2], model_rf_dep=models[3],
            initial_dt_prediction=in_datetime, feature_stats=feature_stats, map_filename_base=map_save_path_base
        )
        
        # Buat URL berdasarkan nama file yang disimpan oleh fungsi predict
        final_map_filename = f"{os.path.basename(map_save_path_base)}_langkah_prediksi.html"
        map_url = url_for('static', filename=final_map_filename)
        return jsonify({'predictions': predictions, 'map_url': map_url})
        
    except Exception as e:
        return jsonify({'message': f'Terjadi error saat prediksi: {str(e)}'}), 500

# --- Eksekusi Utama ---
if __name__ == '__main__':
    # Pastikan folder static ada saat aplikasi pertama kali dijalankan
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, port=5000)
