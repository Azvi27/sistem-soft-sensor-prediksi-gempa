// File: src/components/ResultsColumn.jsx

import React from 'react';

function ResultsColumn({ predictionResult, evaluationResult, activeTab }) {

  // Fungsi untuk merender konten "Selamat Datang"
  const renderWelcome = () => (
    <div className="section">
      <h2 className="section-title">Selamat Datang</h2>
      <p>Gunakan panel di kiri untuk mengevaluasi atau membuat prediksi dengan model.</p>
      <p>Hasil akan muncul di sini.</p>
    </div>
  );

  // Fungsi untuk merender konten "Hasil Evaluasi"
  const renderEvaluation = (evalData) => {
    // Jika tidak ada data evaluasi, tampilkan pesan
    if (!evalData) {
      return (
        <div className="section"><p>Belum ada hasil evaluasi. Silakan tekan tombol 'Evaluasi' terlebih dahulu.</p></div>
      );
    }
    const { metrics, plots, error_analysis } = evalData;
    const metricLabels = {
      "mean_haversine_error_km": "Rata-rata Error Haversine", "median_haversine_error_km": "Median Error Haversine",
      "max_haversine_error_km": "Max Error Haversine", "mae_lat": "MAE Latitude", "rmse_lat": "RMSE Latitude",
      "mae_lon": "MAE Longitude", "rmse_lon": "RMSE Longitude", "mae_mag": "MAE Magnitudo", "rmse_mag": "RMSE Magnitudo",
      "mae_dep": "MAE Kedalaman", "rmse_dep": "RMSE Kedalaman"
    };
    const plotTitles = { 
        "depth_scatter": "Kedalaman: Aktual vs Prediksi", 
        "magnitude_scatter": "Magnitudo: Aktual vs Prediksi",
        "haversine_distribution": "Distribusi Error Haversine"
    };
    return (
      <div className="section">
        <h2 className="section-title">Hasil Evaluasi Model</h2>
        <h3 className="sub-section-title">Metrik Kinerja</h3>
        <div className="evaluation-metrics-grid">
          {Object.entries(metricLabels).map(([key, label]) =>
            metrics[key] ? (<div key={key} className="metric-card"><h4 className="metric-card-title">{label}</h4><p className="metric-card-value">{metrics[key]}</p></div>) : null
          )}
        </div>
        <h3 className="sub-section-title">Visualisasi Plot</h3>
        <div className="plots-grid">
            {plots.depth_scatter && <div className="plot-card"><h4>{plotTitles.depth_scatter}</h4><img src={`http://127.0.0.1:5000${plots.depth_scatter}?t=${new Date().getTime()}`} alt={plotTitles.depth_scatter}/></div>}
            {plots.magnitude_scatter && <div className="plot-card"><h4>{plotTitles.magnitude_scatter}</h4><img src={`http://127.0.0.1:5000${plots.magnitude_scatter}?t=${new Date().getTime()}`} alt={plotTitles.magnitude_scatter}/></div>}
            {plots.haversine_distribution && (<div className="plot-card full-width"><h4>{plotTitles.haversine_distribution}</h4><img src={`http://127.0.0.1:5000${plots.haversine_distribution}?t=${new Date().getTime()}`} alt={plotTitles.haversine_distribution}/></div>)}
        </div>
        <h3 className="sub-section-title">Peta Evaluasi</h3>
        <iframe src={`http://127.0.0.1:5000${plots.evaluation_map}?t=${new Date().getTime()}`} title="Peta Evaluasi" style={{width: '100%', height: '500px', border: 0}}></iframe>
        {error_analysis && error_analysis.length > 0 && (
          <><h3 className="sub-section-title">Analisis Error Terbesar</h3><div className="table-container"><table><thead><tr>{Object.keys(error_analysis[0]).map(key => <th key={key}>{key.replace(/_/g, ' ')}</th>)}</tr></thead><tbody>{error_analysis.map((row, index) => (<tr key={index}>{Object.values(row).map((val, i) => <td key={i}>{typeof val === 'number' ? val.toFixed(4) : val}</td>)}</tr>))}</tbody></table></div></>
        )}
      </div>
    );
  };

  // Fungsi untuk merender konten "Hasil Prediksi"
  const renderPrediction = (predData) => {
    // Jika tidak ada data prediksi, tampilkan pesan
    if (!predData) {
      return (
        <div className="section"><p>Belum ada hasil prediksi. Silakan isi form dan buat prediksi.</p></div>
      );
    }
    return (
    <div>
        <div className="section">
            <h2 className="section-title">Hasil Prediksi</h2>
            {predData.predictions.length > 0 ? (
                <ul style={{paddingLeft: 0, listStyle: 'none'}}>
                    {predData.predictions.map((p, index) => (<li key={index} className="prediction-item"><strong>Prediksi {index + 1}:</strong> Lat: {p.lat.toFixed(4)}, Lon: {p.lon.toFixed(4)}, Kedalaman: {p.depth.toFixed(1)} km, Magnitudo: {p.mag.toFixed(1)}</li>))}
                </ul>
            ) : (<p>Model tidak menghasilkan prediksi untuk input yang diberikan.</p>)}
        </div>
        {predData.map_url && (
            <div className="section">
                <h3 className="sub-section-title">Peta Jalur Prediksi</h3>
                <iframe src={`http://127.0.0.1:5000${predData.map_url}?t=${new Date().getTime()}`} title="Peta Prediksi" style={{width: '100%', height: '500px', border: 0}}></iframe>
            </div>
        )}
    </div>
  )};

  // Ini adalah satu-satunya 'return' di level utama, yang menangani logika tab.
  return (
    <div className="tab-content-wrapper">
      <div className={`tab-content ${activeTab === 'Selamat Datang' ? 'active' : ''}`}>
        {renderWelcome()}
      </div>
      <div className={`tab-content ${activeTab === 'Hasil Evaluasi' ? 'active' : ''}`}>
        {renderEvaluation(evaluationResult)}
      </div>
      <div className={`tab-content ${activeTab === 'Hasil Prediksi' ? 'active' : ''}`}>
        {renderPrediction(predictionResult)}
      </div>
    </div>
  );
}

export default ResultsColumn;