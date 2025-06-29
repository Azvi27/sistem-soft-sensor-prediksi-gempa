// File: src/components/PredictionForm.jsx

import React, { useState } from 'react';
// --- TAMBAHAN 1: Impor Flatpickr ---
import Flatpickr from 'react-flatpickr';

// --- TAMBAHAN 2: Impor tema CSS untuk Flatpickr ---
import "flatpickr/dist/themes/dark.css";


function PredictionForm({ onPredictionSuccess }) {
  const [formData, setFormData] = useState({
    lat: '',
    lon: '',
    depth: '',
    mag: '',
    // Beri nilai awal tanggal & waktu saat ini agar tidak kosong
    datetime: new Date() 
  });
  const [isPredicting, setIsPredicting] = useState(false);

  // --- PERUBAHAN 3: Modifikasi handleInputChange untuk menangani Flatpickr ---
  const handleInputChange = (e) => {
    // Cek jika ini adalah event dari Flatpickr
    if (e instanceof Date) {
        setFormData(prevState => ({
            ...prevState,
            datetime: e
        }));
    } else { // Jika tidak, ini adalah event dari input biasa
        const { name, value } = e.target;
        setFormData(prevState => ({
            ...prevState,
            [name]: value
        }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsPredicting(true);

    // Format tanggal menjadi string Y-m-d H:i:S sebelum dikirim
    const dataToSubmit = {
        ...formData,
        datetime: formData.datetime.toISOString().slice(0, 19).replace('T', ' '),
    };

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(dataToSubmit),
      });
      const result = await response.json();
      if (!response.ok) { throw new Error(result.message || 'Terjadi kesalahan.'); }
      onPredictionSuccess(result);
    } catch (error) {
      alert(`Gagal membuat prediksi: ${error.message}`);
    } finally {
      setIsPredicting(false);
    }
  };
 
  return (
    <div className="section">
      <h2 className="section-title">Input Prediksi</h2>
      <form onSubmit={handleSubmit}>
        <div className="input-group">
            <div><label htmlFor="lat">Latitude</label><input type="number" step="any" id="lat" name="lat" required value={formData.lat} onChange={handleInputChange} /></div>
            <div><label htmlFor="lon">Longitude</label><input type="number" step="any" id="lon" name="lon" required value={formData.lon} onChange={handleInputChange} /></div>
            <div><label htmlFor="depth">Kedalaman (km)</label><input type="number" step="any" id="depth" name="depth" required value={formData.depth} onChange={handleInputChange} /></div>
            <div><label htmlFor="mag">Magnitudo</label><input type="number" step="any" id="mag" name="mag" required value={formData.mag} onChange={handleInputChange} /></div>
        </div>
        <div className="full-width-input">
          <label htmlFor="datetime">Tanggal & Waktu</label>
          {/* --- PERUBAHAN 4: Ganti input teks dengan komponen Flatpickr --- */}
          <Flatpickr
            data-enable-time
            data-enable-seconds
            value={formData.datetime}
            onChange={([date]) => handleInputChange(date)}
            options={{
                dateFormat: "Y-m-d H:i:S",
                time_24hr: true,
            }}
            className="flatpickr-input" // Beri kelas agar bisa di-style jika perlu
          />
        </div>
        
        <button type="submit" style={{ marginTop: '15px', width: '100%' }} disabled={isPredicting}>
          {isPredicting ? 'Memprediksi...' : 'Buat Prediksi'}
        </button>
      </form>
    </div>
  );
}

export default PredictionForm;