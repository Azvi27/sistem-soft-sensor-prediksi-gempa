// File: src/App.jsx

import { useState, useEffect } from 'react';
import ModelControl from './components/ModelControl';
import PredictionForm from './components/PredictionForm';
import ResultsColumn from './components/ResultsColumn';
import Tabs from './components/Tabs';
import './App.css';

function App() {
  const [modelStatus, setModelStatus] = useState('Menghubungi server...');
  const [isModelReady, setIsModelReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false); // Diubah ke false agar tidak loading terus
  
  const [predictionResult, setPredictionResult] = useState(null);
  const [evaluationResult, setEvaluationResult] = useState(null);
  const [activeTab, setActiveTab] = useState('Selamat Datang');

  const fetchStatus = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/status');
      if (!response.ok) throw new Error('Gagal mendapatkan respons.');
      const data = await response.json();
      setModelStatus(data.message);
      setIsModelReady(data.ready);
    } catch (error) {
      setModelStatus('Gagal terhubung ke server.');
      setIsModelReady(false);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const handlePrediction = (result) => {
    // PERBAIKAN: JANGAN hapus hasil evaluasi
    // setEvaluationResult(null); <-- BARIS INI DIHAPUS
    setPredictionResult(result);
    setActiveTab('Hasil Prediksi');
  };

  const handleEvaluate = async () => {
    setIsLoading(true);
    // PERBAIKAN: JANGAN hapus hasil prediksi
    // setPredictionResult(null); <-- BARIS INI DIHAPUS
    try {
      const response = await fetch('http://127.0.0.1:5000/evaluate', { method: 'POST' });
      const result = await response.json();
      if (!response.ok) throw new Error(result.message || 'Gagal evaluasi.');
      setEvaluationResult(result.evaluation);
      setActiveTab('Hasil Evaluasi');
    } catch (error) {
      alert(`Error saat evaluasi: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="dashboard-layout">
      <header><h1>Dashboard Prediksi Gempa</h1></header>
      <div id="controls-column">
        <ModelControl 
          status={modelStatus} 
          isReady={isModelReady} 
          isLoading={isLoading}
          onEvaluate={handleEvaluate} 
        />
        <PredictionForm onPredictionSuccess={handlePrediction} />
      </div>
      <div id="results-column">
        <Tabs activeTab={activeTab} setActiveTab={setActiveTab} />
        <ResultsColumn 
          predictionResult={predictionResult} 
          evaluationResult={evaluationResult}
          activeTab={activeTab} 
        />
      </div>
    </div>
  );
}

export default App;