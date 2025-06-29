import React from 'react';

function ModelControl({ status, isReady, isLoading, onEvaluate }) {
  const getStatusClass = () => {
    if (isLoading) return 'status-training';
    return isReady ? 'status-ready' : 'status-not-ready';
  };
  return (
    <div className="section">
      <h2 className="section-title">Kontrol Model</h2>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <p style={{ margin: 0 }}>Status Model: <span className={`status-text ${getStatusClass()}`}>{status}</span></p>
        {isLoading && <div className="spinner"></div>}
      </div>
      <div className="button-group">
        <button id="evaluate-button" onClick={onEvaluate} disabled={!isReady || isLoading}>Evaluasi</button>
        <button id="train-button" disabled={isLoading}>Latih Ulang</button>
      </div>
    </div>
  );
}
export default ModelControl;