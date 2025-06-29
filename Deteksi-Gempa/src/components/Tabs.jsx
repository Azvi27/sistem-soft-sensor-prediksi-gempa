// File: src/components/Tabs.jsx

import React from 'react';

function Tabs({ activeTab, setActiveTab }) {
  const tabs = ['Selamat Datang', 'Hasil Evaluasi', 'Hasil Prediksi'];

  return (
    <div className="tabs-nav">
      {tabs.map(tab => (
        <button
          key={tab}
          className={`tab-button ${activeTab === tab ? 'active' : ''}`}
          onClick={() => setActiveTab(tab)}
        >
          {tab}
        </button>
      ))}
    </div>
  );
}

export default Tabs;