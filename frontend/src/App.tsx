import React, { useState } from 'react';
import UploadForm from './components/UploadForm';
import PDFUploader from './components/PDFUploader';
import QueryForm from './components/QueryForm';
import Output from './components/Output';
import './App.css';

const App: React.FC = () => {
  const [response, setResponse] = useState<any>(null);
  const [results, setResults] = useState<any[]>([]);

  const handleUploadComplete = (res: any) => {
    setResponse(res);
    setResults([]); // Clear results when a new file is uploaded
  };

  const handleQueryComplete = (res: any) => {
    console.log('Query results:', res);
    setResults(res.result); // 'result' is the array inside 'res'
    setResponse(null);
  };

  return (
    <div className="app">
      <h1>GraphRAG Chatbot</h1>
      <div style={{ display: 'flex', gap: '20px', justifyContent: 'center' }}>
        <UploadForm onUploadComplete={handleUploadComplete} />
        <PDFUploader onUploadComplete={handleUploadComplete} />
      </div>
      <QueryForm onQueryComplete={handleQueryComplete} />
      <Output response={response} results={results} />
    </div>
  );
};

export default App;