import React, { useState } from 'react';
import UploadForm from './components/UploadForm';
import QueryForm from './components/QueryForm';
import Output from './components/Output';
import './App.css';

const App: React.FC = () => {
  const [response, setResponse] = useState<any>(null);
  const [results, setResults] = useState<any[]>([]);

  return (
    <div className="app">
      <h1>RAG Chatbot</h1>
      <UploadForm onUploadComplete={setResponse} />
      <QueryForm onQueryComplete={setResults} />
      <Output response={response} results={results} />
    </div>
  );
};

export default App;