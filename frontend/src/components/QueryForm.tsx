import React, { useState } from 'react';
import axios from 'axios';

interface QueryFormProps {
    onQueryComplete: (response: any[]) => void;
}

const QueryForm: React.FC<QueryFormProps> = ({ onQueryComplete }) => {
    const [query, setQuery] = useState<string>('');
    const [topK, setTopK] = useState<number>(5); // Default top_k value

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!query) return alert('Please enter a query!');

        const payload = {
            query,
            top_k: 1,
            kb: ["kb0000"],
            threshold: 0.2,
        };

        try {
            const res = await axios.post('http://localhost:8000/api/query', payload);
            onQueryComplete(res.data);
        } catch (error) {
            console.error('Error:', error);
        }
    };

    return (
        <div className="form-container">
            <h2>Query GraphRAG Knowledge Base</h2>
            <input
                type="text"
                placeholder="Enter your query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
            />
            {/* <input
                type="number"
                placeholder="Top K results"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                min={1}
            /> */}
            <button type="submit" onClick={handleSubmit}>
                Submit Query
            </button>
        </div>
    );
};

export default QueryForm;