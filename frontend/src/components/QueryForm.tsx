import React, { useState } from 'react';
import axios from 'axios';

interface QueryFormProps {
    onQueryComplete: (results: any[]) => void;
}

const QueryForm: React.FC<QueryFormProps> = ({ onQueryComplete }) => {
    const [query, setQuery] = useState<string>('');

    const handleQuery = async (e: React.FormEvent) => {
        e.preventDefault();
        const payload = {
            query,
            top_k: 5,
            kb: ['kb0000'],
            threshold: 0.2,
        };

        try {
            const res = await axios.post('http://localhost:8000/api/query', payload);
            onQueryComplete(res.data.result);
        } catch (error) {
            console.error('Error:', error);
        }
    };

    const handleClear = () => setQuery('');

    return (
        <div className="form-container">
            <h2>Input Query</h2>
            <textarea
                placeholder="Ask something..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
            />
            <div className="button-group">
                <button type="button" onClick={handleClear}>Clear</button>
                <button type="submit" onClick={handleQuery}>Submit</button>
            </div>
        </div>
    );
};

export default QueryForm;