import React from 'react';

interface OutputProps {
    response: any;
    results: any[];
}

const Output: React.FC<OutputProps> = ({ response, results }) => (
    <div className="output-container">
        {response && (
            <div className="response">
                <h3>Response</h3>
                <p>{response.result}</p>
            </div>
        )}
        {results.length > 0 && (
            <div className="results">
                <h3>Results:</h3>
                {results.map((item, index) => (
                    <div key={index} className="result-item">
                        <p><strong>Content:</strong> {item.content}</p>
                        <p><strong>Score:</strong> {item.score}</p>
                        {item.reference && (
                            <p><strong>Reference:</strong> <a href={item.reference} target="_blank" rel="noopener noreferrer">{item.reference}</a></p>
                        )}
                    </div>
                ))}
            </div>
        )}
    </div>
);

export default Output;