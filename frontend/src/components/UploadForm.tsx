import React, { useState } from 'react';
import axios from 'axios';

interface UploadFormProps {
    onUploadComplete: (response: any) => void;
}

const UploadForm: React.FC<UploadFormProps> = ({ onUploadComplete }) => {
    const [fileUrl, setFileUrl] = useState<string>('');
    const [text, setText] = useState<string>('');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const payload = {
            file_urls: fileUrl ? [fileUrl] : [],
            texts: text ? [text] : [],
            kb: 'kb0000',
            is_re_submit: false,
        };

        try {
            const res = await axios.post('http://localhost:8000/api/insert', payload);
            onUploadComplete(res.data);
        } catch (error) {
            console.error('Error:', error);
        }
    };

    const handleClear = () => {
        setFileUrl('');
        setText('');
    };

    return (
        <div className="form-container">
            <h2>Upload PDF or Add Text</h2>
            <input
                type="text"
                placeholder="Paste PDF URL"
                value={fileUrl}
                onChange={(e) => setFileUrl(e.target.value)}
            />
            <textarea
                placeholder="Or enter text to add to GraphRAG knowledge base"
                value={text}
                onChange={(e) => setText(e.target.value)}
            />
            <div className="button-group">
                <button type="button" onClick={handleClear}>Clear</button>
                <button type="submit" onClick={handleSubmit}>Submit</button>
            </div>
        </div>
    );
};

export default UploadForm;