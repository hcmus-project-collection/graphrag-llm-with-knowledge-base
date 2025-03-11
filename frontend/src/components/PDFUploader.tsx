// components/PdfUploader.tsx
import React, { useState } from 'react';
import axios from 'axios';

const PdfUploader: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [pdfUrl, setPdfUrl] = useState<string>('');
    const [query, setQuery] = useState<string>('');

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const uploadedFile = e.target.files?.[0];
        if (uploadedFile && uploadedFile.type === 'application/pdf') {
            setFile(uploadedFile);
        } else {
            alert('Please upload a valid PDF file.');
        }
    };

    const uploadToCloudinary = async () => {
        if (!file) return alert('No file selected');

        const formData = new FormData();
        formData.append('file', file);
        formData.append('upload_preset', 'pdfdocuments'); // Set this up in Cloudinary

        try {
            const response = await axios.post(
                `https://api.cloudinary.com/v1_1/dh6n8cuqo/auto/upload`,
                formData
            );

            const uploadedUrl = response.data.secure_url;
            setPdfUrl(uploadedUrl);
            alert('File uploaded successfully!');
        } catch (error) {
            console.error('Upload error:', error);
            alert('Failed to upload');
        }
    };

    const handleSubmit = async () => {
        if (!pdfUrl || !query) return alert('Please upload a PDF and enter a query');

        try {
            await axios.post('http://localhost:8000/api/your-endpoint', {
                pdf_url: pdfUrl,
                query,
            });
            alert('Query sent successfully!');
        } catch (error) {
            console.error('Error sending query:', error);
            alert('Failed to send query');
        }
    };

    return (
        <div style={{ padding: '20px' }}>
            <input type="file" accept="application/pdf" onChange={handleFileChange} />
            <button onClick={uploadToCloudinary} disabled={!file}>
                Upload PDF
            </button>

            {pdfUrl && (
                <>
                    <input
                        type="text"
                        placeholder="Enter your query"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                    />
                    <button onClick={handleSubmit}>Submit</button>
                </>
            )}
        </div>
    );
};

export default PdfUploader;