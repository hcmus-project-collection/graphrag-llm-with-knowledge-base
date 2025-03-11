import React, { useState } from 'react';
import axios from 'axios';

interface PdfUploaderProps {
    onUploadComplete: (response: any) => void;
}

const PDFUploader: React.FC<PdfUploaderProps> = ({ onUploadComplete }) => {
    const [file, setFile] = useState<File | null>(null);
    const [isUploading, setIsUploading] = useState(false);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const uploadedFile = event.target.files?.[0] || null;
        setFile(uploadedFile);
    };

    const uploadToCloudinary = async (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('upload_preset', 'pdfdocuments'); // Your Cloudinary preset

        try {
            const response = await axios.post(
                'https://api.cloudinary.com/v1_1/dh6n8cuqo/raw/upload', // For PDF files
                formData
            );

            return response.data.secure_url;
        } catch (error) {
            console.error('Error uploading file:', error);
            throw error;
        }
    };

    const handleSubmit = async (event: React.FormEvent) => {
        event.preventDefault();
        if (!file) return alert('Please upload a file first!');

        setIsUploading(true);
        try {
            const fileUrl = await uploadToCloudinary(file);
            console.log('File uploaded to Cloudinary:', fileUrl);

            // Now send this URL to the backend API
            const payload = {
                file_urls: [fileUrl], // Sending file URL
                texts: [],
                kb: 'kb0000',
                is_re_submit: false,
            };

            const res = await axios.post('http://localhost:8000/api/insert', payload);
            console.log('Backend response:', res.data);
            onUploadComplete(res.data);
        } catch (error) {
            alert('Failed to upload file or send to backend');
            console.error('Error:', error);
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <input type="file" accept="application/pdf" onChange={handleFileChange} />
            <button type="submit" disabled={!file || isUploading}>
                {isUploading ? 'Uploading...' : 'Upload PDF to Cloudinary & Submit'}
            </button>
        </form>
    );
};

export default PDFUploader;