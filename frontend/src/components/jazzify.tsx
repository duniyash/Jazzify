"use client";

import React, { useState, ChangeEvent, useRef } from "react";
import MusicSheet from "./MusicSheet";
import { Button } from "@/components/ui/button"; // adjust import path as needed
import { NewtonsCradle } from "ldrs/react";
import "ldrs/react/NewtonsCradle.css";

export default function JazzifyApp() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [originalMusicxml, setOriginalMusicxml] = useState<string | null>(
        null
    );
    const [processedMusicxml, setProcessedMusicxml] = useState<string | null>(
        null
    );
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

    const api_url = process.env.NEXT_PUBLIC_API_URL;

    // Trigger file selection via hidden input
    const handleFileSelect = (event: ChangeEvent<HTMLInputElement>) => {
        setError(null);
        const file = event.target.files?.[0];
        if (file) {
            setSelectedFile(file);
        }
    };

    // Handle upload: send file via FormData to the API and simulate progress updates
    const handleUpload = async () => {
        if (!selectedFile) {
            setError("No file selected.");
            return;
        }
        try {
            setLoading(true);
            setError(null);

            // Read file content as text for original display
            const originalText = await selectedFile.text();
            setOriginalMusicxml(originalText);
            setProcessedMusicxml(null);

            // Create FormData and append the file under the key "file"
            const formData = new FormData();
            formData.append("file", selectedFile);

            if (!api_url) {
                throw new Error("API URL is not configured");
            }

            // Send the file to the API endpoint
            const response = await fetch(api_url, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}`);
            }

            // The API returns processed MusicXML as raw text
            const processedText = await response.text();
            setProcessedMusicxml(processedText);
        } catch (err: unknown) {
            if (progressIntervalRef.current) {
                clearInterval(progressIntervalRef.current);
            }
            if (err instanceof Error) {
                setError(err.message);
            } else {
                setError("An unknown error occurred.");
            }
        } finally {
            setLoading(false);
        }
    };

    // Handle download of processed score
    const handleDownload = () => {
        if (!processedMusicxml) {
            setError("No processed file available to download.");
            return;
        }

        try {
            // Create blob from the processed MusicXML
            const blob = new Blob([processedMusicxml], {
                type: "application/xml",
            });

            // Create download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = selectedFile
                ? `jazzified_${selectedFile.name}`
                : "jazzified_score.xml";

            // Trigger download
            document.body.appendChild(a);
            a.click();

            // Cleanup
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (err: unknown) {
            if (err instanceof Error) {
                setError(err.message);
            } else {
                setError("Failed to download the processed file.");
            }
        }
    };

    return (
        <div className="flex flex-col justify-center items-center space-y-4 bg-gray-100 p-4 min-h-screen">
            <h1 className="mb-6 font-bold text-3xl">Jazzify</h1>

            <div className="flex flex-col items-center space-y-4 w-full max-w-md">
                {/* Hidden file input */}
                <input
                    id="musicxml-upload"
                    type="file"
                    accept=".xml,.musicxml"
                    onChange={handleFileSelect}
                    ref={fileInputRef}
                    className="hidden"
                />

                {/* Button to trigger file selection */}
                <Button onClick={() => fileInputRef.current?.click()}>
                    Select File
                </Button>

                {selectedFile && (
                    <p className="text-gray-700 text-sm">
                        Selected File: {selectedFile.name}
                    </p>
                )}

                {/* Upload button appears once a file is selected */}
                {selectedFile && (
                    <Button onClick={handleUpload} variant="outline">
                        Upload
                    </Button>
                )}
            </div>

            {/* Display the progress bar while processing */}
            {loading && (
                <div className="flex flex-col items-center space-y-2 mt-4 w-full max-w-md">
                    <NewtonsCradle size="78" speed="1.4" color="black" />
                </div>
            )}

            {error && (
                <p className="mt-4 font-medium text-red-600">Error: {error}</p>
            )}

            {/* Render both original and processed sheets side by side once processing is complete */}
            {originalMusicxml && processedMusicxml && !loading && (
                <>
                    <Button onClick={handleDownload}> Download</Button>
                    <div className="gap-4 grid grid-cols-1 md:grid-cols-2 mt-8 w-full max-w-6xl">
                        <div>
                            <h2 className="mb-2 font-semibold text-xl text-center">
                                Original
                            </h2>
                            <MusicSheet musicxmlContent={originalMusicxml} />
                        </div>
                        <div>
                            <h2 className="mb-2 font-semibold text-xl text-center">
                                Processed
                            </h2>
                            <MusicSheet musicxmlContent={processedMusicxml} />
                        </div>
                    </div>
                </>
            )}

            {/* If only the original is available (and not processing), show it */}
            {originalMusicxml && !processedMusicxml && !loading && (
                <div className="mt-8 w-full max-w-4xl">
                    <h2 className="mb-2 font-semibold text-xl text-center">
                        Original
                    </h2>

                    <MusicSheet musicxmlContent={originalMusicxml} />
                </div>
            )}
        </div>
    );
}
