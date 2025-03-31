"use client";

import React, { useState, ChangeEvent, useRef, useEffect } from "react";
import MusicSheet from "../components/MusicSheet";
import { Button } from "@/components/ui/button"; // adjust import path as needed
import { Progress } from "@/components/ui/progress"; // adjust import path as needed

export default function Home() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [originalMusicxml, setOriginalMusicxml] = useState<string | null>(
        null
    );
    const [processedMusicxml, setProcessedMusicxml] = useState<string | null>(
        null
    );
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [progress, setProgress] = useState<number>(0);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

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
            setProgress(0);

            // Start simulating progress until completion (up to 95%)
            progressIntervalRef.current = setInterval(() => {
                setProgress((prev) => {
                    const next = prev + 5;
                    return next >= 95 ? 95 : next;
                });
            }, 200);

            // Read file content as text for original display
            const originalText = await selectedFile.text();
            setOriginalMusicxml(originalText);
            setProcessedMusicxml(null);

            // Create FormData and append the file under the key "file"
            const formData = new FormData();
            formData.append("file", selectedFile);

            // Send the file to the API endpoint
            const response = await fetch(
                "http://127.0.0.1:8000/process_musicxml",
                {
                    method: "POST",
                    body: formData,
                }
            );

            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}`);
            }

            // The API returns processed MusicXML as raw text
            const processedText = await response.text();
            setProcessedMusicxml(processedText);

            // On success, set progress to 100 and clear the progress interval
            setProgress(100);
            if (progressIntervalRef.current) {
                clearInterval(progressIntervalRef.current);
            }
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

    // Clear the interval on unmount
    useEffect(() => {
        return () => {
            if (progressIntervalRef.current) {
                clearInterval(progressIntervalRef.current);
            }
        };
    }, []);

    return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-4 space-y-4">
            <h1 className="text-3xl font-bold mb-6">Upload a MusicXML File</h1>

            <div className="w-full max-w-md flex flex-col items-center space-y-4">
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
                    <p className="text-sm text-gray-700">
                        Selected File: {selectedFile.name}
                    </p>
                )}

                {/* Upload button appears once a file is selected */}
                {selectedFile && (
                    <Button onClick={handleUpload} variant="secondary">
                        Upload
                    </Button>
                )}
            </div>

            {/* Display the progress bar while processing */}
            {loading && (
                <div className="w-full max-w-md mt-4 flex flex-col items-center space-y-2">
                    <Progress value={progress} className="w-[60%]" />
                    <p className="mt-2 text-blue-600 font-medium">
                        Processing file... {progress}%
                    </p>
                </div>
            )}

            {error && (
                <p className="mt-4 text-red-600 font-medium">Error: {error}</p>
            )}

            {/* Render both original and processed sheets side by side once processing is complete */}
            {originalMusicxml && processedMusicxml && !loading && (
                <div className="mt-8 w-full max-w-6xl grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <h2 className="text-xl font-semibold mb-2 text-center">
                            Original
                        </h2>
                        <MusicSheet musicxmlContent={originalMusicxml} />
                    </div>
                    <div>
                        <h2 className="text-xl font-semibold mb-2 text-center">
                            Processed
                        </h2>
                        <MusicSheet musicxmlContent={processedMusicxml} />
                    </div>
                </div>
            )}

            {/* If only the original is available (and not processing), show it */}
            {originalMusicxml && !processedMusicxml && !loading && (
                <div className="mt-8 w-full max-w-4xl">
                    <h2 className="text-xl font-semibold mb-2 text-center">
                        Original
                    </h2>
                    <MusicSheet musicxmlContent={originalMusicxml} />
                </div>
            )}
        </div>
    );
}
