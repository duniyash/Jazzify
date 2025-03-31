// components/MusicSheet.tsx
import React, { useEffect, useRef } from "react";

interface MusicSheetProps {
    musicxmlContent: string;
}

const MusicSheet: React.FC<MusicSheetProps> = ({ musicxmlContent }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const osmdRef = useRef<any>(null);

    useEffect(() => {
        const renderSheet = async () => {
            const { OpenSheetMusicDisplay } = await import(
                "opensheetmusicdisplay"
            );
            if (containerRef.current) {
                // Initialize OSMD with autoResize and desired drawing parameters
                const osmd = new OpenSheetMusicDisplay(containerRef.current, {
                    autoResize: true,
                    drawingParameters: "compacttight",
                });
                osmdRef.current = osmd;
                await osmd.load(musicxmlContent);
                osmd.render();
            }
        };

        if (musicxmlContent && containerRef.current) {
            renderSheet();
        }
    }, [musicxmlContent]);

    // Use ResizeObserver to trigger a re-render when the container size changes
    useEffect(() => {
        if (!containerRef.current) return;

        const observer = new ResizeObserver(() => {
            if (osmdRef.current) {
                osmdRef.current.render();
            }
        });
        observer.observe(containerRef.current);

        return () => {
            observer.disconnect();
        };
    }, []);

    return (
        <div
            ref={containerRef}
            className="mt-4 p-4 bg-white rounded-lg shadow-lg overflow-auto w-full"
            style={{ minHeight: "400px" }} // Adjust minHeight as needed
        />
    );
};

export default MusicSheet;
