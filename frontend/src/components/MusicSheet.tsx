// src/components/MusicSheet.tsx
import React, { useEffect, useRef } from "react";
import type { OpenSheetMusicDisplay } from "opensheetmusicdisplay";

interface MusicSheetProps {
    musicxmlContent: string;
}

const MusicSheet: React.FC<MusicSheetProps> = ({ musicxmlContent }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const osmdRef = useRef<OpenSheetMusicDisplay | null>(null);

    useEffect(() => {
        const renderSheet = async () => {
            const { OpenSheetMusicDisplay } = await import(
                "opensheetmusicdisplay"
            );
            if (containerRef.current) {
                const osmd = new OpenSheetMusicDisplay(containerRef.current, {
                    autoResize: true,
                    drawingParameters: {
                        drawInstrumentName: false,
                    },
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

    // ResizeObserver to update rendering on container resize
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
            style={{ minHeight: "400px" }} // Adjust as needed
        />
    );
};

export default MusicSheet;
