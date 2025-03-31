// src/types/opensheetmusicdisplay.d.ts
declare module "opensheetmusicdisplay" {
    export interface Options {
        autoResize?: boolean;
        drawingParameters?: string;
        [key: string]: unknown; // Use unknown instead of any
    }

    export class OpenSheetMusicDisplay {
        constructor(container: HTMLElement, options?: Options);
        load(musicxml: string): Promise<void>;
        render(): void;
        // Additional methods can be added here if needed.
    }
}
