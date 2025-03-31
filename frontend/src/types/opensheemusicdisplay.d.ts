declare module "opensheetmusicdisplay" {
    export interface DrawingParameters {
        drawInstrumentName?: boolean;
        // add other drawing parameters as needed
        [key: string]: any;
    }

    export interface Options {
        autoResize?: boolean;
        drawingParameters?: string;
        [key: string]: any;
    }

    export class OpenSheetMusicDisplay {
        constructor(container: HTMLElement, options?: Options);
        load(musicxml: string): Promise<void>;
        render(): void;
        // additional methods if needed
    }
}
