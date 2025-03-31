declare module "opensheetmusicdisplay" {
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
