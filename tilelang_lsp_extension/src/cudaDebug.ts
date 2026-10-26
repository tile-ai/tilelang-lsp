import * as fs from 'fs';
import * as path from 'path';
import type { DebugConfiguration } from 'vscode';

export interface CudaDebugOptions {
    scriptPath: string;
    pythonPath: string;
    cwd: string;
    debugDir: string;
    scriptArgs: string[];
    deviceDebug: boolean;
    breakOnLaunch: boolean;
    cudaGdbPath?: string;
    serverModuleRoot?: string;
    inheritedPythonPath?: string;
    extraEnv?: Record<string, string>;
}

export interface GeneratedSource {
    name: string;
    fsPath: string;
    mtime: number;
}

export function resolveDebugDirectory(configured: string, root: string): string {
    const directory = configured.trim();
    if (!directory) {
        return path.join(root, '.tlsp', 'debug');
    }
    return path.isAbsolute(directory) ? directory : path.join(root, directory);
}

function setEnvironment(environment: Array<{ name: string; value: string }>, name: string, value: string): void {
    const existing = environment.find((entry) => entry.name === name);
    if (existing) {
        existing.value = value;
    } else {
        environment.push({ name, value });
    }
}

export function buildCudaDebugConfiguration(options: CudaDebugOptions): DebugConfiguration {
    const environment: Array<{ name: string; value: string }> = [
        { name: 'PYTHONUNBUFFERED', value: '1' },
        { name: 'TILELANG_DISABLE_CACHE', value: '1' },
        { name: 'TILELANG_ENABLE_IR_SPAN', value: '1' },
    ];
    const extraEnv = options.extraEnv ?? {};
    const pythonPaths = [options.serverModuleRoot, extraEnv['PYTHONPATH'], options.inheritedPythonPath].filter(
        (value): value is string => Boolean(value)
    );
    if (pythonPaths.length) {
        environment.push({ name: 'PYTHONPATH', value: pythonPaths.join(path.delimiter) });
    }
    for (const [name, value] of Object.entries(extraEnv)) {
        if (name !== 'PYTHONPATH') {
            setEnvironment(environment, name, String(value));
        }
    }

    const runnerArgs = ['-m', 'tlsp.debug', '--debug-dir', options.debugDir];
    if (!options.deviceDebug) {
        runnerArgs.push('--no-device-debug');
    }
    runnerArgs.push('--', options.scriptPath, ...options.scriptArgs);

    const debugConfig: DebugConfiguration = {
        type: 'cuda-gdb',
        request: 'launch',
        name: `TileLang CUDA: ${path.basename(options.scriptPath)}`,
        program: options.pythonPath,
        args: runnerArgs,
        cwd: options.cwd,
        environment,
        breakOnLaunch: options.breakOnLaunch,
        stopAtEntry: false,
        onAPIError: 'hide',
        setupCommands: [
            {
                text: 'set breakpoint pending on',
                description: 'Resolve breakpoints when dynamically loaded TileLang kernels appear',
                ignoreFailures: false,
            },
        ],
    };
    const cudaGdbPath = options.cudaGdbPath?.trim();
    if (cudaGdbPath) {
        debugConfig['miDebuggerPath'] = cudaGdbPath;
    }
    return debugConfig;
}

export function findGeneratedSources(generatedDir: string): GeneratedSource[] {
    if (!fs.existsSync(generatedDir)) {
        return [];
    }

    const sources: GeneratedSource[] = [];
    const visit = (directory: string): void => {
        for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
            const fsPath = path.join(directory, entry.name);
            if (entry.isDirectory()) {
                visit(fsPath);
            } else if (entry.isFile() && (entry.name.endsWith('.c') || entry.name.endsWith('.cu'))) {
                sources.push({
                    name: path.relative(generatedDir, fsPath),
                    fsPath,
                    mtime: fs.statSync(fsPath).mtimeMs,
                });
            }
        }
    };
    visit(generatedDir);
    return sources.sort((a, b) => b.mtime - a.mtime || a.name.localeCompare(b.name));
}
