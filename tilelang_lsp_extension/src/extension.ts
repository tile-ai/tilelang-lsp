import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { execFileSync, execSync } from 'child_process';
import { buildCudaDebugConfiguration, findGeneratedSources, resolveDebugDirectory } from './cudaDebug';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;
let outputChannel: vscode.OutputChannel;
let statusBarItem: vscode.StatusBarItem;

export async function activate(context: vscode.ExtensionContext) {
    outputChannel = vscode.window.createOutputChannel('TileLang LSP');
    context.subscriptions.push(outputChannel);

    // Status bar
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = 'tilelang.showOutput';
    context.subscriptions.push(statusBarItem);

    // Command: show output channel
    context.subscriptions.push(
        vscode.commands.registerCommand('tilelang.showOutput', () => {
            outputChannel.show();
        })
    );

    // Command: restart server
    context.subscriptions.push(
        vscode.commands.registerCommand('tilelang.restartServer', async () => {
            await restartClient(context);
            vscode.window.showInformationMessage('TileLang LSP server restarted.');
        })
    );

    // Command: debug the active Python script with CUDA-GDB (via Nsight VSCE).
    context.subscriptions.push(
        vscode.commands.registerCommand('tilelang.debugKernel', () => startCudaDebug(context))
    );

    // Command: open the generated source dumped by the last debug session.
    context.subscriptions.push(
        vscode.commands.registerCommand('tilelang.openGeneratedSource', () => openGeneratedSource())
    );

    // Restart server when configuration changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(async (e) => {
            if (
                e.affectsConfiguration('tilelang.pythonPath') ||
                e.affectsConfiguration('tilelang.enableInlayHints') ||
                e.affectsConfiguration('tilelang.serverArgs') ||
                e.affectsConfiguration('tilelang.serverEnv')
            ) {
                await restartClient(context);
            }
        })
    );

    await startClient(context);
}

/**
 * Resolve a working Python interpreter path.
 * Tries: configured path -> python extension path -> python3 -> python
 */
function resolvePythonPath(configured: string): string {
    // 1. Try user-configured path
    if (configured && configured !== 'python') {
        try {
            execSync(`"${configured}" --version`, { stdio: 'pipe' });
            return configured;
        } catch {
            outputChannel.appendLine(`Configured Python path "${configured}" not found, trying alternatives...`);
        }
    }

    // 2. Try the Python extension's selected interpreter
    const pythonExt = vscode.extensions.getExtension('ms-python.python');
    if (pythonExt?.isActive) {
        const pythonApi = pythonExt.exports;
        try {
            // Python extension API v3+
            const envPath = pythonApi?.environments?.getActiveEnvironmentPath?.();
            if (envPath?.path) {
                outputChannel.appendLine(`Using Python extension interpreter: ${envPath.path}`);
                return envPath.path;
            }
        } catch {
            // ignore
        }
    }

    // 3. Try common names
    for (const candidate of ['python', 'python3']) {
        try {
            execSync(`${candidate} --version`, { stdio: 'pipe' });
            return candidate;
        } catch {
            continue;
        }
    }

    return configured || 'python';
}

async function startClient(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('tilelang');
    const enabled = config.get<boolean>('enableInlayHints', true);
    if (!enabled) {
        outputChannel.appendLine('TileLang inlay hints disabled by configuration.');
        setStatus('$(circle-slash) TileLang LSP: Disabled', 'Disabled by configuration');
        return;
    }

    setStatus('$(loading~spin) TileLang LSP', 'Starting...');

    const configuredPython = config.get<string>('pythonPath', 'python');
    const pythonPath = resolvePythonPath(configuredPython);
    const extraArgs = config.get<string[]>('serverArgs', []);
    const serverEnv = config.get<Record<string, string>>('serverEnv', {});

    // Build PYTHONPATH to include the repo's tlsp module
    const serverModuleRoot = getServerModulePath(context);
    const serverArgs = ['-m', 'tlsp', ...extraArgs];

    outputChannel.appendLine(`Python: ${pythonPath}`);
    outputChannel.appendLine(`Server command: ${pythonPath} ${serverArgs.join(' ')}`);

    // Pre-flight check: can we import tlsp?
    const envPythonPath = serverModuleRoot
        ? [serverModuleRoot, process.env['PYTHONPATH'] || ''].filter(Boolean).join(path.delimiter)
        : process.env['PYTHONPATH'] || '';

    try {
        const checkCmd = `"${pythonPath}" -c "import tlsp; print('OK')"`;
        const checkEnv = serverModuleRoot
            ? { ...process.env, PYTHONPATH: envPythonPath }
            : process.env;
        execSync(checkCmd, { env: checkEnv, stdio: 'pipe', timeout: 10000 });
        outputChannel.appendLine('Pre-flight check: tlsp module found.');
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        outputChannel.appendLine(`Pre-flight check FAILED: ${msg}`);
        outputChannel.appendLine('Please install the tlsp package: pip install -e <path-to-tilelang-lsp-repo>');
        setStatus('$(error) TileLang LSP', 'Server module not found. Click for details.');
        vscode.window.showErrorMessage(
            'TileLang LSP: Cannot find tlsp module. Run `pip install -e .` in the tilelang-lsp repo. See Output for details.',
            'Show Output'
        ).then(choice => {
            if (choice === 'Show Output') { outputChannel.show(); }
        });
        return;
    }

    const serverOptions: ServerOptions = {
        command: pythonPath,
        args: serverArgs,
        options: {
            env: {
                ...process.env,
                ...serverEnv,
                ...(serverModuleRoot ? { PYTHONPATH: envPythonPath } : {}),
            },
        },
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'python' }],
        outputChannel,
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.py'),
        },
    };

    client = new LanguageClient(
        'tilelangLSP',
        'TileLang LSP',
        serverOptions,
        clientOptions
    );

    try {
        await client.start();
        outputChannel.appendLine('TileLang LSP server started and connected.');
        setStatus('$(check) TileLang LSP', 'Server running');
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        outputChannel.appendLine(`Failed to start server: ${msg}`);
        setStatus('$(error) TileLang LSP', 'Failed to start. Click for details.');
        vscode.window.showErrorMessage(
            `TileLang LSP: Failed to start server. See Output for details.`,
            'Show Output'
        ).then(choice => {
            if (choice === 'Show Output') { outputChannel.show(); }
        });
        client = undefined;
    }
}

const NSIGHT_EXTENSION_ID = 'nvidia.nsight-vscode-edition';

function getDebugDir(folder: vscode.WorkspaceFolder | undefined, fallbackRoot?: string): string {
    const configured = vscode.workspace.getConfiguration('tilelang').get<string>('debug.debugDir', '');
    const root =
        folder?.uri.fsPath ??
        fallbackRoot ??
        vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ??
        process.cwd();
    return resolveDebugDirectory(configured, root);
}

/** Resolve a bare command through PATH so the debug adapter gets an absolute program. */
function resolveExecutablePath(command: string): string {
    if (!command || path.isAbsolute(command)) {
        return command;
    }
    try {
        const locator = process.platform === 'win32' ? 'where' : 'command -v';
        const output = execSync(`${locator} ${JSON.stringify(command)}`, {
            encoding: 'utf8',
            stdio: ['ignore', 'pipe', 'ignore'],
        });
        return output.split(/\r?\n/).map((line) => line.trim()).find(Boolean) ?? command;
    } catch {
        return command;
    }
}

async function startCudaDebug(context: vscode.ExtensionContext): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'python' || editor.document.uri.scheme !== 'file') {
        void vscode.window.showErrorMessage('Open a Python kernel file to start a TileLang CUDA debug session.');
        return;
    }
    if (editor.document.isDirty && !(await editor.document.save())) {
        return;
    }
    if (process.platform !== 'linux') {
        void vscode.window.showErrorMessage(
            'TileLang CUDA debugging needs a Linux host with an NVIDIA GPU (use Remote-SSH or WSL from this machine).'
        );
        return;
    }

    const nsight = vscode.extensions.getExtension(NSIGHT_EXTENSION_ID);
    if (!nsight) {
        outputChannel.appendLine('CUDA debugging unavailable: NVIDIA Nsight VS Code Edition is not installed.');
        const action = await vscode.window.showErrorMessage(
            'CUDA debugging needs the NVIDIA Nsight Visual Studio Code Edition extension (debug type "cuda-gdb").',
            'Open Extensions'
        );
        if (action === 'Open Extensions') {
            await vscode.commands.executeCommand('workbench.extensions.search', NSIGHT_EXTENSION_ID);
        }
        return;
    }
    try {
        await nsight.activate();
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        outputChannel.appendLine(`Failed to activate NVIDIA Nsight: ${message}`);
        void vscode.window.showErrorMessage(`TileLang: failed to activate NVIDIA Nsight: ${message}`);
        return;
    }

    const scriptPath = editor.document.uri.fsPath;
    const folder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
    const config = vscode.workspace.getConfiguration('tilelang');
    const pythonPath = resolveExecutablePath(resolvePythonPath(config.get<string>('pythonPath', 'python')));
    const debugDir = getDebugDir(folder, path.dirname(scriptPath));
    const debugConfig = buildCudaDebugConfiguration({
        scriptPath,
        pythonPath,
        cwd: folder?.uri.fsPath ?? path.dirname(scriptPath),
        debugDir,
        scriptArgs: config.get<string[]>('debug.args', []),
        deviceDebug: config.get<boolean>('debug.deviceDebug', true),
        breakOnLaunch: config.get<boolean>('debug.breakOnLaunch', true),
        cudaGdbPath: config.get<string>('debug.cudaGdbPath', ''),
        serverModuleRoot: getServerModulePath(context),
        inheritedPythonPath: process.env['PYTHONPATH'],
        extraEnv: config.get<Record<string, string>>('debug.extraEnv', {}),
    });

    const debugEnvironment = { ...process.env };
    for (const entry of debugConfig.environment as Array<{ name: string; value: string }>) {
        debugEnvironment[entry.name] = entry.value;
    }
    const probe = [
        'import tlsp',
        'import tilelang',
        'from tilelang.transform import PassConfigKey',
        "assert hasattr(PassConfigKey, 'TL_EMIT_LINE_DIRECTIVES'), 'TileLang >= 0.1.15 with tl.emit_line_directives is required'",
    ].join('; ');
    try {
        execFileSync(pythonPath, ['-c', probe], { env: debugEnvironment, stdio: 'pipe', timeout: 30000 });
        outputChannel.appendLine('CUDA debug pre-flight check: tlsp, TileLang, and Python line mapping are available.');
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        outputChannel.appendLine(`CUDA debug pre-flight check FAILED: ${message}`);
        const action = await vscode.window.showErrorMessage(
            'TileLang: the selected Python cannot provide tlsp and TileLang >= 0.1.15 line mapping. Check tilelang.pythonPath and Output.',
            'Show Output'
        );
        if (action === 'Show Output') {
            outputChannel.show();
        }
        return;
    }

    outputChannel.appendLine(
        `Starting CUDA debug session: ${pythonPath} ${debugConfig.args.join(' ')}\n  cwd: ${debugConfig.cwd}\n  generated: ${debugDir}/generated`
    );
    try {
        const started = await vscode.debug.startDebugging(folder, debugConfig);
        if (!started) {
            outputChannel.appendLine('CUDA debug session did not start; inspect the Debug Console for details.');
            void vscode.window.showErrorMessage(
                'TileLang: the CUDA debug session did not start. Check the TileLang LSP and Debug Console output.'
            );
        }
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        outputChannel.appendLine(`Failed to start CUDA debug session: ${message}`);
        void vscode.window.showErrorMessage(`TileLang: failed to start CUDA debug session: ${message}`);
    }
}

async function openGeneratedSource(): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    const folder = editor
        ? vscode.workspace.getWorkspaceFolder(editor.document.uri)
        : vscode.workspace.workspaceFolders?.[0];
    const fallbackRoot = editor ? path.dirname(editor.document.uri.fsPath) : undefined;
    const generatedDir = path.join(getDebugDir(folder, fallbackRoot), 'generated');
    const entries = findGeneratedSources(generatedDir);
    if (!entries.length) {
        void vscode.window.showInformationMessage(
            'TileLang: no generated source yet. Run "TileLang: Debug CUDA Kernel" first.'
        );
        return;
    }

    let selected = entries[0];
    if (entries.length > 1) {
        const picked = await vscode.window.showQuickPick(
            entries.map((entry) => ({
                label: entry.name,
                description: new Date(entry.mtime).toLocaleString(),
                entry,
            })),
            { placeHolder: 'Generated CUDA source' }
        );
        if (!picked) {
            return;
        }
        selected = picked.entry;
    }

    const document = await vscode.workspace.openTextDocument(vscode.Uri.file(selected.fsPath));
    await vscode.window.showTextDocument(document, { preview: false });
}

function getServerModulePath(context: vscode.ExtensionContext): string | undefined {
    // In the repo, the structure is:
    //   tilelang-lsp/
    //     tlsp/           <-- server package
    //     tilelang_lsp_extension/   <-- this extension
    const extensionDir = context.extensionPath;
    const repoRoot = path.resolve(extensionDir, '..');
    const tlspPath = path.join(repoRoot, 'tlsp', '__init__.py');

    if (fs.existsSync(tlspPath)) {
        outputChannel.appendLine(`Found tlsp module at repo root: ${repoRoot}`);
        return repoRoot;
    }

    outputChannel.appendLine(`tlsp module not found at ${tlspPath}, relying on system install.`);
    return undefined;
}

function setStatus(text: string, tooltip: string) {
    statusBarItem.text = text;
    statusBarItem.tooltip = tooltip;
    statusBarItem.show();
}

async function restartClient(context: vscode.ExtensionContext) {
    outputChannel.appendLine('Restarting TileLang LSP server...');
    if (client) {
        await client.stop();
        client = undefined;
    }
    await startClient(context);
}

export async function deactivate(): Promise<void> {
    if (client) {
        await client.stop();
        client = undefined;
    }
}
