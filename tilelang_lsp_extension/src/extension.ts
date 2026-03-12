import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { execSync } from 'child_process';
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

    // Restart server when configuration changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(async (e) => {
            if (
                e.affectsConfiguration('tilelang.pythonPath') ||
                e.affectsConfiguration('tilelang.enableInlayHints') ||
                e.affectsConfiguration('tilelang.serverArgs')
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
        initializationOptions: {},
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
