import * as assert from 'assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { test } from 'node:test';
import {
    buildCudaDebugConfiguration,
    CudaDebugOptions,
    findGeneratedSources,
    resolveDebugDirectory,
} from '../cudaDebug';

const baseOptions: CudaDebugOptions = {
    scriptPath: '/workspace with space/kernel.py',
    pythonPath: '/opt/python env/bin/python',
    cwd: '/workspace with space',
    debugDir: '/workspace with space/.tlsp/debug',
    scriptArgs: ['--value', 'hello world'],
    deviceDebug: true,
    breakOnLaunch: false,
    serverModuleRoot: '/repo root',
    inheritedPythonPath: '/existing',
    extraEnv: {
        PYTHONPATH: '/custom',
        PYTHONUNBUFFERED: '0',
        CUSTOM: 'value',
    },
};

test('builds an Nsight launch config without losing paths or argument boundaries', () => {
    const config = buildCudaDebugConfiguration(baseOptions);
    assert.equal(config.type, 'cuda-gdb');
    assert.equal(config.request, 'launch');
    assert.equal(config.program, '/opt/python env/bin/python');
    assert.equal(config.cwd, '/workspace with space');
    assert.deepEqual(config.args, [
        '-m',
        'tlsp.debug',
        '--debug-dir',
        '/workspace with space/.tlsp/debug',
        '--',
        '/workspace with space/kernel.py',
        '--value',
        'hello world',
    ]);
    assert.deepEqual(config.environment, [
        { name: 'PYTHONUNBUFFERED', value: '0' },
        { name: 'TILELANG_DISABLE_CACHE', value: '1' },
        { name: 'TILELANG_ENABLE_IR_SPAN', value: '1' },
        { name: 'PYTHONPATH', value: ['/repo root', '/custom', '/existing'].join(path.delimiter) },
        { name: 'CUSTOM', value: 'value' },
    ]);
    assert.equal(config.breakOnLaunch, false);
    assert.equal(config.stopAtEntry, false);
    assert.equal(config.setupCommands[0].text, 'set breakpoint pending on');
    assert.equal('miDebuggerPath' in config, false);
});

test('forwards line-only mode, launch stops, and a custom cuda-gdb', () => {
    const config = buildCudaDebugConfiguration({
        ...baseOptions,
        deviceDebug: false,
        breakOnLaunch: true,
        cudaGdbPath: ' /opt/cuda/bin/cuda-gdb ',
    });
    assert.deepEqual(config.args.slice(0, 6), [
        '-m',
        'tlsp.debug',
        '--debug-dir',
        '/workspace with space/.tlsp/debug',
        '--no-device-debug',
        '--',
    ]);
    assert.equal(config.breakOnLaunch, true);
    assert.equal(config.miDebuggerPath, '/opt/cuda/bin/cuda-gdb');
});

test('resolves default, relative, and absolute debug directories', () => {
    assert.equal(resolveDebugDirectory('', '/workspace'), path.join('/workspace', '.tlsp', 'debug'));
    assert.equal(resolveDebugDirectory(' artifacts ', '/workspace'), path.join('/workspace', 'artifacts'));
    assert.equal(resolveDebugDirectory('/tmp/debug', '/workspace'), '/tmp/debug');
});

test('finds JIT and NVCC source files recursively', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'tlsp-cuda-debug-'));
    try {
        const nested = path.join(root, 'nested');
        fs.mkdirSync(nested);
        const jitSource = path.join(root, 'kernel.c');
        const nvccSource = path.join(nested, 'tvm_kernels.cu');
        fs.writeFileSync(jitSource, '// jit\n');
        fs.writeFileSync(nvccSource, '// nvcc\n');
        fs.writeFileSync(path.join(nested, 'kernel.cubin'), 'binary');
        const now = Date.now() / 1000;
        fs.utimesSync(jitSource, now - 20, now - 20);
        fs.utimesSync(nvccSource, now - 10, now - 10);

        const sources = findGeneratedSources(root);
        assert.deepEqual(
            sources.map((source) => source.name),
            [path.join('nested', 'tvm_kernels.cu'), 'kernel.c']
        );
        assert.deepEqual(findGeneratedSources(path.join(root, 'missing')), []);
    } finally {
        fs.rmSync(root, { recursive: true, force: true });
    }
});
