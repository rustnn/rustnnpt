import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { existsSync, readdirSync } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';

function findOrtLibDirs(baseDir) {
  const root = path.join(baseDir, 'target', 'onnxruntime');
  if (!existsSync(root)) return [];
  const dirs = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const libDir = path.join(root, entry.name, 'lib');
    if (existsSync(libDir)) {
      dirs.push(libDir);
    }
  }
  return dirs;
}

function guessedOrtLibDirs(baseDir) {
  const root = path.join(baseDir, 'target', 'onnxruntime');
  const suffixes = [
    'onnxruntime-linux-x64-1.23.2/lib',
    'onnxruntime-linux-aarch64-1.23.2/lib',
    'onnxruntime-osx-arm64-1.23.2/lib',
    'onnxruntime-osx-x86_64-1.23.2/lib',
    'onnxruntime-win-x64-1.23.2/lib'
  ];
  return suffixes.map((suffix) => path.join(root, suffix));
}

function withOrtRuntimeEnv(cwd) {
  const env = { ...process.env };
  const libDirs = [
    ...findOrtLibDirs(cwd),
    ...findOrtLibDirs(path.join(cwd, '..', 'rustnn')),
    ...guessedOrtLibDirs(cwd),
    ...guessedOrtLibDirs(path.join(cwd, '..', 'rustnn'))
  ];
  const uniqueLibDirs = [...new Set(libDirs)];
  if (uniqueLibDirs.length === 0) {
    return env;
  }

  if (process.platform === 'darwin') {
    const existing = env.DYLD_LIBRARY_PATH ?? '';
    env.DYLD_LIBRARY_PATH = existing ? `${uniqueLibDirs.join(':')}:${existing}` : uniqueLibDirs.join(':');
  } else if (process.platform === 'linux') {
    const existing = env.LD_LIBRARY_PATH ?? '';
    env.LD_LIBRARY_PATH = existing ? `${uniqueLibDirs.join(':')}:${existing}` : uniqueLibDirs.join(':');
  } else if (process.platform === 'win32') {
    const existing = env.PATH ?? '';
    env.PATH = existing ? `${uniqueLibDirs.join(';')};${existing}` : uniqueLibDirs.join(';');
  }

  return env;
}

export class RunnerClient {
  constructor({ manifestPath = 'crates/wpt-runner/Cargo.toml', cwd = process.cwd() } = {}) {
    this.cwd = cwd;
    this.proc = spawn('cargo', ['run', '--quiet', '--manifest-path', manifestPath], {
      cwd,
      stdio: ['pipe', 'pipe', 'inherit'],
      env: withOrtRuntimeEnv(cwd)
    });
    this.pending = new Map();

    const rl = createInterface({ input: this.proc.stdout });
    rl.on('line', (line) => {
      if (!line.trim()) return;
      let msg;
      try {
        msg = JSON.parse(line);
      } catch (err) {
        return;
      }
      const waiter = this.pending.get(msg.id);
      if (!waiter) return;
      this.pending.delete(msg.id);
      if (msg.ok) {
        waiter.resolve(msg.outputs ?? {});
      } else {
        const error = new Error(msg.error?.message ?? 'runner error');
        error.kind = msg.error?.kind ?? 'RuntimeExecutionError';
        waiter.reject(error);
      }
    });

    this.proc.on('exit', (code, signal) => {
      const err = new Error(`runner exited (code=${code}, signal=${signal})`);
      for (const { reject } of this.pending.values()) {
        reject(err);
      }
      this.pending.clear();
    });
  }

  async executeGraph({ graph, inputs, expectedOutputs, contextOptions = {} }) {
    const id = randomUUID();
    const payload = {
      cmd: 'execute_graph',
      id,
      graph,
      inputs,
      expected_outputs: expectedOutputs,
      context_options: contextOptions
    };

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.proc.stdin.write(`${JSON.stringify(payload)}\n`);
    });
  }

  async close() {
    if (!this.proc || this.proc.killed) return;
    this.proc.stdin.end();
    this.proc.kill('SIGTERM');
  }
}
