import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { createInterface } from 'node:readline';
import path from 'node:path';

export class RunnerClient {
  constructor({ manifestPath = 'crates/wpt-runner/Cargo.toml', cwd = process.cwd() } = {}) {
    this.cwd = cwd;
    const ortLibDir = path.join(cwd, '..', 'rustnn', 'target', 'onnxruntime', 'onnxruntime-osx-arm64-1.23.2', 'lib');
    const existingDyld = process.env.DYLD_LIBRARY_PATH ?? '';
    const dyldLibraryPath = existingDyld ? `${ortLibDir}:${existingDyld}` : ortLibDir;
    this.proc = spawn('cargo', ['run', '--quiet', '--manifest-path', manifestPath], {
      cwd,
      stdio: ['pipe', 'pipe', 'inherit'],
      env: {
        ...process.env,
        DYLD_LIBRARY_PATH: dyldLibraryPath
      }
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
