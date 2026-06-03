/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { existsSync, readdirSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { join, sep } from 'node:path';
import { env } from 'node:process';
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

function withCargoCheckCfgEnv(env) {
  // The objc/objc_exception crates (macOS only) probe feature="cargo-clippy",
  // producing check-cfg warnings. Only macOS needs this flag. On other platforms
  // those crates aren't built, so injecting RUSTFLAGS here would differ from the
  // build step's flags and force `cargo run` to recompile the whole tree from
  // scratch (deadly with CARGO_INCREMENTAL=0 in CI) — so leave RUSTFLAGS alone.
  if (process.platform !== 'darwin') {
    return env;
  }
  const allowCargoClippy = '--check-cfg=cfg(feature,values("cargo-clippy"))';
  const existing = env.RUSTFLAGS ?? '';
  if (existing.includes('values("cargo-clippy")')) {
    return env;
  }
  env.RUSTFLAGS = existing ? `${existing} ${allowCargoClippy}` : allowCargoClippy;
  return env;
}

function resolveBinary(binName) {
  const extension = '.exe';
  const target = binName.endsWith(extension) ? binName : binName + extension;

  // 1. Try the raw name (let the OS try one last time)
  // 2. Split the PATH and look manually
  const paths = (env.PATH || '').split(';');
  
  for (let p of paths) {
      // Remove quotes often added by Windows path editors
      const cleanPath = p.replace(/^"|"$/g, '');
      const fullPath = join(cleanPath, target);
      
      if (existsSync(fullPath)) {
          return fullPath;
      }
  }
  return binName; // Fallback to original and hope for the best
}

export class RunnerClient {
  constructor({ manifestPath = 'crates/wpt-runner/Cargo.toml', cwd = process.cwd(), runnerFeatures = [], timeoutMs = 60000 } = {}) {
    this.cwd = cwd;
    // Per-request hard timeout. A graph that wedges the runner produces neither a
    // response nor a process exit, so without this the whole run blocks forever.
    // 0 / non-finite disables it.
    this.timeoutMs = Number.isFinite(Number(timeoutMs)) && Number(timeoutMs) > 0 ? Number(timeoutMs) : 0;
    const features = Array.isArray(runnerFeatures)
      ? runnerFeatures.map((f) => String(f).trim()).filter(Boolean)
      : String(runnerFeatures ?? '').split(',').map((f) => f.trim()).filter(Boolean);
    const cargoArgs = ['run', '--quiet', '--manifest-path', manifestPath];
    if (features.length > 0) {
      cargoArgs.push('--no-default-features', '--features', features.join(','));
    }

    const env = withCargoCheckCfgEnv(withOrtRuntimeEnv(cwd));
    // On Windows node.js does not find cargo in the path. Search for it.
    const cargoExecutable = resolveBinary('cargo');

    this.proc = spawn(cargoExecutable, cargoArgs, {
      cwd,
      // Pipe stderr (instead of inherit) so we can capture the runner's crash
      // output — panics / segfault messages — and surface it in the report.
      stdio: ['pipe', 'pipe', 'pipe'],
      env,
      // Own process group so close() can SIGKILL the whole tree (cargo + the
      // wpt-runner child); otherwise a wedged runner orphans and keeps a core busy.
      detached: true
    });
    this.pending = new Map();
    // Bounded tail of the runner's stderr, attached to the exit error for diagnosis.
    this.stderrTail = '';

    if (this.proc.stderr) {
      this.proc.stderr.on('data', (chunk) => {
        const text = chunk.toString();
        // Keep it visible in the CI log as before, and remember the last ~4KB.
        process.stderr.write(text);
        this.stderrTail = (this.stderrTail + text).slice(-4096);
      });
    }

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
      if (waiter.timer) clearTimeout(waiter.timer);
      if (msg.ok) {
        waiter.resolve(msg.outputs ?? {});
      } else {
        const error = new Error(msg.error?.message ?? 'runner error');
        error.kind = msg.error?.kind ?? 'RuntimeExecutionError';
        waiter.reject(error);
      }
    });

    this.proc.on('exit', (code, signal) => {
      const tail = this.stderrTail.trim();
      const detail = tail ? ` :: runner stderr: ${tail.slice(-2000)}` : '';
      const err = new Error(`runner exited (code=${code}, signal=${signal})${detail}`);
      for (const { reject, timer } of this.pending.values()) {
        if (timer) clearTimeout(timer);
        reject(err);
      }
      this.pending.clear();
    });

    this.proc.stdin.on('error', (err) => {
      const wrapped = new Error(`runner stdin error: ${err.message}`);
      for (const { reject, timer } of this.pending.values()) {
        if (timer) clearTimeout(timer);
        reject(wrapped);
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
      if (!this.proc || this.proc.killed || this.proc.exitCode !== null) {
        reject(new Error(`runner exited before request dispatch (exitCode=${this.proc?.exitCode ?? 'unknown'})`));
        return;
      }
      let timer = null;
      if (this.timeoutMs > 0) {
        timer = setTimeout(() => {
          const waiter = this.pending.get(id);
          if (!waiter) return;
          this.pending.delete(id);
          const error = new Error(`runner timed out after ${this.timeoutMs}ms`);
          error.kind = 'RunnerTimeout';
          waiter.reject(error);
        }, this.timeoutMs);
        if (typeof timer.unref === 'function') timer.unref();
      }
      this.pending.set(id, { resolve, reject, timer });
      this.proc.stdin.write(`${JSON.stringify(payload)}\n`, (err) => {
        if (!err) return;
        const waiter = this.pending.get(id);
        if (!waiter) return;
        this.pending.delete(id);
        if (waiter.timer) clearTimeout(waiter.timer);
        waiter.reject(new Error(`failed to send request to runner: ${err.message}`));
      });
    });
  }

  async close() {
    if (!this.proc || this.proc.killed) return;
    try { this.proc.stdin.end(); } catch { /* already closed */ }
    const pid = this.proc.pid;
    try {
      // Negative pid targets the whole process group (works because we spawned
      // detached), so a wedged cargo + wpt-runner child both die. SIGKILL because
      // a runner stuck in a tight loop may ignore SIGTERM.
      process.kill(-pid, 'SIGKILL');
    } catch {
      // Windows / no process group: fall back to killing the cargo process alone.
      try { this.proc.kill('SIGKILL'); } catch { /* already gone */ }
    }
  }
}
