#!/usr/bin/env node
import { existsSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';

const root = process.cwd();
const cacheDir = path.join(root, '.cache');
const wptDir = process.env.WPT_DIR ?? path.join(cacheDir, 'wpt');
const repo = 'https://github.com/web-platform-tests/wpt.git';

function run(cmd, args, cwd = root) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { cwd, stdio: 'inherit' });
    p.on('exit', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} ${args.join(' ')} failed with code ${code}`));
    });
  });
}

await mkdir(cacheDir, { recursive: true });

if (!existsSync(wptDir)) {
  console.log(`Cloning WPT into ${wptDir}...`);
  await run('git', ['clone', '--depth', '1', repo, wptDir]);
} else {
  console.log(`Updating WPT in ${wptDir}...`);
  await run('git', ['fetch', '--depth', '1', 'origin', 'main'], wptDir);
  await run('git', ['reset', '--hard', 'origin/main'], wptDir);
}

console.log('WPT ready.');
