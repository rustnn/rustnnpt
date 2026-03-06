#!/usr/bin/env node

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
  await run('git', ['fetch', '--depth', '1', 'origin', 'master'], wptDir);
  await run('git', ['reset', '--hard', 'origin/master'], wptDir);
}

console.log('WPT ready.');
