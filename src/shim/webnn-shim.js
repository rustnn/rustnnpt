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

import { env } from 'node:process';
import { buildExpectedOutputs, buildGraphJson, buildRuntimeInputs } from '../graph/build-graph-json.js';

function debugEnabled() {
  const v = env.RUSTNNPT_DEBUG || env.RUSTNN_DEBUG || '';
  return v === '1' || v === '2' || v.toLowerCase() === 'true';
}

export async function executeGraphResources(runnerClient, graphResources, contextOptions = {}) {
  const graph = buildGraphJson(graphResources);
  const inputs = buildRuntimeInputs(graphResources);
  const expectedOutputs = buildExpectedOutputs(graphResources);

  if (debugEnabled()) {
    console.error('[rustnnpt] --- input graph (webnn-graph-json) ---');
    console.error(JSON.stringify(graph, null, 2));
    console.error('[rustnnpt] --- end graph ---');
  }

  const outputs = await runnerClient.executeGraph({
    graph,
    inputs,
    expectedOutputs,
    contextOptions
  });

  return outputs;
}
