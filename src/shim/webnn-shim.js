import { buildExpectedOutputs, buildGraphJson, buildRuntimeInputs } from '../graph/build-graph-json.js';

export async function executeGraphResources(runnerClient, graphResources, contextOptions = {}) {
  const graph = buildGraphJson(graphResources);
  const inputs = buildRuntimeInputs(graphResources);
  const expectedOutputs = buildExpectedOutputs(graphResources);

  const outputs = await runnerClient.executeGraph({
    graph,
    inputs,
    expectedOutputs,
    contextOptions
  });

  return outputs;
}
