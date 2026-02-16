import vm from 'node:vm';

export function extractTestsFromSource(sourceText, sourceName = 'wpt-test.js') {
  const arrayMatch = sourceText.match(/const\s+(\w+Tests)\s*=\s*\[/);
  if (!arrayMatch) {
    throw new Error(`No <name>Tests array found in ${sourceName}`);
  }
  const arrayName = arrayMatch[1];

  let code = sourceText;
  code = code.replace(/\nwebnn_conformance_test\([\s\S]*?\);\s*$/m, '\n');

  const wrapped = `${code}\n;globalThis.__captured_tests = ${arrayName};`;
  const context = vm.createContext({
    globalThis: {},
    BigInt,
    Number,
    Math,
    Array,
    Object
  });

  try {
    vm.runInContext(wrapped, context, { filename: sourceName, timeout: 5000 });
  } catch (err) {
    throw new Error(`Failed to evaluate ${sourceName}: ${err.message}`);
  }

  const tests = context.globalThis.__captured_tests;
  if (!Array.isArray(tests)) {
    throw new Error(`Extracted test payload is not an array in ${sourceName}`);
  }
  return tests;
}
