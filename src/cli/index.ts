#!/usr/bin/env node

import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { bootstrap, maybeRunBootstrap } from "./bootstrap.js";
import { main as runMain } from "./main.js";
import { setup } from "./setup.js";
import { main as statusMain } from "./status.js";
import { main as tuiMain } from "./tui.js";

/**
 * Dispatches the public mahout-bench command surface.
 */
export async function main(argv = process.argv.slice(2)): Promise<number> {
  const filteredArgv = argv.filter((entry) => entry !== "--no-bootstrap");
  await maybeRunBootstrap(argv);
  argv = filteredArgv;
  const [command, ...rest] = argv;
  if (!command || command === "run") {
    return runMain(rest);
  }
  if (command === "bootstrap") {
    return bootstrap(rest);
  }
  if (command === "setup") {
    const shouldBootstrap = rest.includes("--bootstrap");
    const setupArgs = rest.filter((entry) => entry !== "--bootstrap");
    const code = await setup(setupArgs);
    if (code !== 0 || !shouldBootstrap) {
      return code;
    }
    return bootstrap(["--skip-setup", ...setupArgs.filter((entry, index) => entry === "--data-root" || setupArgs[index - 1] === "--data-root")]);
  }
  if (command === "status") {
    return statusMain(rest);
  }
  if (command === "tui") {
    if (rest.includes("--help") || rest.includes("-h")) {
      process.stdout.write("mahout-bench tui\n\nStarts the interactive benchmark TUI.\n");
      return 0;
    }
    return tuiMain();
  }
  if (command === "--help" || command === "-h") {
    printHelp();
    return 0;
  }
  throw new Error(`Unknown mahout-bench command: ${command}`);
}

function printHelp(): void {
  process.stdout.write(`mahout-bench\n\n`);
  process.stdout.write(`Commands:\n`);
  process.stdout.write(`  bootstrap   Configure data, secrets, and local/remote model backends.\n`);
  process.stdout.write(`  setup       Download and verify the public data bundle.\n`);
  process.stdout.write(`  status      List runs or report status/ETA for one run.\n`);
  process.stdout.write(`  run         Run the non-interactive benchmark CLI.\n`);
  process.stdout.write(`  tui         Start the interactive TUI.\n\n`);
  process.stdout.write(`Global flags:\n`);
  process.stdout.write(`  --no-bootstrap  Skip first-run bootstrap prompt for this invocation.\n\n`);
  process.stdout.write(`Examples:\n`);
  process.stdout.write(`  mahout-bench bootstrap\n`);
  process.stdout.write(`  mahout-bench setup\n`);
  process.stdout.write(`  mahout-bench status\n`);
  process.stdout.write(`  mahout-bench run --dry-smoke\n`);
  process.stdout.write(`  mahout-bench tui\n`);
}

if (path.resolve(process.argv[1] ?? "") === path.resolve(fileURLToPath(import.meta.url))) {
  main().then(
    (code) => process.exit(code),
    (error) => {
      process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
      process.exit(1);
    }
  );
}
