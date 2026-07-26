import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const scriptPath = fileURLToPath(
  new URL("./update_catalog_time_shifts.py", import.meta.url),
);
const command = process.env.PYTHON || "conda";
const commandPrefix = process.env.PYTHON
  ? []
  : ["run", "-n", "vidale_main", "python"];
const result = spawnSync(
  command,
  [...commandPrefix, scriptPath, ...process.argv.slice(2)],
  {
  stdio: "inherit",
  },
);

if (result.error) {
  throw result.error;
}
process.exit(result.status ?? 1);
