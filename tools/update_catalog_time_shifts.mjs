import fs from "node:fs/promises";
import {
  FileBlob,
  SpreadsheetFile,
} from "/Users/jvidale/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const catalogPath = "/Users/jvidale/Documents/Research/FaultScanR/event_sta_info/catalog_local_hand.xlsx";
const component = (process.argv[2] || "R").toUpperCase();
const outputDir = process.argv[3] || "/Users/jvidale/Documents/Research/FaultScanR/output/20260718_174326_0912";
const shiftsPath = `${outputDir}/all_events_${component}_stack_xcorr_alignment_to_CI_40353472.xlsx`;
const timeShiftHeader = component === "R" ? "time shift" : `time shift ${component}`;
const previewPath = `/private/tmp/catalog_local_hand_time_shifts_${component}_preview.png`;

function normalizedId(value) {
  return String(value ?? "").trim();
}

async function importWorkbook(path) {
  return SpreadsheetFile.importXlsx(await FileBlob.load(path));
}

const catalog = await importWorkbook(catalogPath);
const shifts = await importWorkbook(shiftsPath);

console.log("CATALOG BEFORE");
console.log(
  (await catalog.inspect({
    kind: "workbook,sheet,region,computedStyle",
    maxChars: 8000,
    tableMaxRows: 10,
    tableMaxCols: 20,
    tableMaxCellChars: 80,
  })).ndjson,
);
console.log("SHIFTS");
console.log(
  (await shifts.inspect({
    kind: "workbook,sheet,region",
    maxChars: 8000,
    tableMaxRows: 10,
    tableMaxCols: 20,
    tableMaxCellChars: 80,
  })).ndjson,
);

const catalogSheet = catalog.worksheets.getItemAt(0);
const shiftsSheet = shifts.worksheets.getItemAt(0);
const catalogUsed = catalogSheet.getUsedRange();
const shiftsUsed = shiftsSheet.getUsedRange();
const catalogValues = catalogUsed.values;
const shiftsValues = shiftsUsed.values;

const catalogHeaders = catalogValues[0].map(normalizedId);
const shiftsHeaders = shiftsValues[0].map(normalizedId);
const evidColumn = catalogHeaders.indexOf("evid");
let timeShiftColumn = catalogHeaders.indexOf(timeShiftHeader);
const eventIdColumn = shiftsHeaders.indexOf("event_id");
const leftShiftColumn = shiftsHeaders.indexOf("shift_left_to_align_waveform_seconds");

if (evidColumn < 0) {
  throw new Error('Catalog must contain an "evid" column.');
}
if (eventIdColumn < 0 || leftShiftColumn < 0) {
  throw new Error("Alignment workbook is missing required event ID or waveform shift columns.");
}

const shiftsByEvent = new Map();
for (const row of shiftsValues.slice(1)) {
  const eventId = normalizedId(row[eventIdColumn]);
  const shift = row[leftShiftColumn];
  if (eventId) shiftsByEvent.set(eventId, shift);
}

if (timeShiftColumn < 0) {
  timeShiftColumn = catalogValues[0].length;
  catalogSheet.getCell(0, timeShiftColumn).values = [[timeShiftHeader]];
}

const unmatchedCatalogIds = [];
for (let rowIndex = 1; rowIndex < catalogValues.length; rowIndex += 1) {
  const eventId = normalizedId(catalogValues[rowIndex][evidColumn]);
  if (!eventId) continue;
  if (!shiftsByEvent.has(eventId)) {
    unmatchedCatalogIds.push(eventId);
    continue;
  }
}

const matchedRows = [];
for (let rowIndex = 1; rowIndex < catalogValues.length; rowIndex += 1) {
  const eventId = normalizedId(catalogValues[rowIndex][evidColumn]);
  if (shiftsByEvent.has(eventId)) matchedRows.push(rowIndex + 1);
}
for (const rowNumber of matchedRows) {
  const eventId = normalizedId(catalogValues[rowNumber - 1][evidColumn]);
  catalogSheet.getCell(rowNumber - 1, timeShiftColumn).values = [
    [shiftsByEvent.get(eventId)],
  ];
}

const timeShiftRange = catalogSheet.getRangeByIndexes(
  1,
  timeShiftColumn,
  catalogValues.length - 1,
  1,
);
timeShiftRange.format.numberFormat = "0.000000";
catalogSheet.getCell(0, timeShiftColumn).format.columnWidth = 15;

const output = await SpreadsheetFile.exportXlsx(catalog);
await output.save(catalogPath);

const preview = await catalog.render({
  sheetName: catalogSheet.name,
  autoCrop: "all",
  scale: 1,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

console.log(JSON.stringify({
  catalogSheet: catalogSheet.name,
  component,
  timeShiftHeader,
  totalAlignmentEvents: shiftsByEvent.size,
  updatedRows: matchedRows.length,
  unmatchedCatalogIds,
  previewPath,
}, null, 2));
console.log("CATALOG AFTER");
console.log(
  (await catalog.inspect({
    kind: "region,computedStyle",
    sheetId: catalogSheet.name,
    range: "A1:Z12",
    maxChars: 8000,
  })).ndjson,
);
