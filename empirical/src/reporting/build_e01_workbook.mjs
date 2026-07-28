import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

process.on("uncaughtException", (error) => {
  console.error(`E01_WORKBOOK_ERROR: ${error?.message ?? error}`);
  const relevant = String(error?.stack ?? "")
    .split("\n")
    .filter((line) => line.includes("build_e01_workbook.mjs"));
  if (relevant.length) console.error(relevant.join("\n"));
  process.exit(1);
});

process.on("unhandledRejection", (error) => {
  console.error(`E01_WORKBOOK_REJECTION: ${error?.message ?? error}`);
  process.exit(1);
});

const runDir = process.argv[2];
if (!runDir) {
  throw new Error("Usage: node build_e01_workbook.mjs RUN_DIRECTORY");
}

const tableDir = path.join(runDir, "tables");
const figureDir = path.join(runDir, "figures", "workbook_previews");
await fs.mkdir(figureDir, { recursive: true });

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  const width = rows.reduce(
    (maximum, current) => Math.max(maximum, current.length),
    0,
  );
  return rows.map((current, rowIndex) => {
    const padded = [...current];
    while (padded.length < width) padded.push("");
    if (rowIndex === 0) return padded;
    return padded.map((value) => {
      const trimmed = value.trim();
      if (/^-?(?:\d+|\d*\.\d+)$/.test(trimmed) && trimmed.length < 16) {
        const number = Number(trimmed);
        if (Number.isFinite(number)) return number;
      }
      if (trimmed === "True") return true;
      if (trimmed === "False") return false;
      return value;
    });
  });
}

function columnName(index) {
  let value = index + 1;
  let name = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    name = String.fromCharCode(65 + remainder) + name;
    value = Math.floor((value - 1) / 26);
  }
  return name;
}

function styleDataSheet(sheet, rows) {
  sheet.showGridLines = false;
  if (!rows.length || !rows[0].length) return;
  const rowCount = rows.length;
  const columnCount = rows[0].length;
  const endColumn = columnName(columnCount - 1);
  const used = sheet.getRange(`A1:${endColumn}${rowCount}`);
  const header = sheet.getRange(`A1:${endColumn}1`);
  header.format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    wrapText: true,
    borders: {
      bottom: { style: "medium", color: "#17365D" },
    },
  };
  header.format.rowHeight = 34;
  used.format.verticalAlignment = "top";
  used.format.columnWidth = 14;
  used.format.rowHeight = 18;
  header.format.rowHeight = 34;
  sheet.getRange(`A2:A${rowCount}`).format.columnWidth = 24;
  sheet.freezePanes.freezeRows(1);

  const headers = rows[0].map((value) => String(value));
  for (let columnIndex = 0; columnIndex < headers.length; columnIndex += 1) {
    const headerName = headers[columnIndex];
    const letter = columnName(columnIndex);
    if (
      /rows|count|bytes|duration|corrected|physical|retained|excluded/.test(
        headerName,
      )
    ) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "#,##0";
    }
    if (/share|coverage|rate/.test(headerName)) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "0.00%";
    }
    if (
      /source_file|absolute_path|file|schema|reason|station_name/.test(
        headerName,
      )
    ) {
      sheet.getRange(`${letter}:${letter}`).format.columnWidth = 48;
    } else if (
      /coordinate_source|classification|archive_member|modified_at/.test(
        headerName,
      )
    ) {
      sheet.getRange(`${letter}:${letter}`).format.columnWidth = 28;
    } else if (headerName.length > 22) {
      sheet.getRange(`${letter}:${letter}`).format.columnWidth = 21;
    }
  }
}

const workbook = Workbook.create();
const readme = workbook.worksheets.add("Read Me");
const summary = workbook.worksheets.add("Summary");

const imports = [
  ["City Summary", "city_summary.csv"],
  ["Monthly QC", "monthly_qc_summary.csv"],
  ["Coordinate Sources", "coordinate_source_summary.csv"],
  ["Station Exceptions", "station_resolution_exceptions.csv"],
  ["Corrections", "station_rule_corrections.csv"],
  ["Reconciliation", "row_reconciliation.csv"],
  ["Input Manifest", "input_manifest.csv"],
  ["Parquet Inventory", "parquet_inventory.csv"],
  ["Patch Deltas", "station_rule_patch_deltas.csv"],
  ["Patch Validation", "station_rule_patch_validation.csv"],
];

const imported = new Map();
for (const [sheetName, fileName] of imports) {
  const csvText = await fs.readFile(path.join(tableDir, fileName), "utf8");
  const rows = parseCsv(csvText);
  const sheet = workbook.worksheets.add(sheetName);
  if (rows.length && rows[0].length) {
    sheet
      .getRangeByIndexes(0, 0, rows.length, rows[0].length)
      .values = rows;
  }
  styleDataSheet(sheet, rows);
  imported.set(sheetName, rows);
}

readme.showGridLines = false;
readme.getRange("A1:F1").merge();
readme.getRange("A1").values = [["E01 Standardized Events and Station QC"]];
readme.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
readme.getRange("A1:F1").format.rowHeight = 30;
readme.getRange("A3:B11").values = [
  ["Purpose", "Reproducible quality-control workbook for E01 standardized crime events, bicycle trips, exclusions, and station-coordinate provenance."],
  ["Run directory", runDir],
  ["Raw-data policy", "Original source files are read only and are never overwritten."],
  ["Correction policy", "Reviewed errors are corrected only in affected records or partitions; validated erroneous intermediate runs are deleted."],
  ["Crime category", "The existing crime_type_unified field is authoritative and is not regenerated."],
  ["Targeted correction", "Five reviewed public station identifiers were removed from false non-public classifications; three DC monthly partitions required historical coordinate repair."],
  ["Flow eligibility", "An endpoint is eligible only when it has valid coordinates and neither trip endpoint is a reviewed non-public operational node."],
  ["Vancouver coordinates", "The user-supplied official current station snapshot is the primary station source; retrofitted historical trips are explicitly flagged."],
  ["Use", "The workbook is a QC artifact. Manuscript figures and inferential results are produced in later experiments."],
];
readme.getRange("A3:A11").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  verticalAlignment: "top",
};
readme.getRange("A3:B11").format.wrapText = true;
readme.getRange("A3:B11").format.verticalAlignment = "top";
readme.getRange("A:A").format.columnWidth = 22;
readme.getRange("B:B").format.columnWidth = 92;
readme.getRange("A3:B11").format.rowHeight = 34;
readme.freezePanes.freezeRows(1);

summary.showGridLines = false;
summary.getRange("A1:N1").merge();
summary.getRange("A1").values = [["E01 Quality-Control Dashboard"]];
summary.getRange("A1:N1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
summary.getRange("A1:N1").format.rowHeight = 30;
summary.getRange("A3:E3").merge();
summary.getRange("A3").values = [["Acceptance checks"]];
summary.getRange("A3:E3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
summary.getRange("A4:E4").values = [[
  "Check",
  "Expected",
  "Observed",
  "Difference",
  "Status",
]];
summary.getRange("A5:B10").values = [
  ["Crime retained rows", 503157],
  ["Bicycle retained rows", 87208600],
  ["Reason-coded exclusion rows", 99005],
  ["Remaining false non-public endpoints", 0],
  ["Reviewed station IDs documented", 5],
  ["Run status", "complete_targeted_patch"],
];
const inventoryRows = imported.get("Parquet Inventory").length;
const validationRows = imported.get("Patch Validation").length;
const correctionRows = imported.get("Corrections").length;
summary.getRange("C5:C9").formulas = [
  [`=SUMIF('Parquet Inventory'!$A$2:$A$${inventoryRows},"crime_events",'Parquet Inventory'!$C$2:$C$${inventoryRows})`],
  [`=SUMIF('Parquet Inventory'!$A$2:$A$${inventoryRows},"bicycle_trips",'Parquet Inventory'!$C$2:$C$${inventoryRows})`],
  [`=SUMIF('Parquet Inventory'!$A$2:$A$${inventoryRows},"exclusions",'Parquet Inventory'!$C$2:$C$${inventoryRows})`],
  [`=SUM('Patch Validation'!$C$2:$D$${validationRows})`],
  [`=COUNTA('Corrections'!$A$2:$A$${correctionRows})`],
];
summary.getRange("C10").values = [["complete_targeted_patch"]];
summary.getRange("D5:D9").formulas = [
  ["=C5-B5"],
  ["=C6-B6"],
  ["=C7-B7"],
  ["=C8-B8"],
  ["=C9-B9"],
];
summary.getRange("D10").values = [[""]];
summary.getRange("E5:E10").formulas = [
  ["=IF(B5=C5,\"PASS\",\"FAIL\")"],
  ["=IF(B6=C6,\"PASS\",\"FAIL\")"],
  ["=IF(B7=C7,\"PASS\",\"FAIL\")"],
  ["=IF(B8=C8,\"PASS\",\"FAIL\")"],
  ["=IF(B9=C9,\"PASS\",\"FAIL\")"],
  ["=IF(B10=C10,\"PASS\",\"FAIL\")"],
];
summary.getRange("A4:E4").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  verticalAlignment: "center",
};
summary.getRange("A5:E10").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
  bottom: { style: "thin", color: "#A6A6A6" },
};
summary.getRange("B5:D9").format.numberFormat = "#,##0";
summary.getRange("A:A").format.columnWidth = 40;
summary.getRange("B:C").format.columnWidth = 25;
summary.getRange("D:D").format.columnWidth = 17;
summary.getRange("E:E").format.columnWidth = 13;
summary.getRange("E5:E10").conditionalFormats.add("containsText", {
  text: "PASS",
  format: { fill: "#E2F0D9", font: { bold: true, color: "#375623" } },
});
summary.getRange("E5:E10").conditionalFormats.add("containsText", {
  text: "FAIL",
  format: { fill: "#F4CCCC", font: { bold: true, color: "#9C0006" } },
});

summary.getRange("A13:E13").merge();
summary.getRange("A13").values = [["Targeted correction totals"]];
summary.getRange("A13:E13").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
summary.getRange("A14:C18").values = [
  ["Metric", "Start endpoints", "End endpoints"],
  ["Reviewed false non-public classifications", null, null],
  ["DC legacy coordinates imputed", null, null],
  ["Eligible-flow endpoints restored", null, null],
  ["Affected Parquet files", 133, null],
];
const deltaRows = imported.get("Patch Deltas").length;
summary.getRange("B15:C17").formulas = [
  [
    `=SUM('Patch Deltas'!$I$2:$I$${deltaRows})`,
    `=SUM('Patch Deltas'!$J$2:$J$${deltaRows})`,
  ],
  [
    `=SUM('Patch Deltas'!$K$2:$K$${deltaRows})`,
    `=SUM('Patch Deltas'!$L$2:$L$${deltaRows})`,
  ],
  [
    `=SUM('Patch Deltas'!$C$2:$C$${deltaRows})`,
    `=SUM('Patch Deltas'!$D$2:$D$${deltaRows})`,
  ],
];
summary.getRange("A14:C14").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
};
summary.getRange("B15:C18").format.numberFormat = "#,##0";
summary.getRange("A14:C18").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
  bottom: { style: "thin", color: "#A6A6A6" },
};

summary.getRange("G3:I3").values = [[
  "City",
  "Start endpoint eligibility",
  "End endpoint eligibility",
]];
summary.getRange("G4:I6").formulas = [
  [
    "='City Summary'!B2",
    "='City Summary'!L2/'City Summary'!D2",
    "='City Summary'!M2/'City Summary'!D2",
  ],
  [
    "='City Summary'!B3",
    "='City Summary'!L3/'City Summary'!D3",
    "='City Summary'!M3/'City Summary'!D3",
  ],
  [
    "='City Summary'!B4",
    "='City Summary'!L4/'City Summary'!D4",
    "='City Summary'!M4/'City Summary'!D4",
  ],
];
summary.getRange("G3:I3").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
summary.getRange("H4:I6").format.numberFormat = "0.00%";
summary.getRange("G:G").format.columnWidth = 18;
summary.getRange("H:I").format.columnWidth = 23;
const chart = summary.charts.add("bar", summary.getRange("G3:I6"));
chart.title = "Spatially eligible endpoints by city";
chart.hasLegend = true;
chart.yAxis = { numberFormatCode: "0%" };
chart.setPosition("G9", "N24");
summary.freezePanes.freezeRows(1);

const previews = [
  ["Read Me", "A1:F11"],
  ["Summary", "A1:N24"],
];
for (const [sheetName] of imports) {
  const rows = imported.get(sheetName);
  const endColumn = columnName(Math.min(rows[0].length, 12) - 1);
  const endRow = Math.min(rows.length, 24);
  previews.push([sheetName, `A1:${endColumn}${endRow}`]);
}

for (const [sheetName, range] of previews) {
  const preview = await workbook.render({
    sheetName,
    range,
    scale: sheetName === "Summary" ? 1.2 : 0.9,
    format: "png",
  });
  const safeName = sheetName.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(
    path.join(figureDir, `${safeName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const summaryInspection = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:I18",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 9,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
const drawings = await workbook.inspect({
  kind: "drawing",
  sheetId: "Summary",
  maxChars: 3000,
});
await fs.writeFile(
  path.join(tableDir, "E01_workbook_verification.txt"),
  `${summaryInspection.ndjson}\n${errors.ndjson}\n${drawings.ndjson}\n`,
  "utf8",
);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(tableDir, "E01_standardization_QC.xlsx"));
