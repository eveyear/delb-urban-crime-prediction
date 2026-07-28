import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

process.on("uncaughtException", (error) => {
  console.error(`E02_WORKBOOK_ERROR: ${error?.message ?? error}`);
  const relevant = String(error?.stack ?? "")
    .split("\n")
    .filter((line) => line.includes("build_e02_workbook.mjs"));
  if (relevant.length) console.error(relevant.join("\n"));
  process.exit(1);
});

process.on("unhandledRejection", (error) => {
  console.error(`E02_WORKBOOK_REJECTION: ${error?.message ?? error}`);
  process.exit(1);
});

const runDir = process.argv[2];
if (!runDir) {
  throw new Error("Usage: node build_e02_workbook.mjs RUN_DIRECTORY");
}
const tableDir = path.join(runDir, "tables");
const previewDir = path.join(runDir, "figures", "workbook_previews");
await fs.mkdir(previewDir, { recursive: true });

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
      if (/^-?(?:\d+|\d*\.\d+)$/.test(trimmed) && trimmed.length < 24) {
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
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
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
    borders: { bottom: { style: "medium", color: "#17365D" } },
  };
  header.format.rowHeight = 36;
  used.format.verticalAlignment = "top";
  used.format.columnWidth = 14;
  used.format.rowHeight = 18;
  header.format.rowHeight = 36;
  sheet.getRange(`A2:A${rowCount}`).format.columnWidth = 24;
  sheet.freezePanes.freezeRows(1);
  const headers = rows[0].map((value) => String(value));
  for (let columnIndex = 0; columnIndex < headers.length; columnIndex += 1) {
    const headerName = headers[columnIndex];
    const letter = columnName(columnIndex);
    if (
      /rows|count|events|flow|trips|grids|observations|eligible/.test(
        headerName,
      )
    ) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "#,##0";
    }
    if (
      /share|coverage|ratio|correlation/.test(headerName)
    ) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "0.00%";
    }
    if (/mean|std|p50|p90|p99|minimum|maximum/.test(headerName)) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "#,##0.000";
    }
    if (headerName === "check") {
      sheet.getRange(`${letter}:${letter}`).format.columnWidth = 52;
    } else if (/file|path/.test(headerName)) {
      sheet.getRange(`${letter}:${letter}`).format.columnWidth = 52;
    } else if (headerName === "coordinate_source") {
      sheet.getRange(`${letter}:${letter}`).format.columnWidth = 40;
    } else if (headerName === "disposition") {
      sheet.getRange(`${letter}:${letter}`).format.columnWidth = 30;
    } else if (/crime_type|grid_id|variable/.test(headerName)) {
      sheet.getRange(`${letter}:${letter}`).format.columnWidth = 26;
    } else if (headerName.length > 22) {
      sheet.getRange(`${letter}:${letter}`).format.columnWidth = 21;
    }
  }
}

const workbook = Workbook.create();
const readme = workbook.worksheets.add("Read Me");
const summary = workbook.worksheets.add("Summary");
const imports = [
  ["Acceptance", "acceptance_checklist.csv"],
  ["Panel Integrity", "panel_integrity_checks.csv"],
  ["City Panel", "city_panel_summary.csv"],
  ["Findings", "descriptive_findings_summary.csv"],
  ["Variable Summary", "variable_summary.csv"],
  ["Monthly Patterns", "monthly_patterns.csv"],
  ["Weekday Patterns", "weekday_patterns_normalized.csv"],
  ["Spatial Totals", "spatial_grid_totals.csv"],
  ["Bike Reconciliation", "bicycle_spatial_reconciliation.csv"],
  ["Coordinate Domain", "coordinate_domain_summary.csv"],
  ["Outside Crime", "crime_outside_primary_domain.csv"],
];
const imported = new Map();
for (const [sheetName, fileName] of imports) {
  const text = await fs.readFile(path.join(tableDir, fileName), "utf8");
  const rows = parseCsv(text);
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
readme.getRange("A1").values = [["E02 Spatial-Temporal Panel and Descriptive QC"]];
readme.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
readme.getRange("A1:F1").format.rowHeight = 30;
readme.getRange("A3:B12").values = [
  ["Purpose", "Audit the 1 km daily crime-and-bicycle panel used by later entropy and prediction experiments."],
  ["Run directory", runDir],
  ["Source", "Accepted E01 standardized events and reason-coded endpoint eligibility."],
  ["Primary domain", "Projected 1 km cells containing at least one crime event during the 2020-2021 training period."],
  ["Leakage control", "Validation/test crime locations do not determine which grids enter the primary domain."],
  ["Crime target", "Original integer daily count of PROPERTY_THEFT, VEHICLE_THEFT, and BURGLARY combined; category counts remain separate."],
  ["Bicycle variables", "Daily outbound, inbound, total, and net endpoint flows; lag-1 variables are strictly previous local-calendar day."],
  ["Zero filling", "Every primary grid has an explicit row for every day from 2020-01-01 through 2022-12-31."],
  ["Coverage caveat", "A zero flow outside the observed bicycle-system footprint does not mean zero total human mobility."],
  ["Weather", "Not included because no versioned, forecast-available weather source has passed provenance review."],
];
readme.getRange("A3:A12").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  verticalAlignment: "top",
};
readme.getRange("A3:B12").format.wrapText = true;
readme.getRange("A3:B12").format.verticalAlignment = "top";
readme.getRange("A:A").format.columnWidth = 22;
readme.getRange("B:B").format.columnWidth = 92;
readme.getRange("A3:B12").format.rowHeight = 34;
readme.freezePanes.freezeRows(1);

summary.showGridLines = false;
summary.getRange("A1:N1").merge();
summary.getRange("A1").values = [["E02 Quality-Control Dashboard"]];
summary.getRange("A1:N1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
summary.getRange("A1:N1").format.rowHeight = 30;
summary.getRange("A3:D3").merge();
summary.getRange("A3").values = [["Panel scale"]];
summary.getRange("A3:D3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
summary.getRange("A4:B10").values = [
  ["Metric", "Value"],
  ["Active 1 km grids", null],
  ["Complete grid-days", null],
  ["E01 crime events", null],
  ["Crime events in primary domain", null],
  ["E01 bicycle trip rows", null],
  ["Eligible bicycle endpoints in panel", null],
];
summary.getRange("B5:B10").formulas = [
  ["=SUM('City Panel'!B2:B4)"],
  ["=SUM('City Panel'!D2:D4)"],
  ["=SUM('City Panel'!E2:E4)"],
  ["=SUM('City Panel'!F2:F4)"],
  ["=SUM('City Panel'!H2:H4)"],
  ["=SUM('City Panel'!O2:O4)"],
];
summary.getRange("A4:B4").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
};
summary.getRange("B5:B10").format.numberFormat = "#,##0";
summary.getRange("A:A").format.columnWidth = 39;
summary.getRange("B:B").format.columnWidth = 22;

summary.getRange("A13:E13").merge();
summary.getRange("A13").values = [["Acceptance checks"]];
summary.getRange("A13:E13").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
summary.getRange("A14:E14").values = [[
  "Check",
  "Expected",
  "Observed",
  "Difference",
  "Status",
]];
const acceptanceRows = imported.get("Acceptance").length;
for (let rowIndex = 0; rowIndex < acceptanceRows - 1; rowIndex += 1) {
  const targetRow = 15 + rowIndex;
  const sourceRow = 2 + rowIndex;
  summary.getRange(`A${targetRow}:C${targetRow}`).formulas = [[
    `='Acceptance'!A${sourceRow}`,
    `='Acceptance'!B${sourceRow}`,
    `='Acceptance'!C${sourceRow}`,
  ]];
  summary.getRange(`D${targetRow}`).formulas = [[
    `=C${targetRow}-B${targetRow}`,
  ]];
  summary.getRange(`E${targetRow}`).formulas = [[
    `=IF(D${targetRow}=0,"PASS","FAIL")`,
  ]];
}
const acceptanceEnd = 14 + acceptanceRows - 1;
summary.getRange("A14:E14").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
};
summary
  .getRange(`B15:D${acceptanceEnd}`)
  .format.numberFormat = "#,##0";
summary.getRange(`E15:E${acceptanceEnd}`).conditionalFormats.add(
  "containsText",
  {
    text: "PASS",
    format: { fill: "#E2F0D9", font: { bold: true, color: "#375623" } },
  },
);
summary.getRange(`E15:E${acceptanceEnd}`).conditionalFormats.add(
  "containsText",
  {
    text: "FAIL",
    format: { fill: "#F4CCCC", font: { bold: true, color: "#9C0006" } },
  },
);
summary.getRange("C:D").format.columnWidth = 17;
summary.getRange("E:E").format.columnWidth = 13;

summary.getRange("G3:K3").values = [[
  "City",
  "Crime-domain retention",
  "Training bike-grid coverage",
  "Zero crime grid-days",
  "Zero bike grid-days",
]];
summary.getRange("G4:K6").formulas = [
  [
    "='City Panel'!A2",
    "='City Panel'!U2",
    "='City Panel'!T2",
    "='City Panel'!Q2",
    "='City Panel'!R2",
  ],
  [
    "='City Panel'!A3",
    "='City Panel'!U3",
    "='City Panel'!T3",
    "='City Panel'!Q3",
    "='City Panel'!R3",
  ],
  [
    "='City Panel'!A4",
    "='City Panel'!U4",
    "='City Panel'!T4",
    "='City Panel'!Q4",
    "='City Panel'!R4",
  ],
];
summary.getRange("G3:K3").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
summary.getRange("H4:K6").format.numberFormat = "0.0%";
summary.getRange("G:G").format.columnWidth = 15;
summary.getRange("H:K").format.columnWidth = 23;
const coverageChart = summary.charts.add(
  "bar",
  summary.getRange("G3:I6"),
);
coverageChart.title = "Crime retention and training bicycle coverage";
coverageChart.hasLegend = true;
coverageChart.yAxis = { numberFormatCode: "0%" };
coverageChart.setPosition("G9", "N24");
summary.freezePanes.freezeRows(1);

const previews = [
  ["Read Me", "A1:F12"],
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
    path.join(previewDir, `${safeName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const summaryInspection = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:K21",
  include: "values,formulas",
  tableMaxRows: 24,
  tableMaxCols: 11,
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
  path.join(tableDir, "E02_workbook_verification.txt"),
  `${summaryInspection.ndjson}\n${errors.ndjson}\n${drawings.ndjson}\n`,
  "utf8",
);
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(
  path.join(tableDir, "E02_spatial_temporal_panel_QC.xlsx"),
);
