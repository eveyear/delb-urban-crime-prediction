import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

process.on("uncaughtException", (error) => {
  console.error(`AUDIT_WORKBOOK_ERROR: ${error?.message ?? error}`);
  const relevant = String(error?.stack ?? "")
    .split("\n")
    .filter((line) => line.includes("build_audit_workbook.mjs"));
  if (relevant.length) console.error(relevant.join("\n"));
  process.exit(1);
});

process.on("unhandledRejection", (error) => {
  console.error(`AUDIT_WORKBOOK_REJECTION: ${error?.message ?? error}`);
  const relevant = String(error?.stack ?? "")
    .split("\n")
    .filter((line) => line.includes("build_audit_workbook.mjs"));
  if (relevant.length) console.error(relevant.join("\n"));
  process.exit(1);
});

const runDir = process.argv[2];
if (!runDir) {
  throw new Error("Usage: node build_audit_workbook.mjs RUN_DIRECTORY");
}

const tableDir = path.join(runDir, "tables");
const figureDir = path.join(runDir, "figures");
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
  const width = rows.reduce((maximum, current) => Math.max(maximum, current.length), 0);
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

const workbook = Workbook.create();
const readme = workbook.worksheets.add("Read Me");
readme.showGridLines = false;
readme.getRange("A1:F1").merge();
readme.getRange("A1").values = [["E00 Data Integrity Audit"]];
readme.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
readme.getRange("A1:F1").format.rowHeight = 30;
readme.getRange("A3:B8").values = [
  ["Purpose", "Inventory and quality-control audit before any cleaning."],
  ["Raw-data policy", "Source files are read only and are never overwritten."],
  ["Crime category", "crime_type_unified is the authoritative harmonized field."],
  ["Run directory", runDir],
  ["Primary outputs", "Summary, issues, source-file audits, category counts, and manifest."],
  ["Interpretation", "An audit issue is a documented cleaning action, not a silent exclusion."],
];
readme.getRange("A3:A8").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
};
readme.getRange("A3:B8").format.wrapText = true;
readme.getRange("A3:B8").format.verticalAlignment = "top";
readme.getRange("A3:A8").format.columnWidth = 22;
readme.getRange("B3:B8").format.columnWidth = 78;
readme.freezePanes.freezeRows(1);

const imports = [
  ["Summary", "audit_summary.csv"],
  ["Issues", "issues.csv"],
  ["Crime Files", "crime_file_audit.csv"],
  ["Crime Types", "crime_type_counts.csv"],
  ["Bicycle Files", "bicycle_file_audit.csv"],
  ["Bicycle Members", "bicycle_member_audit.csv"],
  ["Bicycle Schemas", "bicycle_schema_catalog.csv"],
  ["Input Manifest", "input_manifest.csv"],
];

for (const [sheetName, fileName] of imports) {
  const csvText = await fs.readFile(path.join(tableDir, fileName), "utf8");
  const rows = parseCsv(csvText);
  const sheet = workbook.worksheets.add(sheetName);
  if (rows.length && rows[0].length) {
    sheet.getRangeByIndexes(0, 0, rows.length, rows[0].length).values = rows;
  }
  sheet.showGridLines = false;
  const used = sheet.getUsedRange();
  if (used) {
    const header = used.getRow(0);
    header.format = {
      fill: "#1F4E78",
      font: { bold: true, color: "#FFFFFF" },
      verticalAlignment: "center",
      wrapText: true,
      borders: { preset: "outside", style: "thin", color: "#A6A6A6" },
    };
    used.format.verticalAlignment = "top";
    used.format.autofitColumns();
    used.format.autofitRows();
  }
  sheet.freezePanes.freezeRows(1);
}

const summary = workbook.worksheets.getItem("Summary");
summary.getRange("A:A").format.columnWidth = 42;
summary.getRange("B:B").format.columnWidth = 18;
summary.getRange("C:C").format.columnWidth = 14;
summary.getRange("B2:B15").format.numberFormat = "#,##0";

const issues = workbook.worksheets.getItem("Issues");
issues.getRange("A:A").format.columnWidth = 12;
issues.getRange("B:B").format.columnWidth = 12;
issues.getRange("C:C").format.columnWidth = 14;
issues.getRange("D:D").format.columnWidth = 12;
issues.getRange("E:E").format.columnWidth = 32;
issues.getRange("F:F").format.columnWidth = 16;
issues.getRange("G:H").format.columnWidth = 48;
issues.getRange("A:H").format.wrapText = true;
issues.getRange("D2:D1000").conditionalFormats.add("containsText", {
  text: "BLOCKER",
  format: { fill: "#F4CCCC", font: { bold: true, color: "#9C0006" } },
});
issues.getRange("D2:D1000").conditionalFormats.add("containsText", {
  text: "HIGH",
  format: { fill: "#FCE5CD", font: { bold: true, color: "#B45F06" } },
});
issues.getRange("D2:D1000").conditionalFormats.add("containsText", {
  text: "MEDIUM",
  format: { fill: "#FFF2CC", font: { color: "#7F6000" } },
});

const crimeFiles = workbook.worksheets.getItem("Crime Files");
crimeFiles.getRange("A:A").format.columnWidth = 58;
crimeFiles.getRange("D:Q").format.numberFormat = "#,##0";

const bikeFiles = workbook.worksheets.getItem("Bicycle Files");
bikeFiles.getRange("A:A").format.columnWidth = 62;
bikeFiles.getRange("F:R").format.numberFormat = "#,##0";

const manifest = workbook.worksheets.getItem("Input Manifest");
manifest.getRange("D:E").format.columnWidth = 62;
manifest.getRange("H:H").format.numberFormat = "#,##0";
manifest.getRange("J:J").format.columnWidth = 66;

const summaryPreview = await workbook.render({
  sheetName: "Summary",
  range: "A1:C15",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(
  path.join(figureDir, "audit_summary_workbook_preview.png"),
  new Uint8Array(await summaryPreview.arrayBuffer()),
);

const issuesPreview = await workbook.render({
  sheetName: "Issues",
  range: "A1:H30",
  scale: 1.2,
  format: "png",
});
await fs.writeFile(
  path.join(figureDir, "audit_issues_workbook_preview.png"),
  new Uint8Array(await issuesPreview.arrayBuffer()),
);

const inspection = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:C30",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 3,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
await fs.writeFile(
  path.join(tableDir, "workbook_verification.txt"),
  `${inspection.ndjson}\n${errors.ndjson}\n`,
  "utf8",
);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(tableDir, "E00_data_integrity_audit.xlsx"));
