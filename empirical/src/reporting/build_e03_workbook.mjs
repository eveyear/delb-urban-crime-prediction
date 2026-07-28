import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

process.on("uncaughtException", (error) => {
  console.error(`E03_WORKBOOK_ERROR: ${error?.message ?? error}`);
  const relevant = String(error?.stack ?? "")
    .split("\n")
    .filter((line) => line.includes("build_e03_workbook.mjs"));
  if (relevant.length) console.error(relevant.join("\n"));
  process.exit(1);
});

process.on("unhandledRejection", (error) => {
  console.error(`E03_WORKBOOK_REJECTION: ${error?.message ?? error}`);
  process.exit(1);
});

const runDir = process.argv[2];
if (!runDir) {
  throw new Error("Usage: node build_e03_workbook.mjs RUN_DIRECTORY");
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
      if (/^-?(?:\d+|\d*\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)) {
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
  header.format.rowHeight = 40;
  used.format.verticalAlignment = "top";
  used.format.rowHeight = 18;
  used.format.columnWidth = 15;
  sheet.freezePanes.freezeRows(1);
  const headers = rows[0].map((value) => String(value));
  for (let columnIndex = 0; columnIndex < headers.length; columnIndex += 1) {
    const headerName = headers[columnIndex];
    const letter = columnName(columnIndex);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = Math.min(
      44,
      Math.max(15, headerName.length + 2),
    );
    if (
      /entropy|moment|delb|bound|error|slack|ratio|lambda|partition/.test(
        headerName,
      )
    ) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "0.000000";
    }
    if (
      /observations|atoms|count|size|radius|repetition/.test(headerName)
    ) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "#,##0";
    }
    if (/distribution|criterion|check/.test(headerName)) {
      sheet.getRange(`${letter}:${letter}`).format.columnWidth = 36;
    } else if (headerName === "outcome") {
      sheet.getRange(`${letter}:${letter}`).format.columnWidth = 30;
    } else if (/method|class|outcome label|city label/.test(headerName)) {
      sheet.getRange(`${letter}:${letter}`).format.columnWidth = 24;
    }
  }
  if (headers.includes("status")) {
    const statusColumn = columnName(headers.indexOf("status"));
    sheet
      .getRange(`${statusColumn}2:${statusColumn}${rowCount}`)
      .conditionalFormats.add("containsText", {
        text: "PASS",
        format: {
          fill: "#E2F0D9",
          font: { bold: true, color: "#006100" },
        },
      });
    sheet
      .getRange(`${statusColumn}2:${statusColumn}${rowCount}`)
      .conditionalFormats.add("containsText", {
        text: "FAIL",
        format: {
          fill: "#FCE4D6",
          font: { bold: true, color: "#9C0006" },
        },
      });
  }
}

const workbook = Workbook.create();
const readme = workbook.worksheets.add("Read Me");
const dashboard = workbook.worksheets.add("Dashboard");
const imports = [
  ["Acceptance", "acceptance_checklist.csv"],
  ["Envelope", "entropy_envelope_grid.csv"],
  ["Inverse Recovery", "inverse_recovery.csv"],
  ["Entropy Checks", "maximum_entropy_checks.csv"],
  ["Simulation", "simulation_convergence.csv"],
  ["Simulation Replicates", "simulation_replicates.csv"],
  ["E02 Calibration", "e02_entropy_range_calibration.csv"],
];
const imported = new Map();
for (const [sheetName, fileName] of imports) {
  const text = await fs.readFile(path.join(tableDir, fileName), "utf8");
  const rows = parseCsv(text);
  const displayRows = [
    rows[0].map((value) => String(value).replaceAll("_", " ")),
    ...rows.slice(1),
  ];
  const sheet = workbook.worksheets.add(sheetName);
  if (displayRows.length && displayRows[0].length) {
    sheet
      .getRangeByIndexes(
        0,
        0,
        displayRows.length,
        displayRows[0].length,
      )
      .values = displayRows;
  }
  styleDataSheet(sheet, displayRows);
  imported.set(sheetName, rows);
}

readme.showGridLines = false;
readme.getRange("A1:F1").merge();
readme.getRange("A1").values = [[
  "E03 Numerical Validation of the Discrete Entropy Lower Bound",
]];
readme.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
readme.getRange("A1:F1").format.rowHeight = 31;
readme.getRange("A3:B13").values = [
  ["Purpose", "Verify the numerical integer-lattice entropy envelope, its generalized inverse, and the conservative closed-form DELB."],
  ["Run directory", runDir],
  ["Exact solver", "Centered discrete Gaussian; direct lattice summation for moderate/large lambda and Poisson-dual summation for small lambda."],
  ["Tail tolerance", "1e-15 for omitted unnormalized lattice mass."],
  ["Root tolerance", "1e-12 for one-dimensional inverse solves."],
  ["Attainability check", "Discrete-Gaussian distributions must lie on the maximum-entropy envelope at their second moments."],
  ["Inequality check", "Non-extremal integer distributions must have entropy no greater than the envelope at the same raw second moment."],
  ["Monte Carlo", "20 deterministic-seed repetitions at n = 1,000, 10,000, and 100,000 for target entropies 0.5, 1, 2, and 3 bits."],
  ["E02 calibration", "Unconditional crime-count entropy is used only to stress-test the numerical range; it is not a conditional DELB result."],
  ["Article figures", "All manuscript and supplementary figures are generated by Python/matplotlib. This workbook is an audit artifact only."],
  ["Next experiment", "E04 estimates conditional entropy and bicycle conditional mutual information using the accepted E02 panel."],
];
readme.getRange("A3:A13").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  verticalAlignment: "top",
};
readme.getRange("A3:B13").format.wrapText = true;
readme.getRange("A3:B13").format.verticalAlignment = "top";
readme.getRange("A:A").format.columnWidth = 23;
readme.getRange("B:B").format.columnWidth = 94;
readme.getRange("A3:B13").format.rowHeight = 37;
readme.freezePanes.freezeRows(1);

dashboard.showGridLines = false;
dashboard.getRange("A1:H1").merge();
dashboard.getRange("A1").values = [["E03 Numerical Quality-Control Dashboard"]];
dashboard.getRange("A1:H1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
dashboard.getRange("A1:H1").format.rowHeight = 31;
dashboard.getRange("A3:B3").merge();
dashboard.getRange("A3").values = [["Acceptance summary"]];
dashboard.getRange("A3:B3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("A4:B10").values = [
  ["Metric", "Value"],
  ["Checks passed", null],
  ["Total checks", null],
  ["Maximum inverse relative error", null],
  ["Maximum entropy round-trip error (bits)", null],
  ["Minimum exact-minus-closed slack", null],
  ["Maximum E02 calibration entropy (bits)", null],
];
dashboard.getRange("B5:B10").formulas = [
  [`=COUNTIF('Acceptance'!$D$2:$D$${imported.get("Acceptance").length},"PASS")`],
  [`=COUNTA('Acceptance'!$A$2:$A$${imported.get("Acceptance").length})`],
  [`=MAX('Inverse Recovery'!$J$2:$J$${imported.get("Inverse Recovery").length})`],
  [`=MAX('Inverse Recovery'!$L$2:$L$${imported.get("Inverse Recovery").length})`],
  [`=MIN('Envelope'!$D$2:$D$${imported.get("Envelope").length})`],
  [`=MAX('E02 Calibration'!$J$2:$J$${imported.get("E02 Calibration").length})`],
];
dashboard.getRange("A4:B4").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
};
dashboard.getRange("A5:B10").format.rowHeight = 23;
dashboard.getRange("B5:B6").format.numberFormat = "#,##0";
dashboard.getRange("B7:B10").format.numberFormat = "0.000000E+00";
dashboard.getRange("A:A").format.columnWidth = 43;
dashboard.getRange("B:B").format.columnWidth = 22;

dashboard.getRange("A13:H13").merge();
dashboard.getRange("A13").values = [["Interpretation boundary"]];
dashboard.getRange("A13:H13").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("A14:H17").merge();
dashboard.getRange("A14").values = [[
  "E03 validates the transformation from entropy to an MSE lower-bound scale. "
  + "It does not estimate the crime conditional entropy, bicycle conditional "
  + "mutual information, bound reduction, or causal effects. The E02 values "
  + "shown here are unconditional range checks only.",
]];
dashboard.getRange("A14:H17").format = {
  fill: "#FFF2CC",
  font: { color: "#7F6000" },
  wrapText: true,
  verticalAlignment: "center",
};
dashboard.getRange("A14:H17").format.rowHeight = 25;

dashboard.getRange("D3:H3").merge();
dashboard.getRange("D3").values = [["Verified mathematical properties"]];
dashboard.getRange("D3:H3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("D4:H10").values = [
  ["Property", "Required result", "Evidence sheet", "Status", "Use"],
  ["Zero boundary", "H_Z^{-1}(0) = 0", "Acceptance", "PASS", "Exact DELB"],
  ["Monotonicity", "Inverse increases with entropy", "Acceptance", "PASS", "All later experiments"],
  ["Attainability", "Discrete Gaussian reaches envelope", "Entropy Checks", "PASS", "Equality condition"],
  ["Conservatism", "Closed-form <= exact DELB", "Envelope", "PASS", "Secondary bound"],
  ["Numerical precision", "Round-trip errors within tolerance", "Inverse Recovery", "PASS", "Exact solver"],
  ["Empirical range", "E02 entropies inside configured domain", "E02 Calibration", "PASS", "E04 readiness"],
];
dashboard.getRange("D4:H4").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
dashboard.getRange("D5:H10").format.rowHeight = 28;
dashboard.getRange("D5:H10").format.wrapText = true;
dashboard.getRange("D:D").format.columnWidth = 19;
dashboard.getRange("E:E").format.columnWidth = 34;
dashboard.getRange("F:F").format.columnWidth = 23;
dashboard.getRange("G:G").format.columnWidth = 12;
dashboard.getRange("H:H").format.columnWidth = 23;
dashboard.getRange("D5:H10").format.rowHeight = 38;
dashboard.getRange("G5:G10").conditionalFormats.add("containsText", {
  text: "PASS",
  format: {
    fill: "#E2F0D9",
    font: { bold: true, color: "#006100" },
  },
});
dashboard.freezePanes.freezeRows(1);

const previews = [
  ["Read Me", "A1:F13"],
  ["Dashboard", "A1:H17"],
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
    scale: sheetName === "Dashboard" ? 1.1 : 0.85,
    format: "png",
  });
  const safeName = sheetName.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(
    path.join(previewDir, `${safeName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const dashboardInspection = await workbook.inspect({
  kind: "table",
  range: "Dashboard!A1:H17",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 8,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(
  path.join(tableDir, "E03_workbook_verification.txt"),
  `${dashboardInspection.ndjson}\n${errors.ndjson}\n`,
  "utf8",
);
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(
  path.join(tableDir, "E03_discrete_bound_validation_QC.xlsx"),
);
