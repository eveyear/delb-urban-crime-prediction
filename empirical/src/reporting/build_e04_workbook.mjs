import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

process.on("uncaughtException", (error) => {
  console.error(`E04_WORKBOOK_ERROR: ${error?.message ?? error}`);
  const relevant = String(error?.stack ?? "")
    .split("\n")
    .filter((line) => line.includes("build_e04_workbook.mjs"));
  if (relevant.length) console.error(relevant.join("\n"));
  process.exit(1);
});

process.on("unhandledRejection", (error) => {
  console.error(`E04_WORKBOOK_REJECTION: ${error?.message ?? error}`);
  process.exit(1);
});

const runDir = process.argv[2];
if (!runDir) {
  throw new Error("Usage: node build_e04_workbook.mjs RUN_DIRECTORY");
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
    const character = text[index];
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
    } else if (character === '"') {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
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
        const numeric = Number(trimmed);
        if (Number.isFinite(numeric)) return numeric;
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

function styleDataSheet(sheet, rows, rawHeaders) {
  sheet.showGridLines = false;
  if (!rows.length || !rows[0].length) return;
  const rowCount = rows.length;
  const columnCount = rows[0].length;
  const endColumn = columnName(columnCount - 1);
  const used = sheet.getRange(`A1:${endColumn}${rowCount}`);
  used.format.verticalAlignment = "top";
  used.format.rowHeight = 18;
  used.format.columnWidth = 14;
  const header = sheet.getRange(`A1:${endColumn}1`);
  header.format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    wrapText: true,
    borders: { bottom: { style: "medium", color: "#17365D" } },
  };
  header.format.rowHeight = 43;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);

  rawHeaders.forEach((headerName, columnIndex) => {
    const letter = columnName(columnIndex);
    const displayName = headerName.replaceAll("_", " ");
    let width = Math.min(30, Math.max(13, displayName.length + 2));
    if (headerName === "spec_id") {
      width = 56;
    } else if (headerName === "period") {
      width = 22;
    } else if (/criterion|information_evidence|domain_label/.test(headerName)) {
      width = 32;
    } else if (/city_label|period_label|domain$/.test(headerName)) {
      width = 24;
    }
    sheet.getRange(`${letter}1:${letter}${rowCount}`).format.columnWidth = width;
    if (/sample_size|observations|atoms|states|grids|days|blocks|replicate/.test(headerName)) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "#,##0";
    } else if (/share|relative_l_reduction/.test(headerName)) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "0.00%";
    } else if (/bits|mse|bias|_se$|ci_|tertile|error|observed/.test(headerName)) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "0.000000";
    }
  });

  const statusIndex = rawHeaders.indexOf("status");
  if (statusIndex >= 0) {
    const letter = columnName(statusIndex);
    const statusRange = sheet.getRange(`${letter}2:${letter}${rowCount}`);
    statusRange.conditionalFormats.add("containsText", {
      text: "PASS",
      format: {
        fill: "#E2F0D9",
        font: { bold: true, color: "#006100" },
      },
    });
    statusRange.conditionalFormats.add("containsText", {
      text: "FAIL",
      format: {
        fill: "#FCE4D6",
        font: { bold: true, color: "#9C0006" },
      },
    });
  }
  const evidenceIndex = rawHeaders.indexOf("information_evidence");
  if (evidenceIndex >= 0) {
    const letter = columnName(evidenceIndex);
    const evidenceRange = sheet.getRange(`${letter}2:${letter}${rowCount}`);
    evidenceRange.conditionalFormats.add("containsText", {
      text: "positive_with_centered_95pct_ci",
      format: { fill: "#E2F0D9", font: { color: "#006100" } },
    });
    evidenceRange.conditionalFormats.add("containsText", {
      text: "nonpositive",
      format: { fill: "#FCE4D6", font: { color: "#9C0006" } },
    });
  }
}

const imports = [
  ["Acceptance", "acceptance_checklist.csv"],
  ["Primary Results", "primary_city_results.csv"],
  ["All Estimates", "city_information_estimates.csv"],
  ["Bin Thresholds", "bicycle_bin_thresholds.csv"],
  ["Input Diagnostics", "city_input_diagnostics.csv"],
  ["Sparsity", "state_sparsity_diagnostics.csv"],
];

const workbook = Workbook.create();
const readme = workbook.worksheets.add("Read Me");
const dashboard = workbook.worksheets.add("Dashboard");
const imported = new Map();
for (const [sheetName, fileName] of imports) {
  const rows = parseCsv(await fs.readFile(path.join(tableDir, fileName), "utf8"));
  const displayRows = [
    rows[0].map((value) => String(value).replaceAll("_", " ")),
    ...rows.slice(1),
  ];
  const sheet = workbook.worksheets.add(sheetName);
  sheet
    .getRangeByIndexes(0, 0, displayRows.length, displayRows[0].length)
    .values = displayRows;
  styleDataSheet(sheet, displayRows, rows[0].map(String));
  imported.set(sheetName, {
    rows,
    headers: rows[0].map(String),
    rowCount: rows.length,
  });
}

function dataColumn(sheetName, header) {
  const index = imported.get(sheetName).headers.indexOf(header);
  if (index < 0) throw new Error(`Missing ${header} in ${sheetName}`);
  return columnName(index);
}

function sourceCell(sheetName, header, row) {
  return `'${sheetName}'!$${dataColumn(sheetName, header)}$${row}`;
}

readme.showGridLines = false;
readme.getRange("A1:F1").merge();
readme.getRange("A1").values = [[
  "E04 Bicycle Information Value and Discrete Prediction Bounds",
]];
readme.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
readme.getRange("A1:F1").format.rowHeight = 31;
readme.getRange("A3:B15").values = [
  ["Purpose", "Estimate how much strictly lagged bicycle mobility reduces conditional crime-count uncertainty and the exact discrete entropy lower bound (DELB)."],
  ["Run directory", runDir],
  ["Target", "Raw nonnegative integer daily crime_count_all."],
  ["Baseline state", "Grid, lagged crime bin, day of week, season, and holiday indicator."],
  ["Bicycle state", "Strict lag-1 total bicycle flow: zero plus city-specific positive tertiles fitted only on 2020–2021 training data."],
  ["Primary domain", "Grids observed by the bicycle system during training; complete crime domain is retained as a sensitivity analysis."],
  ["Point estimator", "Miller–Madow conditional entropy; plugin estimates are retained for audit."],
  ["Uncertainty", "1,000 deterministic-seed 7-day block-bootstrap replicates per specification."],
  ["Reported interval", "Centered 95% normal interval based on block-bootstrap standard error."],
  ["Percentile diagnostic", "Raw percentile endpoints and bootstrap bias are retained because resampling can remove sparse support atoms and shift discrete entropy estimates."],
  ["Theoretical projection", "Raw conditional mutual information is reported unchanged; only the value inserted into the theoretical bound is projected to be nonnegative."],
  ["Interpretation", "The information gain is predictive and associational. It is neither a causal bicycle effect nor a guaranteed realized MSE improvement for a fitted model."],
  ["Article figures", "Every manuscript and supplementary figure is generated by Python/matplotlib. This Excel workbook is a quality-control and audit artifact only."],
];
readme.getRange("A3:A15").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  verticalAlignment: "top",
};
readme.getRange("A3:B15").format.wrapText = true;
readme.getRange("A3:B15").format.verticalAlignment = "top";
readme.getRange("A:A").format.columnWidth = 24;
readme.getRange("B:B").format.columnWidth = 96;
readme.getRange("A3:B15").format.rowHeight = 40;
readme.freezePanes.freezeRows(1);

dashboard.showGridLines = false;
dashboard.getRange("A1:L1").merge();
dashboard.getRange("A1").values = [[
  "E04 Conditional Information and Exact DELB Dashboard",
]];
dashboard.getRange("A1:L1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
dashboard.getRange("A1:L1").format.rowHeight = 31;

dashboard.getRange("A3:B3").merge();
dashboard.getRange("A3").values = [["Run acceptance"]];
dashboard.getRange("A3:B3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("A4:B8").values = [
  ["Metric", "Value"],
  ["Checks passed", null],
  ["Total checks", null],
  ["Specifications", null],
  ["Bootstrap rows", null],
];
dashboard.getRange("B5:B8").formulas = [
  [`=COUNTIF('Acceptance'!$D$2:$D$${imported.get("Acceptance").rowCount},"PASS")`],
  [`=COUNTA('Acceptance'!$A$2:$A$${imported.get("Acceptance").rowCount})`],
  [`=${sourceCell("Acceptance", "observed", 2)}`],
  [`=${sourceCell("Acceptance", "observed", 3)}`],
];
dashboard.getRange("A4:B4").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
};
dashboard.getRange("B5:B8").format.numberFormat = "#,##0";
dashboard.getRange("A:A").format.columnWidth = 35;
dashboard.getRange("B:B").format.columnWidth = 18;

dashboard.getRange("D3:L3").merge();
dashboard.getRange("D3").values = [["Primary pooled estimates: training bicycle-covered domain"]];
dashboard.getRange("D3:L3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("D4:L8").values = [
  ["City", "N", "H0 (bits)", "HB (bits)", "CMI (bits)", "95% CI low", "95% CI high", "Exact DELB reduction", "Relative reduction"],
  [null, null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null, null],
  ["Definition", null, "Baseline entropy", "Bike-aware entropy", "Raw H0−HB", null, null, "L0−LB", "(L0−LB)/L0"],
];
for (let index = 0; index < 3; index += 1) {
  const dashboardRow = 5 + index;
  const sourceRow = 2 + index;
  dashboard.getRange(`D${dashboardRow}:L${dashboardRow}`).formulas = [[
    `=${sourceCell("Primary Results", "city_label", sourceRow)}`,
    `=${sourceCell("Primary Results", "sample_size", sourceRow)}`,
    `=${sourceCell("Primary Results", "h0_miller_madow_bits", sourceRow)}`,
    `=${sourceCell("Primary Results", "hb_miller_madow_bits", sourceRow)}`,
    `=${sourceCell("Primary Results", "delta_h_miller_madow_raw_bits", sourceRow)}`,
    `=${sourceCell("Primary Results", "delta_h_miller_madow_raw_bits_ci_low", sourceRow)}`,
    `=${sourceCell("Primary Results", "delta_h_miller_madow_raw_bits_ci_high", sourceRow)}`,
    `=${sourceCell("Primary Results", "delta_l_exact_mse", sourceRow)}`,
    `=${sourceCell("Primary Results", "relative_l_reduction", sourceRow)}`,
  ]];
}
dashboard.getRange("D4:L4").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
dashboard.getRange("D5:D8").format.columnWidth = 21;
dashboard.getRange("E:E").format.columnWidth = 13;
dashboard.getRange("F:L").format.columnWidth = 16;
dashboard.getRange("E5:E7").format.numberFormat = "#,##0";
dashboard.getRange("F5:K7").format.numberFormat = "0.000000";
dashboard.getRange("L5:L7").format.numberFormat = "0.00%";
dashboard.getRange("D8:L8").format = {
  fill: "#F2F2F2",
  font: { italic: true, color: "#595959" },
  wrapText: true,
};

dashboard.getRange("A11:F11").merge();
dashboard.getRange("A11").values = [["Spatial-domain sensitivity"]];
dashboard.getRange("A11:F11").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("A12:F15").values = [
  ["City", "Complete-domain CMI", "Covered-domain CMI", "Covered / complete", "Complete-domain ΔL", "Covered-domain ΔL"],
  ["DC", null, null, null, null, null],
  ["NY", null, null, null, null, null],
  ["VAN", null, null, null, null, null],
];
const allRows = imported.get("All Estimates").rowCount;
const allCity = dataColumn("All Estimates", "city");
const allDomain = dataColumn("All Estimates", "domain");
const allPeriod = dataColumn("All Estimates", "period");
const allCmi = dataColumn("All Estimates", "delta_h_miller_madow_raw_bits");
const allDeltaL = dataColumn("All Estimates", "delta_l_exact_mse");
for (let row = 13; row <= 15; row += 1) {
  dashboard.getRange(`B${row}:F${row}`).formulas = [[
    `=SUMIFS('All Estimates'!$${allCmi}$2:$${allCmi}$${allRows},'All Estimates'!$${allCity}$2:$${allCity}$${allRows},$A${row},'All Estimates'!$${allDomain}$2:$${allDomain}$${allRows},"complete_crime_domain",'All Estimates'!$${allPeriod}$2:$${allPeriod}$${allRows},"pooled_2020_2022")`,
    `=SUMIFS('All Estimates'!$${allCmi}$2:$${allCmi}$${allRows},'All Estimates'!$${allCity}$2:$${allCity}$${allRows},$A${row},'All Estimates'!$${allDomain}$2:$${allDomain}$${allRows},"bike_covered_training",'All Estimates'!$${allPeriod}$2:$${allPeriod}$${allRows},"pooled_2020_2022")`,
    `=C${row}/B${row}`,
    `=SUMIFS('All Estimates'!$${allDeltaL}$2:$${allDeltaL}$${allRows},'All Estimates'!$${allCity}$2:$${allCity}$${allRows},$A${row},'All Estimates'!$${allDomain}$2:$${allDomain}$${allRows},"complete_crime_domain",'All Estimates'!$${allPeriod}$2:$${allPeriod}$${allRows},"pooled_2020_2022")`,
    `=SUMIFS('All Estimates'!$${allDeltaL}$2:$${allDeltaL}$${allRows},'All Estimates'!$${allCity}$2:$${allCity}$${allRows},$A${row},'All Estimates'!$${allDomain}$2:$${allDomain}$${allRows},"bike_covered_training",'All Estimates'!$${allPeriod}$2:$${allPeriod}$${allRows},"pooled_2020_2022")`,
  ]];
}
dashboard.getRange("A12:F12").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
dashboard.getRange("A:A").format.columnWidth = 24;
dashboard.getRange("B:F").format.columnWidth = 22;
dashboard.getRange("B13:C15").format.numberFormat = "0.000000";
dashboard.getRange("D13:D15").format.numberFormat = "0.00x";
dashboard.getRange("E13:F15").format.numberFormat = "0.000000";

dashboard.getRange("H11:L11").merge();
dashboard.getRange("H11").values = [["Inference and interpretation boundary"]];
dashboard.getRange("H11:L11").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("H12:L15").merge();
dashboard.getRange("H12").values = [[
  "All three primary pooled city estimates are positive with centered 95% "
  + "block-bootstrap intervals above zero. This supports the claim that "
  + "strictly lagged bicycle flow contains incremental predictive information "
  + "conditional on the matched baseline state. It does not establish a causal "
  + "effect, and the DELB reduction is not a guaranteed fitted-model MSE gain. "
  + "Annual sparse-state estimates, especially Vancouver, require robustness "
  + "and placebo checks in subsequent experiments.",
]];
dashboard.getRange("H12:L15").format = {
  fill: "#FFF2CC",
  font: { color: "#7F6000" },
  wrapText: true,
  verticalAlignment: "center",
};
dashboard.getRange("H12:L15").format.rowHeight = 25;
dashboard.freezePanes.freezeRows(1);

const previews = [
  ["Read Me", "A1:F15", 0.85],
  ["Dashboard", "A1:L15", 1.0],
];
for (const [sheetName] of imports) {
  const item = imported.get(sheetName);
  const endColumn = columnName(Math.min(item.headers.length, 12) - 1);
  const endRow = Math.min(item.rowCount, 22);
  previews.push([sheetName, `A1:${endColumn}${endRow}`, 0.72]);
}
for (const [sheetName, range, scale] of previews) {
  const preview = await workbook.render({
    sheetName,
    range,
    scale,
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
  range: "Dashboard!A1:L15",
  include: "values,formulas",
  tableMaxRows: 18,
  tableMaxCols: 12,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(
  path.join(tableDir, "E04_workbook_verification.txt"),
  `${dashboardInspection.ndjson}\n${errors.ndjson}\n`,
  "utf8",
);
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(tableDir, "E04_city_information_QC.xlsx"));
