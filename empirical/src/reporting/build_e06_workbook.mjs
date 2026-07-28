import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

process.on("uncaughtException", (error) => {
  console.error(`E06_WORKBOOK_ERROR: ${error?.message ?? error}`);
  const relevant = String(error?.stack ?? "")
    .split("\n")
    .filter((line) => line.includes("build_e06_workbook.mjs"));
  if (relevant.length) console.error(relevant.join("\n"));
  process.exit(1);
});

process.on("unhandledRejection", (error) => {
  console.error(`E06_WORKBOOK_REJECTION: ${error?.message ?? error}`);
  process.exit(1);
});

const runDir = process.argv[2];
if (!runDir) {
  throw new Error("Usage: node build_e06_workbook.mjs RUN_DIRECTORY");
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
  const endColumn = columnName(rows[0].length - 1);
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
  header.format.rowHeight = 44;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);

  rawHeaders.forEach((headerName, index) => {
    const letter = columnName(index);
    const displayName = headerName.replaceAll("_", " ");
    let width = Math.min(31, Math.max(13, displayName.length + 2));
    if (headerName === "grid_id") {
      width = 29;
    } else if (headerName === "criterion") {
      width = 55;
    } else if (/parameters_json|features_json/.test(headerName)) {
      width = 50;
    } else if (headerName === "model_family") {
      width = 43;
    } else if (/city_label|model_label|information_set/.test(headerName)) {
      width = 28;
    } else if (/test_start|test_end|validation_start|validation_end/.test(headerName)) {
      width = 17;
    } else if (/expected|observed/.test(headerName)) {
      width = 25;
    }
    sheet.getRange(`${letter}1:${letter}${rowCount}`).format.columnWidth = width;
    if (/criterion|parameters_json|features_json/.test(headerName)) {
      sheet.getRange(`${letter}2:${letter}${rowCount}`).format.wrapText = true;
      sheet.getRange(`${letter}2:${letter}${rowCount}`).format.rowHeight = 34;
    }
    if (
      /rows|days|grids|observations|total|replicate|iterations|maximum|check/.test(
        headerName,
      )
    ) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "#,##0";
    } else if (
      /relative|ratio|accuracy|share/.test(headerName)
    ) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "0.00%";
    } else if (
      /mse|rmse|mae|deviance|error|ci_|p_value|q_value|mean/.test(headerName)
    ) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "0.000000";
    }
  });

  const statusIndex = rawHeaders.indexOf("status");
  if (statusIndex >= 0) {
    const letter = columnName(statusIndex);
    const range = sheet.getRange(`${letter}2:${letter}${rowCount}`);
    range.conditionalFormats.add("containsText", {
      text: "PASS",
      format: {
        fill: "#E2F0D9",
        font: { bold: true, color: "#006100" },
      },
    });
    range.conditionalFormats.add("containsText", {
      text: "FAIL",
      format: {
        fill: "#FCE4D6",
        font: { bold: true, color: "#9C0006" },
      },
    });
  }
  for (const booleanHeader of [
    "matched_to_delb",
    "selected",
    "optimizer_converged",
    "positive_improvement_screen",
    "test_used_for_selection",
  ]) {
    const index = rawHeaders.indexOf(booleanHeader);
    if (index >= 0) {
      const letter = columnName(index);
      const range = sheet.getRange(`${letter}2:${letter}${rowCount}`);
      range.conditionalFormats.add("containsText", {
        text: "TRUE",
        format: { fill: "#E2F0D9", font: { color: "#006100" } },
      });
      range.conditionalFormats.add("containsText", {
        text: "FALSE",
        format: { fill: "#F2F2F2", font: { color: "#595959" } },
      });
    }
  }
}

const imports = [
  ["Acceptance", "acceptance_checklist.csv"],
  ["Test Metrics", "test_metrics.csv"],
  ["Paired Contrasts", "paired_mse_contrasts.csv"],
  ["Validation Tuning", "validation_tuning_results.csv"],
  ["Local Test Metrics", "local_test_metrics.csv"],
  ["Input Diagnostics", "city_input_diagnostics.csv"],
  ["Selected Models", "selected_model_metadata.csv"],
  ["E04 Thresholds", "e04_bicycle_thresholds_reused.csv"],
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
  "E06 Matched Out-of-Sample Predictive Models Audit",
]];
readme.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
readme.getRange("A1:F1").format.rowHeight = 31;
readme.getRange("A3:B16").values = [
  ["Purpose", "Test whether the frozen bicycle information state is exploited by matched count predictors in a fully held-out period."],
  ["Run directory", runDir],
  ["Primary domain", "All deterministic 1 km grids observed by the bicycle system during 2020–2021 training."],
  ["Training", "2020-01-01 through 2021-12-31."],
  ["Validation", "2022-01-01 through 2022-06-30; used for hyperparameter selection only."],
  ["Test", "2022-07-01 through 2022-12-31; never used for tuning."],
  ["Baseline information", "Grid, lagged crime bin, day of week, season, and holiday."],
  ["Bicycle-aware information", "Baseline state plus the frozen E04 city-specific lag-1 bicycle total-flow bin."],
  ["Matched models", "State-mean lookup, Poisson GLM, negative-binomial GLM, and Poisson histogram gradient boosting."],
  ["Prediction rule", "Continuous conditional means are retained; primary DELB comparison uses nonnegative half-up integer predictions."],
  ["Primary metric", "Held-out integer-prediction MSE; continuous MSE, RMSE, MAE, Poisson deviance, and calibration are auxiliary."],
  ["Uncertainty", "1,000 paired nonoverlapping 7-day test-calendar block resamples, preserving all grids within each city-day."],
  ["External benchmark", "Lag-7 seasonal naive; deliberately labeled nonmatched and excluded from DELB comparisons."],
  ["Article figures", "Every manuscript and supplementary figure is generated by Python/matplotlib. This workbook is a quality-control artifact only."],
];
readme.getRange("A3:A16").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  verticalAlignment: "top",
};
readme.getRange("A3:B16").format.wrapText = true;
readme.getRange("A3:B16").format.verticalAlignment = "top";
readme.getRange("A:A").format.columnWidth = 25;
readme.getRange("B:B").format.columnWidth = 98;
readme.getRange("A3:B16").format.rowHeight = 42;
readme.freezePanes.freezeRows(1);

dashboard.showGridLines = false;
dashboard.getRange("A1:L1").merge();
dashboard.getRange("A1").values = [[
  "E06 Matched Predictive-Model Quality-Control Dashboard",
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
dashboard.getRange("A4:B10").values = [
  ["Metric", "Value"],
  ["Checks passed", null],
  ["Total checks", null],
  ["Unique test observations", null],
  ["Saved prediction rows", null],
  ["Serialized matched models", null],
  ["Paired bootstrap rows", null],
];
const acceptanceRows = imported.get("Acceptance").rowCount;
dashboard.getRange("B5:B10").formulas = [
  [`=COUNTIF('Acceptance'!$E$2:$E$${acceptanceRows},"PASS")`],
  [`=COUNTA('Acceptance'!$A$2:$A$${acceptanceRows})`],
  [`=${sourceCell("Acceptance", "observed", 3)}`],
  [`=${sourceCell("Acceptance", "observed", 4)}`],
  [`=${sourceCell("Acceptance", "observed", 9)}`],
  [`=${sourceCell("Acceptance", "observed", 7)}`],
];
dashboard.getRange("A4:B4").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
};
dashboard.getRange("B5:B10").format.numberFormat = "#,##0";
dashboard.getRange("A:A").format.columnWidth = 31;
dashboard.getRange("B:B").format.columnWidth = 18;

dashboard.getRange("D3:L3").merge();
dashboard.getRange("D3").values = [["Held-out matched MSE contrasts"]];
dashboard.getRange("D3:L3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("D4:L16").values = [
  ["City", "Model", "Baseline MSE", "Bike MSE", "ΔMSE", "Relative", "95% CI low", "95% CI high", "Screen"],
  ...Array.from({ length: 12 }, () => Array(9).fill(null)),
];
for (let index = 0; index < 12; index += 1) {
  const dashboardRow = 5 + index;
  const sourceRow = 2 + index;
  dashboard.getRange(`D${dashboardRow}:L${dashboardRow}`).formulas = [[
    `=${sourceCell("Paired Contrasts", "city_label", sourceRow)}`,
    `=${sourceCell("Paired Contrasts", "model_label", sourceRow)}`,
    `=${sourceCell("Paired Contrasts", "mse_baseline_integer", sourceRow)}`,
    `=${sourceCell("Paired Contrasts", "mse_bicycle_integer", sourceRow)}`,
    `=${sourceCell("Paired Contrasts", "delta_mse_integer", sourceRow)}`,
    `=${sourceCell("Paired Contrasts", "relative_mse_improvement", sourceRow)}`,
    `=${sourceCell("Paired Contrasts", "normal_ci_low", sourceRow)}`,
    `=${sourceCell("Paired Contrasts", "normal_ci_high", sourceRow)}`,
    `=${sourceCell("Paired Contrasts", "positive_improvement_screen", sourceRow)}`,
  ]];
}
dashboard.getRange("D4:L4").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
dashboard.getRange("D:D").format.columnWidth = 20;
dashboard.getRange("E:E").format.columnWidth = 27;
dashboard.getRange("F:J").format.columnWidth = 15;
dashboard.getRange("K:L").format.columnWidth = 16;
dashboard.getRange("F5:H16").format.numberFormat = "0.000000";
dashboard.getRange("I5:I16").format.numberFormat = "0.00%";
dashboard.getRange("J5:K16").format.numberFormat = "0.000000";
dashboard.getRange("L5:L16").conditionalFormats.add("containsText", {
  text: "TRUE",
  format: { fill: "#E2F0D9", font: { bold: true, color: "#006100" } },
});
dashboard.getRange("L5:L16").conditionalFormats.add("containsText", {
  text: "FALSE",
  format: { fill: "#F2F2F2", font: { color: "#595959" } },
});

dashboard.getRange("A13:B13").merge();
dashboard.getRange("A13").values = [["Interpretation boundary"]];
dashboard.getRange("A13:B13").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("A14:B19").merge();
dashboard.getRange("A14").values = [[
  "The exact DELB reduction is a property of the available information set; "
  + "it does not guarantee that every fitted model lowers its held-out MSE. "
  + "E06 therefore retains positive, null, and negative model-specific changes. "
  + "A performance improvement is predictive rather than causal. E07 will link "
  + "these empirical changes to the theoretical predictability gap, while E09 "
  + "and E10 provide rolling/robustness and placebo checks.",
]];
dashboard.getRange("A14:B19").format = {
  fill: "#FFF2CC",
  font: { color: "#7F6000" },
  wrapText: true,
  verticalAlignment: "center",
};
dashboard.getRange("A14:B19").format.rowHeight = 26;
dashboard.freezePanes.freezeRows(1);

const previews = [
  ["Read Me", "A1:F16", 0.80],
  ["Dashboard", "A1:L19", 0.90],
];
for (const [sheetName] of imports) {
  const item = imported.get(sheetName);
  const endColumn = columnName(Math.min(item.headers.length, 12) - 1);
  const endRow = Math.min(item.rowCount, 20);
  previews.push([sheetName, `A1:${endColumn}${endRow}`, 0.70]);
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
  range: "Dashboard!A1:L19",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 12,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(
  path.join(tableDir, "E06_workbook_verification.txt"),
  `${dashboardInspection.ndjson}\n${errors.ndjson}\n`,
  "utf8",
);
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(tableDir, "E06_predictive_models_QC.xlsx"));
