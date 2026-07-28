import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

process.on("uncaughtException", (error) => {
  console.error(`E08_WORKBOOK_ERROR: ${error?.message ?? error}`);
  process.exit(1);
});

process.on("unhandledRejection", (error) => {
  console.error(`E08_WORKBOOK_REJECTION: ${error?.message ?? error}`);
  process.exit(1);
});

const runDir = process.argv[2];
if (!runDir) {
  throw new Error("Usage: node build_e08_workbook.mjs RUN_DIRECTORY");
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
  header.format.rowHeight = 50;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);

  rawHeaders.forEach((headerName, index) => {
    const letter = columnName(index);
    const displayName = headerName.replaceAll("_", " ");
    let width = Math.min(34, Math.max(13, displayName.length + 2));
    if (/criterion|detail|expected|observed/.test(headerName)) {
      width = headerName === "criterion" ? 55 : 42;
    } else if (/parameters_json|features_json|refit_splits_json/.test(headerName)) {
      width = 54;
    } else if (headerName === "domain") {
      width = 36;
    } else if (/^outcome(?:_a|_b)?$/.test(headerName)) {
      width = 23;
    } else if (/outcome_(?:a_|b_)?label|outcome_label/.test(headerName)) {
      width = 21;
    } else if (headerName === "model_family") {
      width = 42;
    } else if (headerName === "model_label") {
      width = 31;
    } else if (headerName === "target_field") {
      width = 34;
    } else if (/information_set(?:_label)?/.test(headerName)) {
      width = 23;
    } else if (/selection_metric|test_used_for_selection/.test(headerName)) {
      width = 25;
    } else if (/training_split|validation_split/.test(headerName)) {
      width = 18;
    } else if (headerName === "period") {
      width = 22;
    } else if (/city_label|period_label/.test(headerName)) {
      width = 24;
    } else if (/period_start|period_end|test_start|test_end|validation_start|validation_end/.test(headerName)) {
      width = 17;
    }
    sheet.getRange(`${letter}1:${letter}${rowCount}`).format.columnWidth = width;
    if (/criterion|detail|expected|observed|json/.test(headerName)) {
      sheet.getRange(`${letter}2:${letter}${rowCount}`).format.wrapText = true;
    }
    if (
      /rows|days|grids|observations|blocks|replicate|states|atoms|count|events|iterations/.test(
        headerName,
      )
    ) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "#,##0";
    } else if (/share|accuracy|efficiency|reduction/.test(headerName)) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "0.00%";
    } else if (
      /bits|mse|rmse|mae|gap|ratio|deviance|standard_error|ci_|p_value|z_statistic|mean|maximum|upper|seconds/.test(
        headerName,
      )
    ) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "0.000000";
    }
  });

  const statusIndex = rawHeaders.indexOf("status");
  if (statusIndex >= 0) {
    sheet.getRange(`A2:${endColumn}${rowCount}`).format.rowHeight = 35;
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

  for (const flag of [
    "positive_information_screen",
    "delta_l_heterogeneity_screen",
    "positive_prediction_screen",
    "selected",
    "matched_to_delb",
  ]) {
    const flagIndex = rawHeaders.indexOf(flag);
    if (flagIndex < 0) continue;
    const letter = columnName(flagIndex);
    const range = sheet.getRange(`${letter}2:${letter}${rowCount}`);
    range.conditionalFormats.add("containsText", {
      text: "TRUE",
      format: { fill: "#E2F0D9", font: { color: "#006100" } },
    });
  }
}

const imports = [
  ["Acceptance", "acceptance_checklist.csv"],
  ["Composition", "crime_type_composition.csv"],
  ["Information Bounds", "crime_type_information_bounds.csv"],
  ["Theory Pairs", "paired_theory_heterogeneity.csv"],
  ["Prediction Contrasts", "crime_type_prediction_contrasts.csv"],
  ["Prediction Pairs", "paired_prediction_heterogeneity.csv"],
  ["Test Metrics", "crime_type_test_metrics.csv"],
  ["Validation Tuning", "crime_type_validation_tuning.csv"],
  ["Model Metadata", "crime_type_model_metadata.csv"],
  ["Input Diagnostics", "crime_type_input_diagnostics.csv"],
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
  "E08 Crime-Type Heterogeneity Audit",
]];
readme.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
readme.getRange("A1:F1").format.rowHeight = 31;
readme.getRange("A3:B18").values = [
  ["Purpose", "Test whether the theoretical and empirical value of lagged bicycle information differs across property theft, vehicle theft, and burglary."],
  ["Run directory", runDir],
  ["Authoritative outcomes", "PROPERTY_THEFT, VEHICLE_THEFT, and BURGLARY are taken directly from crime_type_unified; no E08 remapping is performed."],
  ["Analysis domain", "The E04 frozen 1-km training bicycle-covered domain. Rows lacking a strict category-specific lag-1 target count are excluded."],
  ["Theory periods", "Pooled 2020–2022 estimates and a matched held-out test-window estimate for 2022-07-01 through 2022-12-31."],
  ["Information estimand", "Conditional mutual information is the entropy decrease after adding the frozen E04 bicycle-flow state to the category-specific baseline state."],
  ["Exact floor change", "Delta L is the decrease in the exact discrete entropy lower bound, measured in count-squared MSE units."],
  ["Theory uncertainty", "1,000 shared nonoverlapping 7-day block resamples per city and period; category contrasts use aligned replicates."],
  ["Prediction design", "Four matched model families are independently validation-tuned for each city, outcome, and information set, then evaluated once on the held-out test window."],
  ["Positive prediction screen", "Delta MSE must be positive, its normal interval must exclude zero, and its BH-adjusted one-sided p-value must not exceed 0.05."],
  ["Gap narrowing", "Delta MSE minus Delta L. Positive values indicate that the fitted model's MSE fell by more than the matched theoretical-floor reduction."],
  ["Supplementary benchmark", "The seasonal naive benchmark appears only in Test Metrics and is intentionally excluded from matched DELB contrasts."],
  ["Large artifacts", "Bootstrap replicates and 2,051,784 row-level predictions are stored as Parquet beside this workbook rather than duplicated into Excel."],
  ["Article figures", "Every manuscript and supplementary figure is generated by the E08 Python code. This workbook is a quality-control artifact."],
  ["Causal boundary", "The results describe predictive information conditional on observed states. They do not identify a causal effect of bicycle activity on crime."],
  ["Interpretive caution", "A lower bicycle-aware error floor does not require every fitted model to improve; sparse integer predictions can remain unchanged after adding information."],
];
readme.getRange("A3:A18").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  verticalAlignment: "top",
};
readme.getRange("A3:B18").format.wrapText = true;
readme.getRange("A3:B18").format.verticalAlignment = "top";
readme.getRange("A:A").format.columnWidth = 28;
readme.getRange("B:B").format.columnWidth = 105;
readme.getRange("A3:B18").format.rowHeight = 43;
readme.freezePanes.freezeRows(1);

dashboard.showGridLines = false;
dashboard.getRange("A1:J1").merge();
dashboard.getRange("A1").values = [[
  "E08 Crime-Type Heterogeneity Quality-Control Dashboard",
]];
dashboard.getRange("A1:J1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
dashboard.getRange("A1:J1").format.rowHeight = 31;

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
  ["Theory rows", null],
  ["Matched contrasts", null],
  ["Test prediction rows", null],
  ["Serialized models", null],
];
const acceptanceRows = imported.get("Acceptance").rowCount;
const acceptanceStatusColumn = dataColumn("Acceptance", "status");
dashboard.getRange("B5:B10").formulas = [
  [`=COUNTIF('Acceptance'!$${acceptanceStatusColumn}$2:$${acceptanceStatusColumn}$${acceptanceRows},"PASS")`],
  [`=COUNTA('Acceptance'!$A$2:$A$${acceptanceRows})`],
  [`=${sourceCell("Acceptance", "observed", 2)}`],
  [`=${sourceCell("Acceptance", "observed", 6)}`],
  [`=${sourceCell("Acceptance", "observed", 9)}`],
  [`=${sourceCell("Acceptance", "observed", 10)}`],
];
dashboard.getRange("A4:B4").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
};
dashboard.getRange("B5:B10").format.numberFormat = "#,##0";
dashboard.getRange("A:A").format.columnWidth = 30;
dashboard.getRange("B:B").format.columnWidth = 18;

dashboard.getRange("D3:J3").merge();
dashboard.getRange("D3").values = [["Pooled 2020–2022 theoretical information value"]];
dashboard.getRange("D3:J3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("D4:J13").values = [
  ["City", "Crime type", "CMI (bits)", "95% interval", "Delta L", "Relative floor reduction", "Screen"],
  ...Array.from({ length: 9 }, () => Array(7).fill(null)),
];
const informationRows = imported.get("Information Bounds").rows;
const pooledSourceRows = informationRows
  .slice(1)
  .map((row, index) => ({ row, sourceRow: index + 2 }))
  .filter(({ row }) => row[imported.get("Information Bounds").headers.indexOf("period")] === "pooled_2020_2022")
  .map(({ sourceRow }) => sourceRow);
for (let index = 0; index < pooledSourceRows.length; index += 1) {
  const dashboardRow = index + 5;
  const sourceRow = pooledSourceRows[index];
  dashboard.getRange(`D${dashboardRow}:J${dashboardRow}`).formulas = [[
    `=${sourceCell("Information Bounds", "city_label", sourceRow)}`,
    `=${sourceCell("Information Bounds", "outcome_label", sourceRow)}`,
    `=${sourceCell("Information Bounds", "delta_h_miller_madow_raw_bits", sourceRow)}`,
    `=TEXT(${sourceCell("Information Bounds", "delta_h_raw_normal_ci_low", sourceRow)},"0.0000")&" to "&TEXT(${sourceCell("Information Bounds", "delta_h_raw_normal_ci_high", sourceRow)},"0.0000")`,
    `=${sourceCell("Information Bounds", "delta_l_exact_mse", sourceRow)}`,
    `=${sourceCell("Information Bounds", "relative_l_reduction", sourceRow)}`,
    `=IF(${sourceCell("Information Bounds", "positive_information_screen", sourceRow)},"PASS","FAIL")`,
  ]];
}
dashboard.getRange("D4:J4").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
dashboard.getRange("D:D").format.columnWidth = 21;
dashboard.getRange("E:E").format.columnWidth = 19;
dashboard.getRange("F:F").format.columnWidth = 14;
dashboard.getRange("G:G").format.columnWidth = 22;
dashboard.getRange("H:H").format.columnWidth = 14;
dashboard.getRange("I:I").format.columnWidth = 19;
dashboard.getRange("J:J").format.columnWidth = 12;
dashboard.getRange("F5:F13").format.numberFormat = "0.000000";
dashboard.getRange("H5:H13").format.numberFormat = "0.000000";
dashboard.getRange("I5:I13").format.numberFormat = "0.00%";

dashboard.getRange("D16:J16").merge();
dashboard.getRange("D16").values = [["Held-out model comparisons passing the positive prediction screen"]];
dashboard.getRange("D16:J16").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("D17:J21").values = [
  ["City", "Crime type", "Model", "Delta MSE", "95% interval", "Delta L", "Gap narrowing"],
  ...Array.from({ length: 4 }, () => Array(7).fill(null)),
];
const contrastRows = imported.get("Prediction Contrasts").rows;
const screenColumn = imported.get("Prediction Contrasts").headers.indexOf("positive_prediction_screen");
const positiveSourceRows = contrastRows
  .slice(1)
  .map((row, index) => ({ row, sourceRow: index + 2 }))
  .filter(({ row }) => row[screenColumn] === true)
  .map(({ sourceRow }) => sourceRow);
for (let index = 0; index < positiveSourceRows.length; index += 1) {
  const dashboardRow = index + 18;
  const sourceRow = positiveSourceRows[index];
  dashboard.getRange(`D${dashboardRow}:J${dashboardRow}`).formulas = [[
    `=${sourceCell("Prediction Contrasts", "city_label", sourceRow)}`,
    `=${sourceCell("Prediction Contrasts", "outcome_label", sourceRow)}`,
    `=${sourceCell("Prediction Contrasts", "model_label", sourceRow)}`,
    `=${sourceCell("Prediction Contrasts", "delta_mse_integer", sourceRow)}`,
    `=TEXT(${sourceCell("Prediction Contrasts", "delta_mse_normal_ci_low", sourceRow)},"0.0000")&" to "&TEXT(${sourceCell("Prediction Contrasts", "delta_mse_normal_ci_high", sourceRow)},"0.0000")`,
    `=${sourceCell("Prediction Contrasts", "delta_l_exact_mse", sourceRow)}`,
    `=${sourceCell("Prediction Contrasts", "gap_narrowing", sourceRow)}`,
  ]];
}
dashboard.getRange("D17:J17").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
dashboard.getRange("F:F").format.columnWidth = 29;
dashboard.getRange("G:G").format.columnWidth = 14;
dashboard.getRange("H:H").format.columnWidth = 23;
dashboard.getRange("I:J").format.columnWidth = 15;
dashboard.getRange("G18:G21").format.numberFormat = "0.000000";
dashboard.getRange("I18:J21").format.numberFormat = "0.000000";
dashboard.getRange("J18:J21").conditionalFormats.add("cellIs", {
  operator: "greaterThan",
  formula: 0,
  format: { fill: "#E2F0D9", font: { color: "#006100" } },
});
dashboard.getRange("J18:J21").conditionalFormats.add("cellIs", {
  operator: "lessThan",
  formula: 0,
  format: { fill: "#FCE4D6", font: { color: "#9C0006" } },
});

dashboard.getRange("A13:B13").merge();
dashboard.getRange("A13").values = [["Interpretation boundary"]];
dashboard.getRange("A13:B13").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("A14:B21").merge();
dashboard.getRange("A14").values = [[
  "All nine pooled city–crime-type specifications show a positive conditional "
  + "information gain, but only four of 36 matched model comparisons pass the "
  + "positive held-out prediction screen. Theory measures how the information "
  + "set changes the irreducible error floor; empirical improvement additionally "
  + "depends on estimator use, integer rounding, and sparse outcomes. These "
  + "associations are predictive, not causal.",
]];
dashboard.getRange("A14:B21").format = {
  fill: "#FFF2CC",
  font: { color: "#7F6000" },
  wrapText: true,
  verticalAlignment: "center",
};
dashboard.getRange("A14:B21").format.rowHeight = 25;
dashboard.freezePanes.freezeRows(1);

const previews = [
  ["Read Me", "A1:F18", 0.76],
  ["Dashboard", "A1:J21", 0.80],
];
for (const [sheetName] of imports) {
  const item = imported.get(sheetName);
  const endColumn = columnName(Math.min(item.headers.length, 12) - 1);
  const endRow = Math.min(item.rowCount, 20);
  previews.push([sheetName, `A1:${endColumn}${endRow}`, 0.64]);
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
  range: "Dashboard!A1:J21",
  include: "values,formulas",
  tableMaxRows: 21,
  tableMaxCols: 10,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(
  path.join(tableDir, "E08_workbook_verification.txt"),
  `${dashboardInspection.ndjson}\n${errors.ndjson}\n`,
  "utf8",
);
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(
  path.join(tableDir, "E08_crime_type_heterogeneity_QC.xlsx"),
);
