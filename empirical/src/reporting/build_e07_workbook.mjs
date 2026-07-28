import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

process.on("uncaughtException", (error) => {
  console.error(`E07_WORKBOOK_ERROR: ${error?.message ?? error}`);
  const relevant = String(error?.stack ?? "")
    .split("\n")
    .filter((line) => line.includes("build_e07_workbook.mjs"));
  if (relevant.length) console.error(relevant.join("\n"));
  process.exit(1);
});

process.on("unhandledRejection", (error) => {
  console.error(`E07_WORKBOOK_REJECTION: ${error?.message ?? error}`);
  process.exit(1);
});

const runDir = process.argv[2];
if (!runDir) {
  throw new Error("Usage: node build_e07_workbook.mjs RUN_DIRECTORY");
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
  header.format.rowHeight = 48;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);

  rawHeaders.forEach((headerName, index) => {
    const letter = columnName(index);
    const displayName = headerName.replaceAll("_", " ");
    let width = Math.min(32, Math.max(13, displayName.length + 2));
    if (headerName === "grid_id") {
      width = 29;
    } else if (headerName === "criterion") {
      width = 52;
    } else if (/detail|reason/.test(headerName)) {
      width = 45;
    } else if (headerName === "model_family") {
      width = 43;
    } else if (/city_label|model_label|theory_practice_category/.test(headerName)) {
      width = 30;
    } else if (/period_start|period_end/.test(headerName)) {
      width = 17;
    } else if (/expected|observed/.test(headerName)) {
      width = 38;
    }
    sheet.getRange(`${letter}1:${letter}${rowCount}`).format.columnWidth = width;
    if (/criterion|detail|reason|expected|observed/.test(headerName)) {
      sheet.getRange(`${letter}2:${letter}${rowCount}`).format.wrapText = true;
    }
    if (
      /rows|days|grids|observations|blocks|replicate|states|atoms|count/.test(
        headerName,
      )
    ) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "#,##0";
    } else if (/share|efficiency/.test(headerName)) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "0.00%";
    } else if (
      /bits|mse|gap|ratio|rho|slope|standard_error|ci_|p_value|r_squared/.test(
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
    sheet.getRange(`A2:${endColumn}${rowCount}`).format.rowHeight = 30;
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
  const categoryIndex = rawHeaders.indexOf("theory_practice_category");
  if (categoryIndex >= 0) {
    const letter = columnName(categoryIndex);
    const range = sheet.getRange(`${letter}2:${letter}${rowCount}`);
    const colors = {
      high_theory__high_empirical: "#E2F0D9",
      high_theory__low_empirical: "#FFF2CC",
      low_theory__high_empirical: "#D9EAF7",
      low_theory__low_empirical: "#E7E6E6",
    };
    for (const [text, fill] of Object.entries(colors)) {
      range.conditionalFormats.add("containsText", {
        text,
        format: { fill },
      });
    }
  }
}

const imports = [
  ["Acceptance", "acceptance_checklist.csv"],
  ["City DELB", "test_window_city_delb.csv"],
  ["City Model Gaps", "city_model_predictability_gaps.csv"],
  ["Local Theory", "pretest_local_theory.csv"],
  ["Local Comparison", "local_theory_practice_comparison.csv"],
  ["Local Spearman", "local_spearman_associations.csv"],
  ["FE Regressions", "city_fixed_effect_regressions.csv"],
  ["Classification", "theory_practice_classification_summary.csv"],
  ["E04 Thresholds", "e04_bicycle_thresholds_reused.csv"],
  ["Local Exclusions", "pretest_local_theory_exclusions.csv"],
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
  "E07 Theory–Practice Predictability-Gap Audit",
]];
readme.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
readme.getRange("A1:F1").format.rowHeight = 31;
readme.getRange("A3:B17").values = [
  ["Purpose", "Compare matched discrete entropy lower bounds with held-out E06 integer-prediction MSE, then evaluate local theory–practice correspondence."],
  ["Run directory", runDir],
  ["City theory period", "2022-07-01 through 2022-12-31, exactly matching the E06 test observations."],
  ["Local theory period", "Training plus validation observations ending 2022-06-30; no test outcomes enter local theory."],
  ["Local empirical period", "Held-out E06 predictions from 2022-07-01 through 2022-12-31."],
  ["Predictability gap", "Empirical integer-prediction MSE minus the matched exact DELB."],
  ["Information-use efficiency", "Matched DELB divided by empirical MSE; a proximity-to-bound measure in [0,1]."],
  ["Gap narrowing", "Delta MSE minus delta DELB; positive values mean empirical error falls by more than the theoretical floor."],
  ["Realization ratio", "Delta MSE divided by delta DELB; an unbounded diagnostic, not a literal efficiency percentage."],
  ["City uncertainty", "1,000 shared nonoverlapping 7-day block resamples jointly recompute both DELB and paired model MSE."],
  ["Local inference", "Spearman correlation with 2,000 paired-grid bootstrap resamples; BH adjustment across 12 city–model tests."],
  ["Pooled model", "OLS of local delta MSE on pretest local delta DELB with city fixed effects and HC3 covariance."],
  ["Classification", "City median delta DELB and city–model median delta MSE; ties are assigned to the low group."],
  ["Article figures", "All manuscript and supplementary figures are generated by Python/matplotlib. This workbook is a QC artifact only."],
  ["Causal boundary", "All quantities measure predictive information and model use; none identifies a causal bicycle-flow effect on crime."],
];
readme.getRange("A3:A17").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  verticalAlignment: "top",
};
readme.getRange("A3:B17").format.wrapText = true;
readme.getRange("A3:B17").format.verticalAlignment = "top";
readme.getRange("A:A").format.columnWidth = 27;
readme.getRange("B:B").format.columnWidth = 100;
readme.getRange("A3:B17").format.rowHeight = 42;
readme.freezePanes.freezeRows(1);

dashboard.showGridLines = false;
dashboard.getRange("A1:M1").merge();
dashboard.getRange("A1").values = [[
  "E07 Theory–Practice Quality-Control Dashboard",
]];
dashboard.getRange("A1:M1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
dashboard.getRange("A1:M1").format.rowHeight = 31;

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
  ["Test observations", null],
  ["City–model comparisons", null],
  ["Eligible local grids", null],
  ["Local joined rows", null],
];
const acceptanceRows = imported.get("Acceptance").rowCount;
const acceptanceStatusColumn = dataColumn("Acceptance", "status");
dashboard.getRange("B5:B10").formulas = [
  [`=COUNTIF('Acceptance'!$${acceptanceStatusColumn}$2:$${acceptanceStatusColumn}$${acceptanceRows},"PASS")`],
  [`=COUNTA('Acceptance'!$A$2:$A$${acceptanceRows})`],
  [`=${sourceCell("Acceptance", "observed", 5)}`],
  [`=${sourceCell("Acceptance", "observed", 3)}`],
  [`=${sourceCell("Acceptance", "observed", 6)}`],
  [`=${sourceCell("Acceptance", "observed", 7)}`],
];
dashboard.getRange("A4:B4").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
};
dashboard.getRange("B5:B10").format.numberFormat = "#,##0";
dashboard.getRange("A:A").format.columnWidth = 31;
dashboard.getRange("B:B").format.columnWidth = 18;

dashboard.getRange("D3:M3").merge();
dashboard.getRange("D3").values = [["Matched test-window theory–practice comparison"]];
dashboard.getRange("D3:M3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("D4:M16").values = [
  ["City", "Model", "L0", "LB", "ΔL", "ΔMSE", "Gap 0", "Gap B", "Gap narrowing", "95% interval"],
  ...Array.from({ length: 12 }, () => Array(10).fill(null)),
];
for (let index = 0; index < 12; index += 1) {
  const dashboardRow = 5 + index;
  const sourceRow = 2 + index;
  dashboard.getRange(`D${dashboardRow}:M${dashboardRow}`).formulas = [[
    `=${sourceCell("City Model Gaps", "city_label", sourceRow)}`,
    `=${sourceCell("City Model Gaps", "model_label", sourceRow)}`,
    `=${sourceCell("City Model Gaps", "l0_exact_mse", sourceRow)}`,
    `=${sourceCell("City Model Gaps", "lb_exact_mse", sourceRow)}`,
    `=${sourceCell("City Model Gaps", "delta_l_exact_mse", sourceRow)}`,
    `=${sourceCell("City Model Gaps", "delta_mse_integer", sourceRow)}`,
    `=${sourceCell("City Model Gaps", "gap_baseline", sourceRow)}`,
    `=${sourceCell("City Model Gaps", "gap_bicycle", sourceRow)}`,
    `=${sourceCell("City Model Gaps", "gap_narrowing", sourceRow)}`,
    `=TEXT(${sourceCell("City Model Gaps", "gap_narrowing_ci_low", sourceRow)},"0.0000")&" to "&TEXT(${sourceCell("City Model Gaps", "gap_narrowing_ci_high", sourceRow)},"0.0000")`,
  ]];
}
dashboard.getRange("D4:M4").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
dashboard.getRange("D:D").format.columnWidth = 20;
dashboard.getRange("E:E").format.columnWidth = 29;
dashboard.getRange("F:L").format.columnWidth = 14;
dashboard.getRange("M:M").format.columnWidth = 21;
dashboard.getRange("F5:L16").format.numberFormat = "0.000000";
dashboard.getRange("L5:L16").conditionalFormats.add("cellIs", {
  operator: "greaterThan",
  formula: 0,
  format: { fill: "#E2F0D9", font: { color: "#006100" } },
});
dashboard.getRange("L5:L16").conditionalFormats.add("cellIs", {
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
dashboard.getRange("A14:B20").merge();
dashboard.getRange("A14").values = [[
  "The DELB reduction is a property of the matched information set, whereas "
  + "the empirical MSE change depends on the fitted model and held-out sample. "
  + "A lower bicycle-aware floor therefore does not imply that every model "
  + "must improve. Local associations are descriptive and do not identify a "
  + "causal effect of bicycle activity on crime.",
]];
dashboard.getRange("A14:B20").format = {
  fill: "#FFF2CC",
  font: { color: "#7F6000" },
  wrapText: true,
  verticalAlignment: "center",
};
dashboard.getRange("A14:B20").format.rowHeight = 26;
dashboard.freezePanes.freezeRows(1);

const previews = [
  ["Read Me", "A1:F17", 0.78],
  ["Dashboard", "A1:M20", 0.82],
];
for (const [sheetName] of imports) {
  const item = imported.get(sheetName);
  const endColumn = columnName(Math.min(item.headers.length, 12) - 1);
  const endRow = Math.min(item.rowCount, 20);
  previews.push([sheetName, `A1:${endColumn}${endRow}`, 0.65]);
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
  range: "Dashboard!A1:M20",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 13,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(
  path.join(tableDir, "E07_workbook_verification.txt"),
  `${dashboardInspection.ndjson}\n${errors.ndjson}\n`,
  "utf8",
);
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(tableDir, "E07_theory_practice_QC.xlsx"));
