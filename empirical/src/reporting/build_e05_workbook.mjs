import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

process.on("uncaughtException", (error) => {
  console.error(`E05_WORKBOOK_ERROR: ${error?.message ?? error}`);
  const relevant = String(error?.stack ?? "")
    .split("\n")
    .filter((line) => line.includes("build_e05_workbook.mjs"));
  if (relevant.length) console.error(relevant.join("\n"));
  process.exit(1);
});

process.on("unhandledRejection", (error) => {
  console.error(`E05_WORKBOOK_REJECTION: ${error?.message ?? error}`);
  process.exit(1);
});

const runDir = process.argv[2];
if (!runDir) {
  throw new Error("Usage: node build_e05_workbook.mjs RUN_DIRECTORY");
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
  header.format.rowHeight = 43;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);

  rawHeaders.forEach((headerName, index) => {
    const letter = columnName(index);
    const displayName = headerName.replaceAll("_", " ");
    let width = Math.min(30, Math.max(13, displayName.length + 2));
    if (headerName === "grid_id") {
      width = 29;
    } else if (
      /criterion|information_evidence|exclusion_reasons/.test(headerName)
    ) {
      width = 36;
    } else if (/city_label|support_driver|metric|cluster/.test(headerName)) {
      width = 26;
    }
    sheet.getRange(`${letter}1:${letter}${rowCount}`).format.columnWidth = width;
    if (
      /grids|days|events|states|atoms|observations|repetitions|replicate|count/.test(
        headerName,
      )
    ) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "#,##0";
    } else if (
      /share|relative_l_reduction|q_value|p_value/.test(headerName)
    ) {
      sheet
        .getRange(`${letter}2:${letter}${rowCount}`)
        .format.numberFormat = "0.0000";
    } else if (
      /bits|mse|error|bias|_se$|ci_|moran|spearman|longitude|latitude/.test(
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
    "eligible",
    "significant_positive_bound_reduction",
    "local_moran_significant",
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
  ["City Summary", "city_local_summary.csv"],
  ["Eligibility", "grid_eligibility.csv"],
  ["Local Estimates", "local_information_estimates.csv"],
  ["Global Moran", "global_moran_results.csv"],
  ["Local Moran", "local_moran_results.csv"],
  ["Support Diagnostic", "support_association_diagnostics.csv"],
  ["Reconciliation", "city_local_to_pooled_reconciliation.csv"],
  ["Sparsity", "local_state_sparsity.csv"],
  ["DELB Validation", "bootstrap_delb_interpolation_validation.csv"],
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
  "E05 Spatially Localized Bicycle Information and DELB Audit",
]];
readme.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
readme.getRange("A1:F1").format.rowHeight = 31;
readme.getRange("A3:B16").values = [
  ["Purpose", "Estimate grid-level conditional bicycle information, exact DELB reduction, multiple-testing screens, and spatial concentration."],
  ["Run directory", runDir],
  ["Spatial unit", "Deterministic projected 1 km grid."],
  ["Primary domain", "Training bicycle-covered grids that pass prespecified support-only eligibility rules."],
  ["Eligibility", "1,095 strict-lag days; at least 30 crime events; at least 55 positive bicycle days; no zero-flow run longer than 365 days; at least two target and bicycle states."],
  ["Baseline state", "Lagged crime bin, day of week, season, and holiday; grid identity is constant and therefore omitted locally."],
  ["Bicycle state", "Frozen E04 city-specific lag-1 total-flow bins; no E05 refitting."],
  ["Point estimator", "Miller–Madow conditional entropy; plugin values and support atoms are retained."],
  ["Uncertainty", "1,000 shared city-level 7-day calendar-block resamples per eligible grid."],
  ["Multiple testing", "One-sided raw-CMI normal screen with Benjamini–Hochberg adjustment, plus two-sided CMI and DELB lower-endpoint checks."],
  ["Spatial analysis", "Row-standardized queen contiguity with 999 spatial randomizations for global and local Moran statistics."],
  ["Finite-sample warning", "High screen-pass rates are not null-calibrated evidence by themselves. Local CMI is strongly associated with singleton-state support; E10 placebos are required."],
  ["Raw bootstrap", "All 339,000 bootstrap rows remain in Zstandard Parquet and are intentionally not duplicated in this workbook."],
  ["Article figures", "All manuscript and supplementary figures are generated by Python/matplotlib. This workbook is a quality-control artifact only."],
];
readme.getRange("A3:A16").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  verticalAlignment: "top",
};
readme.getRange("A3:B16").format.wrapText = true;
readme.getRange("A3:B16").format.verticalAlignment = "top";
readme.getRange("A:A").format.columnWidth = 24;
readme.getRange("B:B").format.columnWidth = 98;
readme.getRange("A3:B16").format.rowHeight = 42;
readme.freezePanes.freezeRows(1);

dashboard.showGridLines = false;
dashboard.getRange("A1:L1").merge();
dashboard.getRange("A1").values = [[
  "E05 Local Information-Value Quality-Control Dashboard",
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
dashboard.getRange("A4:B9").values = [
  ["Metric", "Value"],
  ["Checks passed", null],
  ["Total checks", null],
  ["Covered grids", null],
  ["Eligible grids", null],
  ["Bootstrap rows", null],
];
dashboard.getRange("B5:B9").formulas = [
  [`=COUNTIF('Acceptance'!$D$2:$D$${imported.get("Acceptance").rowCount},"PASS")`],
  [`=COUNTA('Acceptance'!$A$2:$A$${imported.get("Acceptance").rowCount})`],
  [`=${sourceCell("Acceptance", "observed", 2)}`],
  [`=${sourceCell("Acceptance", "observed", 3)}`],
  [`=${sourceCell("Acceptance", "observed", 5)}`],
];
dashboard.getRange("A4:B4").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
};
dashboard.getRange("B5:B9").format.numberFormat = "#,##0";
dashboard.getRange("A:A").format.columnWidth = 30;
dashboard.getRange("B:B").format.columnWidth = 17;

dashboard.getRange("D3:L3").merge();
dashboard.getRange("D3").values = [["Primary local summaries"]];
dashboard.getRange("D3:L3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("D4:L7").values = [
  ["City", "Covered", "Eligible", "Pass screen", "Pass share", "Median CMI", "Median ΔL", "Median relative ΔL", "Projection grids"],
  [null, null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null, null],
  [null, null, null, null, null, null, null, null, null],
];
for (let index = 0; index < 3; index += 1) {
  const dashboardRow = 5 + index;
  const sourceRow = 2 + index;
  dashboard.getRange(`D${dashboardRow}:L${dashboardRow}`).formulas = [[
    `=${sourceCell("City Summary", "city_label", sourceRow)}`,
    `=${sourceCell("City Summary", "covered_grids", sourceRow)}`,
    `=${sourceCell("City Summary", "eligible_grids", sourceRow)}`,
    `=${sourceCell("City Summary", "significant_positive_bound_reduction_grids", sourceRow)}`,
    `=${sourceCell("City Summary", "significant_share_of_eligible", sourceRow)}`,
    `=${sourceCell("City Summary", "median_raw_cmi_bits", sourceRow)}`,
    `=${sourceCell("City Summary", "median_exact_delb_reduction_mse", sourceRow)}`,
    `=${sourceCell("City Summary", "median_relative_bound_reduction", sourceRow)}`,
    `=${sourceCell("City Summary", "projection_applied_grids", sourceRow)}`,
  ]];
}
dashboard.getRange("D4:L4").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
dashboard.getRange("D:D").format.columnWidth = 21;
dashboard.getRange("E:G").format.columnWidth = 13;
dashboard.getRange("H:L").format.columnWidth = 18;
dashboard.getRange("E5:G7").format.numberFormat = "#,##0";
dashboard.getRange("H5:H7").format.numberFormat = "0.0%";
dashboard.getRange("I5:J7").format.numberFormat = "0.000000";
dashboard.getRange("K5:K7").format.numberFormat = "0.0%";
dashboard.getRange("L5:L7").format.numberFormat = "#,##0";

dashboard.getRange("A12:F12").merge();
dashboard.getRange("A12").values = [["Global Moran results: exact DELB reduction"]];
dashboard.getRange("A12:F12").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("A13:F16").values = [
  ["City", "Moran's I", "Expected I", "Permutation p", "Spatial units", "Islands"],
  ["DC", null, null, null, null, null],
  ["NY", null, null, null, null, null],
  ["VAN", null, null, null, null, null],
];
const moranRows = imported.get("Global Moran").rowCount;
const moranCity = dataColumn("Global Moran", "city");
const moranMetric = dataColumn("Global Moran", "metric");
for (let row = 14; row <= 16; row += 1) {
  const criteria = [
    `'Global Moran'!$${moranCity}$2:$${moranCity}$${moranRows},$A${row}`,
    `'Global Moran'!$${moranMetric}$2:$${moranMetric}$${moranRows},"delta_l_exact_mse"`,
  ].join(",");
  dashboard.getRange(`B${row}:F${row}`).formulas = [[
    `=SUMIFS('Global Moran'!$${dataColumn("Global Moran", "moran_i")}$2:$${dataColumn("Global Moran", "moran_i")}$${moranRows},${criteria})`,
    `=SUMIFS('Global Moran'!$${dataColumn("Global Moran", "analytical_expected_i")}$2:$${dataColumn("Global Moran", "analytical_expected_i")}$${moranRows},${criteria})`,
    `=SUMIFS('Global Moran'!$${dataColumn("Global Moran", "permutation_p_value_two_sided")}$2:$${dataColumn("Global Moran", "permutation_p_value_two_sided")}$${moranRows},${criteria})`,
    `=SUMIFS('Global Moran'!$${dataColumn("Global Moran", "spatial_units")}$2:$${dataColumn("Global Moran", "spatial_units")}$${moranRows},${criteria})`,
    `=SUMIFS('Global Moran'!$${dataColumn("Global Moran", "island_units")}$2:$${dataColumn("Global Moran", "island_units")}$${moranRows},${criteria})`,
  ]];
}
dashboard.getRange("A13:F13").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
};
dashboard.getRange("A:A").format.columnWidth = 23;
dashboard.getRange("B:D").format.columnWidth = 17;
dashboard.getRange("E:F").format.columnWidth = 15;
dashboard.getRange("B14:D16").format.numberFormat = "0.000000";
dashboard.getRange("E14:F16").format.numberFormat = "#,##0";

dashboard.getRange("H12:L12").merge();
dashboard.getRange("H12").values = [["Finite-sample warning"]];
dashboard.getRange("H12:L12").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D", size: 12 },
};
dashboard.getRange("H13:L16").merge();
dashboard.getRange("H13").values = [[
  "The block bootstrap estimates sampling variability around the observed joint "
  + "distribution; it is not a null-independence randomization. Local CMI is "
  + "strongly associated with singleton-state support (Spearman ρ = 0.85, "
  + "0.78, and 0.62 for DC, NYC, and Vancouver). Therefore, screen-pass rates "
  + "must remain provisional until E09 estimator robustness and E10 temporal/"
  + "spatial placebo tests are completed.",
]];
dashboard.getRange("H13:L16").format = {
  fill: "#FFF2CC",
  font: { color: "#7F6000" },
  wrapText: true,
  verticalAlignment: "center",
};
dashboard.getRange("H13:L16").format.rowHeight = 25;
dashboard.freezePanes.freezeRows(1);

const previews = [
  ["Read Me", "A1:F16", 0.80],
  ["Dashboard", "A1:L16", 0.95],
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
  range: "Dashboard!A1:L16",
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
  path.join(tableDir, "E05_workbook_verification.txt"),
  `${dashboardInspection.ndjson}\n${errors.ndjson}\n`,
  "utf8",
);
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(
  path.join(tableDir, "E05_local_spatial_information_QC.xlsx"),
);
