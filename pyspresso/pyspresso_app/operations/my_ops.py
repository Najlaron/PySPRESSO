from pyspresso_app.core.registry import register_operation
from pyspresso_app.core.operation_models import OperationTag, ParameterDef
from pyspresso_app.core.workflow_models import WorkflowState
from pyspresso_app.core.html_reporter import (
    add_text,
    add_table,
    add_figure,
)


# ================================================================================
# CREATING CUSTOM PySPRESSO OPERATIONS
# ================================================================================
#
# Write your operations in this file or another .py file in the operations folder. 
# Restart PySPRESSO after changing the code.
# (It loads all python files in the operations folder at the start.)
#
# Existing operations provide examples of filtering, correction, statistics,
# plotting, and reporting. Import any additional libraries your operation needs;
# those libraries must also be available in the running Python environment or
# packaged application.
#
#
# 1. REGISTERING AN OPERATION
# --------------------------
# Use @register_operation(...) above your function.
#
#   id               Unique, stable identifier, e.g. "my_custom_filter".
#   label            Readable operation name.
#   description      Short explanation of what the operation does.
#   category_tags    Relevant OperationTag values; see existing operations.
#   parameter_schema Parameters accepted from the user.
#   requires         State fields needed before the operation can run.
#   produces         State fields/output categories produced by the operation.
#   citation         Optional reference for the implemented method.
#
# Each ParameterDef should provide a name, supported type, default, label,
# and concise help text. Use parameter types already used by other operations.
# Parameter names must match the arguments of your function.
#
# Function structure:
#
#   def my_operation(state: WorkflowState, threshold=0.5):
#       ...
#       return {"message": "Operation completed."}
#
# The first argument is the workflow state. Other arguments are user parameters.
# Registration does not perform calculations, save outputs, or update state:
# your function must do those things explicitly.
#
#
# 2. ACCESSING DATA
# -----------------
# state.data
#   pandas DataFrame: FEATURES IN ROWS, SAMPLES IN COLUMNS.
#   The first column, "cpdID", contains feature identifiers.
#   The remaining columns contain sample measurements.
#
#   feature_ids = state.data["cpdID"]
#   sample_names = state.data.columns[1:].tolist()
#   X = state.data.iloc[:, 1:].copy()
#
#   Feature-wise statistics across samples use axis=1.
#   Algorithms expecting samples in rows usually require X.T.
#   Never include cpdID in numeric calculations.
#
# state.metadata
#   Sample metadata. "Sample File" identifies the corresponding data column.
#   Other metadata columns depend on the imported dataset.
#
# state.variable_metadata
#   Feature metadata. Match its rows to state.data using "cpdID".
#
# state.batch_info
#   Sample-level batch/injection information. Its available columns depend on
#   the initializer. Check the columns needed by your operation.
#
# state.batch
#   Batch labels corresponding to the sample columns of state.data.
#
# Special sample references:
#   state.QC_samples
#   state.blank_samples
#   state.standard_samples
#   state.dilution_series_samples
#   state.dil_concentrations
#
# Sample lists contain sample identifiers. Dilution concentrations must remain
# associated with their corresponding dilution-series samples.
# Optional tables or references may be None.
#
#
# 3. MATCHING DATASETS AFTER FILTERING OR REORDERING
# ------------------------------------------------
# The intensity matrix and its associated tables must describe the same samples
# and features. Resetting a DataFrame index does NOT align identifiers.
#
# When removing features:
#   - Update state.data.
#   - Remove the corresponding rows from state.variable_metadata.
#   - Preserve cpdID values.
#
# Existing helper:
#
#   from pyspresso_app.operations.filter_ops import (
#       _filter_match_variable_metadata,
#   )
#
#   state.variable_metadata = _filter_match_variable_metadata(
#       state.data,
#       state.variable_metadata,
#   )
#
# This helper keeps feature-metadata rows whose cpdID remains in state.data.
# It DOES NOT reorder those rows or update sample-related information.
# If your calculation needs identical feature order, align explicitly by cpdID.
#
# When removing or reordering samples:
#   - Keep "cpdID" as the first data column.
#   - Match metadata["Sample File"] to the retained data-column names.
#   - Filter/reorder batch_info and state.batch consistently.
#   - Remove deleted samples from the QC, blank, standard, and dilution lists.
#   - Keep dilution concentrations paired with the retained dilution samples.
#
# Prefer identifier-based matching. Only use positional selection when you have
# verified that the original tables are already aligned.
#
# Example of aligning sample metadata to data-column order:
#
#   sample_names = state.data.columns[1:].tolist()
#   metadata_by_sample = state.metadata.set_index("Sample File")
#   # First validate unique IDs and that all sample_names exist in metadata.
#   metadata = metadata_by_sample.loc[sample_names].reset_index()
#
# See filter_ops.py for existing filtering patterns. Do not assume that a helper
# updating one table automatically updates every related state field.
#
# If samples are excluded only from a particular analysis, work on local copies
# rather than removing them from the whole workflow.
#
#
# 4. MODIFYING DATA
# -----------------
# Validate inputs and calculate results before assigning them back to state:
#
#   data = state.data.copy()
#   # Validate and modify data here.
#   state.data = data
#
# Check the current transformation state when relevant:
#   state.was_log_transformed
#   state.log_base
#   state.was_centered
#   state.was_scaled
#
# Update the corresponding flags if your operation changes the transformation.
# Do not assume that measurements are raw intensities.
#
# Decide explicitly how your operation handles missing values, zeros, negative
# values, and constant features. Do not silently treat missing values as zero.
#
# Previously fitted models/statistics may no longer apply after modifying data.
# Recompute them when needed rather than reusing results from a different matrix.
#
#
# 5. SAVING OUTPUTS AND MODELS
# ---------------------------
# Use state.main_folder as the current workflow's output directory.
# Do not hardcode a user's filesystem path or write into the source-code folder.
#
# Existing subfolders include:
#   figures          Saved plots.
#   statistics       Statistical tables and analysis outputs.
#   models           Saved fitted models.
#   dropped_features Lists of removed features.
#   data_checkpoints Saved dataset checkpoints.
#
# Create your output directory if necessary:
#
#   import os
#   model_folder = os.path.join(state.main_folder, "models")
#   os.makedirs(model_folder, exist_ok=True)
#
# Use operation-specific filenames. Use versioned/unique names when earlier
# outputs should remain available; existing operations show this pattern.
#
# For reusable models, save the fitted model together with feature identifiers
# in training order and any preprocessing/class information needed to reuse it.
# See statistics_PLSDA() for a joblib model-saving example.
#
#
# Adding state.my_result does not automatically make that attribute persistent.
# New persistent state fields require changes to WorkflowState and its save/load
# handling. Otherwise, save results to a known workflow-specific file that your
# later operation can locate.
#
#
# 6. PLOTS AND THE HTML REPORT
# ---------------------------
# Save plots inside the workflow folder and return their paths using recognized
# keys so that PySPRESSO can display them:
#
#   return {
#       "message": "Plot created.",
#       "figure_path": figure_path,
#   }
#
# For multiple plots:
#
#   return {
#       "message": "Plots created.",
#       "figure_paths": figure_paths,
#   }
#
# Use saved image-file paths, not Matplotlib objects, in the returned dictionary.
# Follow existing plotting operations for figure setup and close figures when
# finished. Do not use plt.show() in a backend operation.
#
# Add results to the HTML report with the imported reporting functions:
#
#   add_text(state, "Explanation of the result.", title="My operation")
#   add_table(state, result_table, title="Results", include_index=False)
#   add_figure(state, figure_path, title="My plot")
#
# Report meaningful settings, counts, results, and limitations.
#
# These are separate actions:
#   - Updating state makes results available to later operations.
#   - Reporting functions add content to the HTML report.
#   - Returning figure_path/figure_paths exposes saved plots to the interface.
#
#
# 7. VALIDATION AND TESTING
# ------------------------
# Raise a clear ValueError when required inputs are missing or unsuitable:
#
#   if state.data is None:
#       raise ValueError("No data loaded. Initialize a dataset first.")
#
# Return a small summary dictionary on success. Do not return a success message
# after catching an error that prevented the operation from completing.
#
# Test on a small dataset first:
#   - Check that sample and feature identifiers remain correctly aligned.
#   - Check the resulting data, saved outputs, plots, and HTML report.
#   - For reusable models/results, test access after reloading the workflow.
#
# If using an AI assistant, provide these instructions and a relevant existing
# operation. Ask it to follow the actual state structure and available helpers,
# rather than inventing new framework functions.
#
#
# 
# =============================================================================
# EXAMPLE OPERATION: SAMPLE MEDIAN OVERVIEW
# =============================================================================
# This example summarizes the current measurements without modifying state.data.
# It demonstrates registration, a user parameter, data access, saving a plot,
# HTML reporting, and returning the plot path for display in PySPRESSO.
#
# TO SEE AN EXAMPLE OPERATION APPEAR IN THE USER-INTERFACE: Select the example below, use your editor's "Uncomment Lines" command, save the file, and restart PySPRESSO.
#
#
# import os
# from uuid import uuid4
# import numpy as np
#
# import pandas as pd
# from matplotlib.figure import Figure
#
# @register_operation(
#     id="example_sample_median_overview",
#     label="Example: Sample Median Overview",
#     description=(
#         "Calculate the median measurement across features for each sample "
#         "and display their distribution. Uses the current data scale "
#         "without changing the dataset."
#     ),
#     citation="",
#     category_tags=[
#         OperationTag.STATISTICS,
#         OperationTag.VISUALIZATION,
#     ],
#     parameter_schema=[
#         ParameterDef(
#             name="bins", # Must match the function argument below.
#             type="int",
#             required=False,
#             default=20,
#             label="Histogram bins",
#             help="Number of histogram bins, from 1 to 100.",
#         ),
#     ],
#     requires=["data"],
#     produces=["figures"],
# )
# def example_sample_median_overview(state: WorkflowState, bins=20):
#     """Summarize sample medians and create a report and frontend plot."""

#     # 1. Validate inputs before calculating or writing outputs.
#     if state.data is None or state.data.empty:
#         raise ValueError("No data available. Initialize a dataset first.")

#     if state.data.columns[0] != "cpdID" or state.data.shape[1] < 2:
#         raise ValueError(
#             "Expected cpdID as the first column and at least one sample column."
#         )

#     bins = int(bins)
#     if not 1 <= bins <= 100:
#         raise ValueError("Histogram bins must be between 1 and 100.")

#     if not state.main_folder:
#         raise ValueError("The workflow output folder has not been initialized.")

#     # 2. Exclude feature identifiers from numeric calculations.
#     # Rows are features; columns are samples.
#     # Skip cpdID: only sample columns contain measurements.
#     measurements = state.data.iloc[:, 1:].apply(
#         pd.to_numeric,
#         errors="raise",
#     )

#     # This example requires finite measurements. Zeros and negative values
#     # remain valid; missing or infinite values require handling beforehand.
#     if not np.isfinite(measurements.to_numpy(dtype=float)).all():
#         raise ValueError(
#             "This example requires finite measurements. "
#             "Handle missing or infinite values before running it."
#         )

#     # axis=0 summarizes features within each sample column.
#     sample_medians = measurements.median(axis=0)

#     summary = pd.DataFrame({
#         "Sample File": sample_medians.index,
#         "Median measurement": sample_medians.to_numpy(),
#     })

#     # 3. Save the figure inside this workflow's output folder.
#     # A unique filename preserves figures from previous runs.
#     # Save inside the current workflow, not the application/source folder.
#     figures_folder = os.path.join(state.main_folder, "figures")
#     os.makedirs(figures_folder, exist_ok=True)

#     figure_path = os.path.join(
#         figures_folder,
#         f"example_sample_medians_{uuid4().hex}.png",
#     )

#     # Create a figure directly without opening an interactive plotting window.
#     fig = Figure(figsize=(8, 4))
#     ax = fig.subplots()
#     ax.hist(sample_medians.to_numpy(), bins=bins, edgecolor="white")
#     ax.set_xlabel("Sample median (current data scale)")
#     ax.set_ylabel("Number of samples")
#     ax.set_title("Distribution of sample medians")
#     fig.tight_layout()
#     fig.savefig(figure_path, dpi=150, bbox_inches="tight")
#     fig.clear()

#     # 4. Add an explanation, table, and saved figure to the HTML report.
#     add_text(
#         state,
#         (
#             f"Calculated sample medians across {measurements.shape[0]} "
#             f"features for {measurements.shape[1]} samples. "
#             f"The histogram uses {bins} bins. "
#             "Measurements were used on their current scale; "
#             "the dataset was not modified."
#         ),
#         title="Sample median overview",
#     )

#     add_table(
#         state,
#         summary,
#         title="Median measurement per sample",
#         include_index=False,
#     )
#     # Adds the saved plot to the HTML report.
#     add_figure(
#         state,
#         figure_path,
#         title="Distribution of sample medians",
#     )

#     # 5. Return the image path so the frontend can display the saved figure.
#     return {
#         "message": "Sample median overview created.",
#         # Use the exact key "figure_path" so the plot appears in the interface.
#         # For multiple plots, use "figure_paths": [path1, path2, ...].
#         "figure_path": figure_path, 
#         "sample_count": int(measurements.shape[1]),
#         "feature_count": int(measurements.shape[0]),
#     }