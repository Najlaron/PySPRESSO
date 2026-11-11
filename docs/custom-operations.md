<h1>
  Creating custom operations
  <img src="https://raw.githubusercontent.com/Najlaron/PySPRESSO/PySPRESSO-GUI/frontend/media/logo-light-color-png.png" width="48" alt="PySPRESSO logo">
</h1>

PySPRESSO can be extended with Python operations that appear in the
graphical interface and can be added to analysis workflows.

You can write an operation yourself or use an AI coding assistant.
The supplied example demonstrates registration, user parameters, data
access, plotting, and HTML reporting.

This tutorial applies to running PySPRESSO from source. Editing operations
in a packaged Windows application depends on how that build loads custom
code.

## 1. Find the operations folder ☕

In your local copy of the source code, open:

    pyspresso/pyspresso_app/operations/

This folder contains the existing operations and a file named `my_ops.py`
for your own additions. 

You can also add operations in another `.py` file in this folder.
PySPRESSO loads operation modules at startup, and correctly registered
functions become available in the interface.

Unregistered functions can be used as helpers by your operations but do
not appear as standalone workflow operations. This is intended behaviour.
To use a helper defined in another module, import it explicitly.


Open `my_ops.py` in a Python-capable text editor. Its opening comments
describe the dataset structure, operation registration, output folders,
and reporting conventions.

> [!TIP]
> Start with the guide and example in `my_ops.py` from your installed
> version of PySPRESSO. Existing operations provide further examples.

## 2. Enable the example operation ☕

An example operation is provided in `my_ops.py`. 

To enable it:

1. Select the commented example, including its imports, registration,
   and function.
2. Use your editor's **Uncomment Lines** command.
3. Preserve the indentation and comments inside the function.
4. Save the file.
5. Stop and restart the PySPRESSO backend.

> [!IMPORTANT]
> Save your changes and restart the PySPRESSO backend.
> Refreshing the browser alone does not reload Python operation files.

## 3. Inspect the example operation in user-interface ☕

Open a workflow containing an initialized dataset, then search for:

**Example: Sample Median Overview**

You should be able to find the operation, add it into workflow, change
its parameters, inspect help texts and run it as any other operation.


After execution, check that:

- A figure appears in the interface.
- The HTML report contains explanatory text, a table, and a figure.
- A PNG file is saved in the workflow's `figures` folder.
- The original dataset remains unchanged.


## 4. Create your own operation ☕

Use the example as a starting point. Add your operation to `my_ops.py`
or another Python file in the operations folder.

When copying the example:

1. Give the operation a unique registration `id` and Python function name.
2. Choose a readable label and describe what the operation does.
3. Define the parameters users should be able to change.
4. Make parameter names match the function arguments.
5. Replace the example calculation with your method.
6. Update the report text, output filenames, and returned summary.

Two registered operations cannot share the same ID.

See the guide in `my_ops.py` and existing operation files for implementation
details.

### Citations

If your operation implements a published method or uses software that
should be acknowledged, add the appropriate reference to the `citation`
field in its registration.

PySPRESSO collects citations from executed operations into a workflow
citation table saved alongside the HTML report. This helps users identify
which methods and software to cite when reporting their analysis—for
example, the correction method used in their workflow.

If you develop and share an operation implementing your own algorithm,
include a reference to its paper, preprint, or archived software release
with a DOI. This makes the reference available to other users when they
run your operation and helps them credit your work.

Leave the field empty if no citation is applicable. Users should review
the collected references when preparing a publication.


## 5. Keep datasets aligned ☕

PySPRESSO stores features in rows and samples in columns.
The first data column, `cpdID`, contains feature identifiers rather than
measurements.

If your operation removes or reorders features or samples, update the
associated tables and references consistently:

- Match feature metadata using `cpdID`.
- Match sample metadata using `Sample File`.
- Keep batch information aligned with the sample columns.
- Update affected QC, blank, standard, and dilution-series references.

> [!IMPORTANT]
> The feature-metadata matching helper described in `my_ops.py` does not
> update sample metadata, batch information, or special sample lists.
> Keep these aligned whenever your operation removes or reorders samples.

Check whether the data have already been log-transformed, centered, or
scaled, and update the corresponding state flags if your operation
changes these properties. See `my_ops.py` for the relevant fields.

For exclusions that apply only to one analysis, use local copies rather
than removing samples from the entire workflow.

## 6. Save and report results ☕

Use the current workflow's output folder for generated files.
Save reusable models in its `models` subfolder.

Adding a plot to the HTML report and displaying it in the interface are
separate steps. Follow the example:

- Use the reporting functions to add text, tables, and figures to the report.
- Return a saved image path under `figure_path`, or multiple image paths
  under `saved_paths`, to display plots in the interface.
- The dictionary keys `figure_path` and `saved_paths` must use these exact
  names; your local Python variables can have other names.

If another operation needs a fitted model later, save it together with
the feature order and other information required to reuse it. Do not
assume an in-memory object will survive reloading the workflow.

## 7. Test your operation ☕

Start with a small dataset whose expected results you can check.

Confirm that:

- The calculation gives the expected result.
- Parameters affect the operation as intended.
- Invalid input produces a useful error message.
- Sample and feature identifiers remain correctly aligned.
- Plots and report entries describe the actual results.
- Saved models or results can be reused after reloading the workflow,
  if applicable.

Additional Python packages must be installed in the environment running
PySPRESSO. Importing a package in your file does not install it.

<h2>
  Troubleshooting
  <img src="https://raw.githubusercontent.com/Najlaron/PySPRESSO/PySPRESSO-GUI/frontend/media/spilled_coffee.png"
       width="48"
       align="center"
       alt="Spilled coffee">
</h2>

### The operation does not appear

Save the Python file and restart the backend. Check the backend console
for syntax errors, incorrect registration, missing dependencies, 
or duplicate operation IDs.

### The operation fails when run

Read the error message and check that the required data and metadata are
available. Confirm that parameter names match the function arguments.

### The plot is saved but not visible in the interface

Check that the returned dictionary includes `figure_path` or `saved_paths`
and that those paths point to saved image files.

### The operation works until the workflow is reloaded

Check whether it relies on a temporary Python object or a custom state
attribute that is not saved. Persist reusable results using the conventions
described in `my_ops.py`.
