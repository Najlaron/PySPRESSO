<h1>
  Creating operations with AI assistance
  <img src="https://raw.githubusercontent.com/Najlaron/PySPRESSO/PySPRESSO-GUI/frontend/media/logo-light-color-png.png" width="48" alt="PySPRESSO logo">
</h1>

You can use an AI coding assistant or chatbot to help create a custom
PySPRESSO operation. Explain what the operation should do and provide
the relevant code so the assistant can follow PySPRESSO's existing
conventions.

Start with the [custom-operation tutorial](custom-operations.md) to learn
where to place the generated code and how to run it.

## 1. Prepare the files ☕

Attach these files from your installed version of PySPRESSO:

- **`operations/my_ops.py`** — the developer guide and example operation.
- **`core/operation_models.py`** — the registration and parameter definitions.
- **`core/workflow_models.py`** — the workflow state and its save/load handling.

These paths are relative to `pyspresso/pyspresso_app/`.

If possible, also attach the operation file most relevant to your request:

| What you want to create | Useful reference |
|---|---|
| Filtering samples or features | `operations/filter_ops.py` |
| Correction | `operations/correction_ops.py` |
| Transformation or scaling | `operations/transform_ops.py` |
| Statistical analysis | `operations/statistics_ops.py` |
| Visualization | `operations/visualization_ops.py` |
| Data import | `operations/initialization_ops.py` |
| Saving or loading checkpoints | `operations/io_ops.py` |

If the assistant needs to inspect the reporting functions, also provide
`core/html_reporter.py`. For registration behaviour, provide
`core/registry.py`.

An agent with access to your local repository can read these files
directly instead of receiving attachments.

> [!TIP]
> Use files from the same PySPRESSO version that you will run.
> Start with the essential files and one relevant example rather than
> uploading the entire application.

## 2. Describe the operation ☕

Explain:

- What the operation should calculate or change.
- Which data or metadata it should use.
- Which settings users should be able to adjust.
- How missing values, zeros, or negative values should be handled.
- Whether it should change the dataset or only produce results.
- Which plots, tables, or saved files you expect.

If implementing a published method, provide its paper, documentation,
or reference implementation. Specify which behaviour you want reproduced.

You do not need to know every implementation detail. Ask the assistant
to clarify decisions that could change the scientific result.

## 3. Copy and complete this prompt ☕

> [!TIP]
> Give the assistant the code files, not just this prompt. Ask it to reuse
> PySPRESSO's existing structures and established library functions.
> If you are unsure about a methodological choice, ask for an explanation
> rather than letting the assistant silently choose.

Replace the bracketed sections before sending it with the files.

```text
Create one custom operation for my installed version of PySPRESSO.

Read the attached my_ops.py guide and example first, then inspect the
provided framework definitions and relevant existing operations.
If you have repository access, read those files directly.

MY REQUEST

Purpose:
[Explain what the operation should do.]

Inputs:
[Describe the data and metadata it needs, including relevant column names.]

User-adjustable parameters:
[List settings and preferred defaults, or ask for suggestions.]

Expected outputs:
[Specify dataset changes, plots, report tables, or saved models/files.]

Missing and invalid measurements:
[Describe how to handle missing values, zeros, negative values, etc.
If unsure, ask the assistant to explain the choices.]

Method or reference:
[Provide a published method, documentation, or existing implementation,
if applicable. Otherwise describe the intended calculation.]

IMPLEMENTATION RULES

1. Follow the existing PySPRESSO structures.
   Use the registration, workflow state, output conventions, and reporting
   functions shown in the supplied code. Do not create a replacement
   framework, a separate application, or a new configuration system.

2. Do not invent APIs.
   Use only registration keywords, ParameterDef arguments, OperationTag
   members, state fields, and helper functions verified in the supplied
   files. Do not assume that a helper exists because its name sounds
   plausible. Ask for the relevant file if necessary.

3. Register the operation exactly as the existing examples demonstrate.
   Use a unique operation ID and function name. Parameter names must match
   function arguments. Use supported parameter types, sensible defaults,
   and concise labels and help text. Declare the actual inputs required
   and outputs produced.

4. Reuse existing implementations.
   First check PySPRESSO's existing operations and helpers.
   For standard calculations, data manipulation, plotting, or modelling,
   prefer suitable functions from libraries already used by the project,
   such as pandas, NumPy, SciPy, Matplotlib, and scikit-learn.

   Do not reimplement an established algorithm when an appropriate library
   implementation is available. Check that its behaviour and defaults
   match the requested method and that it supports the installed version.

   Write only the additional logic needed to connect the calculation to
   PySPRESSO. Avoid unnecessary wrappers or compatibility layers.
   If an additional dependency is needed, explain why.

5. Respect the dataset structure.
   Features are rows; samples are columns; cpdID is the identifier column.
   Exclude identifiers from numeric calculations.
   Match sample metadata using Sample File and feature metadata using cpdID.
   Do not assume tables are aligned solely because they have equal lengths.

6. Preserve consistency when filtering or reordering.
   Update all affected metadata, batch information, and special sample
   references. Keep dilution concentrations paired with their samples.
   Do not assume the feature-metadata matching helper updates other tables.
   For analysis-only exclusions, use local copies.

7. Respect the current measurement scale.
   Check relevant log-transformation, centering, and scaling flags.
   Do not silently transform data, replace missing values with zero, or
   change the scientific method. Update state flags when appropriate.
   Ask about scientifically meaningful ambiguities before implementing them.

8. Use the existing output and reporting conventions.
   Save outputs under state.main_folder and models in its models subfolder.
   Use filenames that avoid unintentionally overwriting previous results.
   Return figure_path or saved_paths for frontend plots.
   Use add_text, add_table, and add_figure for the HTML report.
   Do not use plt.show().

9. Handle persistence explicitly.
   Do not assume a new state attribute or an in-memory model survives
   saving and reloading a workflow. Follow an existing persistence pattern.
   If new persistent fields are genuinely required, explain the necessary
   core changes rather than silently introducing them.

10. Keep changes local.
    Prefer adding code to my_ops.py or one new operation module.
    If the requested functionality requires changes elsewhere, explain
    which files and why before proposing those changes.
    Identify any additional dependencies; do not assume they are installed.
    Do not upgrade packages, change dependency versions, or refactor
    unrelated code without asking first.

11. Preserve methodological accuracy.
    For a named or published method, follow the supplied reference and
    identify any deviations. Do not substitute a different algorithm
    while keeping the original method's name.
    Include an appropriate citation when applicable; do not invent one.

12. Validate clearly.
    Raise informative errors for unsuitable inputs.
    Do not suppress failures and report success.
    Use a reproducible seed where randomness affects the result.

WHAT TO RETURN

- A brief explanation of what the operation does and any assumptions.
- Required imports and the complete registered operation.
- Exact instructions for where to put the code.
- Any additional dependencies or necessary changes outside the operation.
- A short test procedure with expected results.
- A clear statement of what you actually tested and what remains untested.

If essential information is missing, ask focused questions rather than
guessing or generating placeholder functionality.
```

## 4. Add and test the generated operation ☕

Edit the .py file in the operations folder or add a new one.
Save the file, and restart the PySPRESSO backend.

Test the operation on a small, non-confidential dataset first. Check the
numerical results, sample and feature alignment, plots, and HTML report.
If the operation saves reusable results or models, test reuse after
reloading the workflow.

> [!IMPORTANT]
> Code that runs successfully is not necessarily scientifically correct.
> Check the calculation against a simple example with known results.
> For unfamiliar methods, ask someone with relevant expertise to review it.

<h2>
  If something fails
  <img src="https://raw.githubusercontent.com/Najlaron/PySPRESSO/PySPRESSO-GUI/frontend/media/spilled_coffee.png"
       width="48"
       align="center"
       alt="Spilled coffee">
</h2>

Send the assistant:

- The exact error message and relevant backend traceback.
- The generated operation code.
- The parameter values used.
- A description of the input structure or a small synthetic example.

Ask it to identify the cause and provide a focused correction while
preserving PySPRESSO's existing conventions.

Do not upload confidential study data, patient identifiers, credentials,
or private paths containing sensitive information.
