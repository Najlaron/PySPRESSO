<h1>
  PySPRESSO
  <img src="https://raw.githubusercontent.com/Najlaron/PySPRESSO/PySPRESSO-GUI/frontend/media/logo-light-color-png.png" width="48" alt="PySPRESSO logo">
</h1>

**PySPRESSO** — **Py**thon **S**tatistical **P**rocessing and **RE**porting for **S**cientific **S**tudies in **O**mics.

PySPRESSO is a local, browser-based application for creating and running customizable workflows for omics feature-table analysis, with a focus on metabolomics and lipidomics.

Combine data import, filtering, correction, transformations, statistical analyses, and visualizations through a graphical interface. Extend the available operations with your own Python functions.

**[Download the Windows app](https://github.com/Najlaron/PySPRESSO/releases/latest)** · **[Build from source](#build-from-source-windows)** · **[Report an issue](https://github.com/Najlaron/PySPRESSO/issues)**

## Features ☕

- **Workflow creation:** Assemble analysis workflows and configure operations through a graphical interface.
- **Data import:** Import feature tables from Compound Discoverer, SCIEX OS, or PySPRESSO outputs.
- **Filtering and transformations:** Filter features based on missingness and quality criteria, and transform or scale measurements.
- **QC-based correction:** Apply spline or LOESS/LOWESS correction to address technical variation.
- **Statistical analysis:** Explore data using methods including PCA and PLS-DA.
- **Reporting:** Generate HTML reports and workflow citation tables.
- **Custom operations:** Extend workflows with Python functions using the operation registration system.

> [!NOTE]
> PySPRESSO works with feature tables exported by upstream software.
> It does not perform peak picking directly from raw mass-spectrometry files.

## Tutorials and custom operations ☕

> [!TIP]
> PySPRESSO supports custom Python operations that appear in the graphical
> interface. These operations can provide configurable parameters, plots,
> HTML report entries, and citations.

- [Create custom operations](docs/custom-operations.md) — follow the
  example in `my_ops.py` and learn how to add your own operations.
- [Create operations with AI assistance](docs/AI-assisted-operations.md) —
  use the recommended reference files and a ready-to-copy prompt.

## Download and run on Windows ☕

The packaged Windows application is available from the **[latest release](https://github.com/Najlaron/PySPRESSO/releases/latest)**.

### Start the application 

1. Under **Assets**, download the attached Windows application ZIP.
2. Extract the **entire ZIP** into a folder on your computer.
3. Open the extracted `PySPRESSO` folder and double-click **`PySPRESSO.exe`**.
4. Keep the console window open while using the application.
5. Your browser should open automatically at **http://127.0.0.1:5000**. If it does not, open this address manually.

Keep the executable and its accompanying files together. Do not run the executable from inside the ZIP.

The packaged application includes its runtime dependencies; you do not need to install Python, Node.js, or Docker separately to use it.

> **Which download should I choose?**
>
> Select the attached **Windows application ZIP**. The automatically generated
> **Source code** archives contain the project source, not the ready-to-run application.

### Use the application 

Use the browser interface to create a workflow, import your data, add and configure operations, and inspect the results.

The application runs locally on your computer. The browser provides its interface.

### Stop the application 

Close the console window opened by `PySPRESSO.exe`.

Closing only the browser tab does not stop the backend.

## Where your data is stored ☕

The packaged Windows application stores its working data in:

```text
%LOCALAPPDATA%\PySPRESSO
```

This includes:

- Uploaded files
- Generated outputs
- The local database

To open this location, paste the path into the Windows File Explorer address bar.

Working data is stored separately from the application folder, so replacing the application package does not delete it. Back up this directory before updating to a new version.

## Build from source (Windows) ☕

Use this route if you want to build the executable yourself or modify the application.

The build script installs the project dependencies, builds the frontend, and packages the application into a standalone Windows folder.

### Requirements 

Install these tools before building:

- [Python](https://www.python.org/downloads/), including pip
- [Node.js](https://nodejs.org/en/download), including npm


### 1. Download the source 

Download and extract the source from the **`PySPRESSO-GUI`** branch.

Alternatively, if Git is installed, run:

```powershell
git clone --branch PySPRESSO-GUI https://github.com/Najlaron/PySPRESSO.git
cd PySPRESSO
```

### 2. Create a build environment 

Open PowerShell in the project root—the folder containing `build-desktop.bat`—and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
$env:Path = "$PWD\.venv\Scripts;$env:Path"
```

This keeps the Python build dependencies in a dedicated environment. The PATH change applies only to the current PowerShell session.

### 3. Build the application 

In the same PowerShell window, run:

```powershell
.\build-desktop.bat
```

The script:

1. Installs frontend dependencies with npm.
2. Builds the frontend.
3. Installs backend dependencies from `requirements.txt`, together with PyInstaller and its hooks.
4. Packages the application using `PySPRESSO.spec`.

Wait until the console displays **Build complete!**

The application is created in:

```text
pyspresso\dist\PySPRESSO\
```

Rebuilding replaces the application build in this output folder. Keep any files you want to preserve outside it.

### 4. Run or distribute the build ☕

Open the output folder and double-click **`PySPRESSO.exe`**.

To distribute the application, copy or ZIP the **entire `PySPRESSO` output folder**, including the executable and all accompanying files.

<h2>
  Troubleshooting
  <img src="https://raw.githubusercontent.com/Najlaron/PySPRESSO/PySPRESSO-GUI/frontend/media/spilled_coffee.png"
       width="48"
       align="center"
       alt="Spilled coffee">
</h2>

### The browser does not open

Keep the application console open and visit:

```text
http://127.0.0.1:5000
```

If the page does not load, check the console for an error message.


### The application does not start 


Confirm that you extracted the entire ZIP and kept the executable together with its accompanying files.

Include any console error message when reporting the problem.

### A source build fails

Check the console output for the first error. Confirm that Python, pip, Node.js, and npm are available:

```powershell
python --version
python -m pip --version
node --version
npm --version
```

Run the build from the project root, in the same PowerShell session used to configure the build environment.

## Feedback and contributions ☕

PySPRESSO is under active development. Bug reports, feature suggestions, and contributions are welcome.

Please use [GitHub Issues](https://github.com/Najlaron/PySPRESSO/issues) to report a problem or suggest a feature.

For errors, include: <img src="https://raw.githubusercontent.com/Najlaron/PySPRESSO/PySPRESSO-GUI/frontend/media/spilled_coffee.png"
       width="48"
       align="center"
       alt="Spilled coffee">

- The PySPRESSO version and your operating system
- Whether you used the packaged application or built it from source
- The steps needed to reproduce the problem
- The error message, if available

Please avoid uploading confidential study data. A small synthetic example is preferable when demonstrating a problem.

## License

See [LICENSE](LICENSE) for the project's licensing terms.
