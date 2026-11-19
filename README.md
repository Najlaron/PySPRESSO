<h1>
  PySPRESSO
  <img src="https://raw.githubusercontent.com/Najlaron/PySPRESSO/PySPRESSO-GUI/frontend/media/logo-light-color-png.png" width="48" alt="PySPRESSO logo">
</h1>

**PySPRESSO** — **Py**thon **S**tatistical **P**rocessing and **RE**porting for **S**cientific **S**tudies in **O**mics.

PySPRESSO is a local, Python-based application for creating and running customizable workflows for omics feature-table analysis, with a focus on metabolomics and lipidomics.

Combine data import, filtering, correction, transformations, statistical analyses, and visualizations through a graphical interface. Extend the available operations with your own Python functions.

**[Download the Windows app](https://github.com/Najlaron/PySPRESSO/releases/latest)** · **[Try the demo](docs/example-workflow.md)** · **[Run from source](#run-from-source)** · **[Build the executable](#build-executable)** · **[Report an issue](https://github.com/Najlaron/PySPRESSO/issues)**

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

## Download and run on Windows ☕

The packaged Windows application is available from the **[latest release](https://github.com/Najlaron/PySPRESSO/releases/latest)**.

Use this option to run PySPRESSO without installing Python or Node.js.
If you want to add or edit Python operations, follow the
[run-from-source instructions](#run-from-source).

### Start the application

1. Under **Assets**, download the attached Windows application ZIP.
2. Extract the **entire ZIP** into a folder on your computer.
3. Open the extracted `PySPRESSO` folder and double-click **`PySPRESSO.exe`**.
4. Keep the console window open while using the application.
5. Your browser should open automatically at **http://127.0.0.1:5000**. If it does not, open this address manually.

Keep the executable and its accompanying files together. Do not run the executable from inside the ZIP.

The packaged application includes its runtime dependencies; you do not need to install Python, Node.js, or Docker separately to use it.

> [!TIP]
> Select the attached **Windows application ZIP**. The automatically generated
> **Source code** archives contain the project source, not the ready-to-run application.

### Use the application

Use the browser interface to create a workflow, import your data,
configure operations and inspect the results. The application runs
locally on your computer; the browser provides its interface.

For a guided first analysis, follow the
[coffee-data tutorial](docs/example-workflow.md).

### Stop and restart the application

Close the console window opened by `PySPRESSO.exe`.

Closing only the browser tab does not stop the backend.

To start PySPRESSO again, double-click `PySPRESSO.exe` in the extracted
application folder.

## Tutorials and demo data ☕

### Try the example workflow

Follow the [coffee-data tutorial](docs/example-workflow.md) to create
a project, import a prepared workflow, run the analysis and inspect
the plots and HTML report.

The tutorial uses a simplified coffee dataset and includes screenshots
of the main steps. The [demo folder](demo/) contains the input data,
sample metadata and workflow template.

### Create your own operations

PySPRESSO supports custom Python operations that appear in the graphical
interface, with configurable parameters, plots, HTML report entries
and citations.

- [Create custom operations](docs/custom-operations.md) — follow the
  example in `my_ops.py` and learn how to add your own operations.
- [Create operations with AI assistance](docs/AI-assisted-operations.md) —
  use the recommended reference files and a ready-to-copy prompt.

## Where your data is stored ☕

All execution options store working data locally on your computer.
The location depends on how you run PySPRESSO.

### Packaged Windows application

The Windows executable stores its working data in:

```text
%LOCALAPPDATA%\PySPRESSO
```

This includes uploaded files, generated outputs and the local database.

To open this location, paste the path into the Windows File Explorer address bar.

Working data is stored separately from the application folder, so replacing
the application package does not delete it. Back up this directory before
updating to a new version.

This location also applies to executables you build yourself.

### Running from source

When running directly from source, files are stored relative to the
project root—the folder containing `frontend` and `pyspresso`:

| Content | Location |
|---|---|
| Uploaded files | `uploads/` |
| Analysis outputs and reports | `pyspresso/outputs/` |
| Local database | `pyspresso/instance/workflows.db` |
| Python operations | `pyspresso/pyspresso_app/operations/` |

Back up your working data and custom operation files before replacing
or deleting your source folder.

<a id="run-from-source"></a>

## Run from source (Windows) ☕

Use this option to run PySPRESSO directly from its source code and add
or edit custom operations. Working data remains in the project folders
described above.

The frontend is built during setup and served by the Python backend,
so only one terminal is needed when running the application.

### Requirements

Install these tools before starting:

- [Python](https://www.python.org/downloads/), including pip
- [Node.js](https://nodejs.org/en/download), including npm

### 1. Download the source

Download and extract the source from the
**[PySPRESSO-GUI branch](https://github.com/Najlaron/PySPRESSO/tree/PySPRESSO-GUI)**
using **Code → Download ZIP**.

Alternatively, if Git is installed, run:

```powershell
git clone --branch PySPRESSO-GUI https://github.com/Najlaron/PySPRESSO.git
cd PySPRESSO
```

### 2. Install dependencies and prepare the interface

Open PowerShell in the project root—the folder containing `frontend`
and `pyspresso`. You can type `powershell` into the File Explorer
address bar and press **Enter**.

Create a dedicated Python environment and install the backend dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\pyspresso\requirements.txt
```

Then install and build the frontend:

```powershell
cd frontend
npm install
npm run build
cd ..
```

Wait for each command to finish successfully before continuing.
These setup steps are needed only for the initial installation.

### 3. Start PySPRESSO

From the project root, run:

```powershell
cd pyspresso
..\.venv\Scripts\python.exe -m pyspresso_app.desktop
```

Your browser should open automatically at **http://127.0.0.1:5000**.
If it does not, open this address manually.

Keep the terminal open while using PySPRESSO. To stop the application,
press **Ctrl+C** in the terminal. Closing only the browser tab does
not stop the application.

### Starting PySPRESSO again

For each later session:

1. Open your downloaded project folder—the folder containing
   `frontend`, `pyspresso` and `.venv`.
2. Open PowerShell in that folder by typing `powershell` into the
   File Explorer address bar and pressing **Enter**.
3. Run:

```powershell
cd pyspresso
..\.venv\Scripts\python.exe -m pyspresso_app.desktop
```

Keep the terminal open while working. You do not need to repeat
the installation commands or run a second terminal.

### Adding custom operations

Add your operation to `my_ops.py` or another Python file in:

```text
pyspresso/pyspresso_app/operations/
```

Follow [Creating custom operations](docs/custom-operations.md)
for registration, data access and reporting conventions.

After adding or editing an operation, save the Python file and stop
PySPRESSO with **Ctrl+C**. From the same terminal, restart it with:

```powershell
..\.venv\Scripts\python.exe -m pyspresso_app.desktop
```

Refreshing the browser alone does not reload the operation code.

If your operation requires an additional Python package, install it
in the same `.venv` environment used to run PySPRESSO.

<a id="build-executable"></a>

## Build the Windows executable ☕

Use this option if you want to create the standalone Windows application
yourself or distribute a build containing your changes.

The build script installs the project dependencies, builds the frontend,
and packages the application into a standalone Windows folder.

> [!NOTE]
> Building an executable is optional when running from source.
> A self-built executable stores working data in
> `%LOCALAPPDATA%\PySPRESSO`, just like the downloaded Windows application.

### Requirements

Install these tools before building:

- [Python](https://www.python.org/downloads/), including pip
- [Node.js](https://nodejs.org/en/download), including npm

### 1. Download the source

If you have already downloaded the source using the instructions above,
use that project folder.

Otherwise, download and extract the source from the
**[PySPRESSO-GUI branch](https://github.com/Najlaron/PySPRESSO/tree/PySPRESSO-GUI)**.

Alternatively, if Git is installed, run:

```powershell
git clone --branch PySPRESSO-GUI https://github.com/Najlaron/PySPRESSO.git
cd PySPRESSO
```

### 2. Prepare the build environment

Open PowerShell in the project root—the folder containing
`build-desktop.bat`.

If you have not already created the `.venv` environment, run:

```powershell
python -m venv .venv
```

Then run:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
$env:Path = "$PWD\.venv\Scripts;$env:Path"
```

This keeps the Python build dependencies in a dedicated environment.
The PATH change applies only to the current PowerShell session.

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

Rebuilding replaces the application build in this output folder.
Keep any files you want to preserve outside it.

### 4. Run or distribute the build

Open the output folder and double-click **`PySPRESSO.exe`**.

To distribute the application, copy or ZIP the **entire `PySPRESSO`
output folder**, including the executable and all accompanying files.

Custom operations and their dependencies must be included when building
the application. Rebuild the executable after changing the operation code.

<h2>
  Troubleshooting
  <img src="https://raw.githubusercontent.com/Najlaron/PySPRESSO/PySPRESSO-GUI/frontend/media/spilled_coffee.png"
       width="48"
       align="center"
       alt="Spilled coffee">
</h2>

### The browser does not open

Keep the application terminal or console open and visit:

```text
http://127.0.0.1:5000
```

If the page does not load, check the terminal or console for an error message.

### The packaged application does not start

Confirm that you extracted the entire ZIP and kept the executable
together with its accompanying files.

Include any console error message when reporting the problem.

### Running from source does not work

Confirm that the dependency installation and `npm run build` completed
successfully.

Start the application from the `pyspresso` folder using the Python
executable in the project's `.venv` environment:

```powershell
..\.venv\Scripts\python.exe -m pyspresso_app.desktop
```

If PowerShell cannot find that executable, check that `.venv` exists
in the project root and that your terminal is in the `pyspresso` folder.

Check the terminal output for the error message.

### Installation or an executable build fails

Check the console output for the first error. Confirm that Python,
pip, Node.js, and npm are available:

```powershell
python --version
python -m pip --version
node --version
npm --version
```

For an executable build, run `build-desktop.bat` from the project root,
in the same PowerShell session used to configure the build environment.

### A new operation does not appear

Save the operation file and restart the application. Check the terminal
for syntax errors or missing dependencies.

For further guidance, see the troubleshooting section in
[Creating custom operations](docs/custom-operations.md).

## Feedback and contributions ☕

PySPRESSO is under active development. Bug reports, feature suggestions,
and contributions are welcome.

Please use [GitHub Issues](https://github.com/Najlaron/PySPRESSO/issues)
to report a problem or suggest a feature.

For errors, include:

- The PySPRESSO version and your operating system
- Whether you used the downloaded Windows application, ran from source,
  or built the executable yourself
- The steps needed to reproduce the problem
- The error message, if available

Please avoid uploading confidential study data. A small synthetic example
is preferable when demonstrating a problem.

## License

See [LICENSE](LICENSE) for the project's licensing terms.
