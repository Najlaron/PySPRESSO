# PySPRESSO ☕

**PySPRESSO** — **Py**thon **S**tatistical **P**rocessing and **RE**porting for **S**cientific **S**tudies in **O**mics.

PySPRESSO is a local, browser-based application for building and running customizable workflows for omics feature-table analysis, with a focus on metabolomics and lipidomics.

Combine data import, filtering, correction, transformations, statistical analyses, and visualizations through a graphical interface. Extend the available operations with your own Python functions.

**[Download the Windows app](https://github.com/Najlaron/PySPRESSO/releases/latest)** · **[Build from source](#build-from-source-windows)** · **[Report an issue](https://github.com/Najlaron/PySPRESSO/issues)**

## Features

- **Workflow creation:** Assemble analysis workflows and configure operations through a graphical interface.
- **Data import:** Import feature tables from Compound Discoverer, SCIEX OS, or PySPRESSO outputs.
- **Filtering and transformations:** Process missing values, filter features, and transform or scale measurements.
- **QC-based correction:** Apply spline or LOESS/LOWESS correction to address technical variation.
- **Statistical analysis:** Explore data using methods including PCA and PLS-DA.
- **Reporting:** Generate HTML reports and workflow citation tables.
- **Custom operations:** Add Python operations using the operation registration system.

PySPRESSO processes feature tables exported by upstream software; it is not a raw mass-spectrometry data processing or peak-picking tool.

## Download and run on Windows

Download the packaged application from the **[latest release](https://github.com/Najlaron/PySPRESSO/releases/latest)**.

1. Under **Assets**, select the attached Windows application ZIP.
2. Extract the **entire ZIP** into a folder on your computer.
3. Open the extracted `PySPRESSO` folder and double-click **`PySPRESSO.exe`**.
4. Keep the console window open while using the application.
5. Your browser should open automatically at **http://127.0.0.1:5000**. If it does not, open this address manually.

Keep the executable and its accompanying files together. Do not run the executable from inside the ZIP.

The automatically generated **Source code** archives contain the project source, not the ready-to-run Windows application.

### Using and stopping the application

Use PySPRESSO through the browser to create a workflow, import your data, add operations, and inspect the results.

The application runs on your computer. The browser provides its interface.

To stop PySPRESSO, close the console window opened by `PySPRESSO.exe`. Closing only the browser tab does not stop the backend.

## Where your data is stored

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

## Build from source (Windows)

Use this route if you want to build the executable yourself or modify the application.

### Requirements

- [Python](https://www.python.org/downloads/)
- [Node.js and npm](https://nodejs.org/en/download)

Download or clone the **`PySPRESSO-GUI`** branch. If downloading the source as a ZIP, extract it before continuing.

### Build

From the project root, run:

```powershell
.\build-desktop.bat
```

The build script:

- Builds the frontend
- Packages the application with PyInstaller
- Creates the application folder at:

```text
pyspresso\dist\PySPRESSO\
```

**Rebuilding replaces the entire `dist\PySPRESSO` folder**, including the previous executable and bundled libraries. Keep any files you want to preserve outside that build-output folder.

### Run

Open:

```text
pyspresso\dist\PySPRESSO\
```

Then double-click **`PySPRESSO.exe`** and follow the launch instructions above.

## Extending PySPRESSO

PySPRESSO supports custom Python operations through its operation registration system. Operations can define their own parameters and citations and access workflow data for processing or analysis.

Custom operations may require additional Python dependencies. When introducing a new dependency, ensure that it is available in the runtime environment or included in a rebuilt application package.

## Feedback and contributions

PySPRESSO is under active development. Bug reports, suggestions, and contributions are welcome.

Please use [GitHub Issues](https://github.com/Najlaron/PySPRESSO/issues) to report a problem or suggest a feature. For errors, include:

- The PySPRESSO version and your operating system
- The steps needed to reproduce the problem
- The error message, if available

Please avoid uploading confidential study data. A small synthetic example is preferable when demonstrating a problem.

## License

See [LICENSE](LICENSE) for the project’s licensing terms.
