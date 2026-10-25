import { useState } from "react"
import { useNavigate } from "react-router-dom"
import StepBadge from '../../molecules/HomePage/StepBadge'
import { formatNetworkError } from "../../../utils/helpers"
import { FaRegQuestionCircle } from "react-icons/fa"
import Tooltip from "../../molecules/WorkflowLayout/Tooltip"
import { API_BASE_URL } from "../../../config"
import { PiUploadSimpleBold } from "react-icons/pi"
import { MdDone } from "react-icons/md"

const DATA_FORMAT_STORAGE_KEY = "pyspresso:lastDataFormat"
const SUPPORTED_DATA_FORMATS = ["cd", "sciexos", "pyspresso"]

const TABLE_UPLOADS = {
    cd: [
        { key: "data", label: "Data" },
        { key: "batchInfo", label: "Batch info" },
    ],
    sciexos: [
        { key: "data", label: "SciexOS export" },
    ],
    pyspresso: [
        { key: "data", label: "Data" },
        { key: "metadata", label: "Sample metadata" },
        { key: "variableMetadata", label: "Variable metadata" },
        { key: "batchInfo", label: "Batch info" },
    ],
}

function WorkflowCreationForm({ filesState, filesDispatch, loadError, setLoadError }) {
    const navigate = useNavigate()

    // údaje pro vytvoření workflow
    const [workflowName, setWorkflowName] = useState("")
    const [folderName, setFolderName] = useState("")
    const [reportFileName, setReportFileName] = useState("")
    const [dataFormat, setDataFormat] = useState(() => {
        try {
            const savedFormat = localStorage.getItem(DATA_FORMAT_STORAGE_KEY)

            return SUPPORTED_DATA_FORMATS.includes(savedFormat)
                ? savedFormat
                : "cd"
        } catch {
            return "cd"
        }
    })

    const requiredUploads = TABLE_UPLOADS[dataFormat]

    const fileErrors = [
        ...requiredUploads.map(({ key }) => filesState?.[key]?.error),
        filesState?.importFile?.error,
    ].filter(Boolean)

    const hasFileErrors = fileErrors.length > 0
    const hasImportFile = Boolean(filesState?.importFile?.file)

    const [tableDragKey, setTableDragKey] = useState(null)

    function handleDataFormatChange(format) {
        setDataFormat(format)
        setLoadError(null)
        setTableDragKey(null)

        for (const key of ["data", "metadata", "variableMetadata", "batchInfo"]) {
            filesDispatch({
                type: "SET_ERROR",
                key,
                error: null,
            })
        }
    }

    // pro změnu designu
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [importDragActive, setImportDragActive] = useState(false)

    // submit formuláře
    async function onSubmit(e) {
        e.preventDefault()

        if (isSubmitting) return
        if (hasFileErrors) {
            setLoadError(fileErrors[0])
            return
        }

        // kontrola, jestli uživatel nahrál jak data tak batch info
        const missingFiles = requiredUploads.filter(
            ({ key }) => !filesState?.[key]?.file
        )

        if (missingFiles.length) {
            missingFiles.forEach(({ key, label }) => {
                filesDispatch({
                    type: "SET_ERROR",
                    key,
                    error: `${label} file is required.`,
                })
            })

            setLoadError(
                `Please upload: ${missingFiles.map(({ label }) => label).join(", ")}.`
            )
            return
        }

        const formData = new FormData()

        formData.append("workflowName", workflowName)
        formData.append("folderName", folderName)
        formData.append("reportFileName", reportFileName)
        formData.append("dataFormat", dataFormat)

        for (const { key } of requiredUploads) {
            formData.append(key, filesState[key].file)
        }

        if (filesState?.importFile?.file) {
            formData.append("importFile", filesState.importFile.file)
        }

        setIsSubmitting(true)
        try {
            const response = await fetch(API_BASE_URL + "/new_workflow", {
                method: "POST",
                body: formData,
            })

            const responseData = await response.json()

            if (!response.ok) {
                setLoadError(responseData?.message ?? "Failed to create new workflow.")
                return
            }
            // Remember the format after successful workflow creation.
            try {
                localStorage.setItem(DATA_FORMAT_STORAGE_KEY, dataFormat)
            } catch {
                // Continue if preference storage is unavailable.
            }

            setWorkflowName("")
            setFolderName("")
            setReportFileName("")
            filesDispatch({ type: 'CLEAR' })
            setLoadError(null)

            // přesměrování na layout
            navigate(`/workflow/${responseData.workflowId}`)
        } catch (err) {
            setLoadError(formatNetworkError(err))
        } finally {
            setIsSubmitting(false)
        }
    }

    // funkce pro získání importu, musí být JSON
    function handleFileInputChange(file) {
        if (!file) return

        if (file.name.toLowerCase().endsWith('.json')) {
            filesDispatch({ type: 'SET_FILE', key: 'importFile', file })
            filesDispatch({ type: 'SET_ERROR', key: 'importFile', error: null })
        } else {
            filesDispatch({ type: 'SET_ERROR', key: 'importFile', error: 'Please select a JSON file' })
        }
    }

    function handleTableFileChange(file, key) {
        if (!file) return

        filesDispatch({
            type: "SET_FILE",
            key,
            file,
        })
    }

    function processDroppedFile(file, key) {
        if (!file) return

        if (key === "importFile") {
            handleFileInputChange(file)
        } else {
            handleTableFileChange(file, key)
        }
    }

    return (
        <form onSubmit={onSubmit} className="flex flex-col gap-ds-xl items-center">

            {/* první část formuláře - název wf a název složky */}
            <div className="flex flex-col items-center gap-ds-lg">
                <div className="flex justify-center items-center gap-ds-lg">
                    <StepBadge
                        stepNumber={1}
                    />
                    <h2 className="text-3xl font-bold text-noir">Project configuration</h2>
                </div>
                <div className="flex flex-col gap-ds-sm">
                    <label htmlFor="workflowName" className="font-medium text-noir text-2xl flex flex-col justify-center">
                        <div className="flex items-center gap-ds-sm">
                            <Tooltip
                                text={"Name of the workflow (used for display)"}
                            >
                                <FaRegQuestionCircle size="1.5rem" color="341100" className="shrink-0" />
                            </Tooltip>
                            Workflow name *
                        </div>
                    </label>
                    <input
                        type="text"
                        id="workflowName"
                        value={workflowName}
                        onChange={(e) => setWorkflowName(e.target.value)}
                        className="border border-roast/50 rounded-[10px] px-3 py-2 h-14 w-78 focus:border-noir focus:outline-none focus:border-2"
                        required
                    />
                </div>
                <div className="flex flex-col gap-ds-sm">
                    <label htmlFor="folderName" className="font-medium text-noir text-2xl flex flex-col justify-center">
                        <div className="flex items-center gap-ds-sm">
                            <Tooltip
                                text={"Folder where visualizations and outputs will be saved"}
                            >
                                <FaRegQuestionCircle size="1.5rem" color="341100" className="shrink-0" />
                            </Tooltip>
                            Folder name *
                        </div>
                    </label>
                    <input
                        type="text"
                        id="folderName"
                        value={folderName}
                        onChange={(e) => setFolderName(e.target.value)}
                        className="border border-roast/50 rounded-[10px] px-3 py-2 h-14 w-78 focus:border-noir focus:outline-none focus:border-2"
                        required
                    />
                </div>
                {/* <div className="flex flex-col">
                    <label htmlFor="reportFileName" className="mb-[8px] font-medium text-noir text-2xl">
                        <div className="flex items-center gap-ds-sm">
                            <Tooltip
                                text={"Filename for the generated report"}
                            >
                                <FaRegQuestionCircle size="1.5rem" color="341100" className="shrink-0" />
                            </Tooltip>
                            Report file name *
                        </div>
                    </label>
                    <input
                        type="text"
                        id="reportFileName"
                        value={reportFileName}
                        onChange={(e) => setReportFileName(e.target.value)}
                        className="border border-roast/50 rounded-[10px] px-3 py-2 h-14 w-78 focus:border-noir focus:outline-none focus:border-2"
                        required
                    />
                </div> */}
            </div >

            {/* druhá část formuláře - formát vstupních dat */}
            <div>
                <div className="flex justify-center items-center gap-ds-lg mb-ds-lg">
                    <StepBadge
                        stepNumber={2}
                    />
                    <h2 className="text-3xl font-bold text-noir">Data format</h2>
                </div>

                <div className="flex flex-col">
                    <label htmlFor="format" className="mb-ds-sm font-medium text-noir text-2xl">
                        <div className="flex items-center gap-ds-sm">
                            <Tooltip
                                text={"Choose the format before uploading. Compound Discoverer requires two files; SciexOS requires one export; " +
                                        "PySPRESSO requires data, sample metadata, variable metadata and batch info."
                                    }
                            >
                                <FaRegQuestionCircle size="1.5rem" color="341100" className="shrink-0" />
                            </Tooltip>
                            Data format *
                        </div>
                    </label>
                    <select
                        id="format"
                        value={dataFormat}
                        onChange={(e) => handleDataFormatChange(e.target.value)}
                        className="border border-roast/75 rounded-lg p-ds-sm h-18 w-80 focus:border-noir focus:outline-none focus:border-2"
                        required
                    >
                        <option value="cd">Compound Discoverer</option>
                        <option value="sciexos">SciexOS</option>
                        <option value="pyspresso">PySPRESSO</option>
                    </select>
                </div>
            </div>


            {/* třetí část formuláře - vstupní data a batch info */}
            <div>
                <div className="flex justify-center items-center gap-ds-lg mb-ds-lg">
                    <StepBadge
                        stepNumber={3}
                    />
                    <h2 className="text-3xl font-bold text-noir">Data import</h2>
                </div>

                <div className="flex flex-wrap justify-center gap-ds-xl">
                    {requiredUploads.map(({ key, label }) => {
                        const file = filesState?.[key]?.file
                        const error = filesState?.[key]?.error
                        const inputId = `${key}FileInput`

                        return (
                            <div key={key} className="flex flex-col">
                                <label
                                    htmlFor={inputId}
                                    className="mb-[8px] font-medium text-noir text-2xl"
                                >
                                    Upload {label.toLowerCase()} *
                                </label>

                                <input
                                    id={inputId}
                                    type="file"
                                    accept=".csv,.tsv,.txt,.xlsx,.xls"
                                    onChange={(e) =>
                                        handleTableFileChange(e.target.files?.[0], key)
                                    }
                                    className="sr-only"
                                />

                                <label
                                    htmlFor={inputId}
                                    onDrop={(e) => {
                                        e.preventDefault()
                                        setTableDragKey(null)
                                        processDroppedFile(e.dataTransfer.files?.[0], key)
                                    }}
                                    onDragOver={(e) => e.preventDefault()}
                                    onDragEnter={() => setTableDragKey(key)}
                                    onDragLeave={() => setTableDragKey(null)}
                                    className={`
                                        w-78 border rounded-[10px] px-ds-md py-ds-xl
                                        flex flex-col justify-center items-center gap-ds-sm
                                        cursor-pointer hover:bg-crema/50 transition
                                        ${
                                            tableDragKey === key
                                                ? "ring-2 ring-espresso border-transparent"
                                                : "border-dashed border-roast/75"
                                        }
                                    `}
                                >
                                    {file ? (
                                        <MdDone size="3.5rem" className="text-green-600" />
                                    ) : (
                                        <PiUploadSimpleBold size="3.5rem" color="#713105" />
                                    )}

                                    <span className="text-espresso font-medium text-xl text-center break-all px-2">
                                        {file?.name || "No file selected"}
                                    </span>
                                </label>

                                {error ? (
                                    <p className="text-red-600">{error}</p>
                                ) : null}
                            </div>
                        )
                    })}
                </div>
            </div>

            {/* čtvrtá část formuláře - import metod */}
            <div>
                <div className="flex justify-center items-center gap-ds-lg mb-ds-lg">
                    <StepBadge
                        stepNumber={4}
                    />
                    <h2 className="text-3xl font-bold text-noir">Methods import</h2>
                </div>
                <div className="mb-ds-md">
                    <label className="block mb-[8px] font-medium text-noir text-2xl">
                        <div className="flex items-center gap-ds-sm">
                            <Tooltip
                                text={"Upload exported methods from another workflow (JSON)"}
                            >
                                <FaRegQuestionCircle size="1.5rem" color="341100" className="shrink-0" />
                            </Tooltip>
                            Upload exported methods
                        </div>
                    </label>
                    <input
                        id="importFileInput"
                        type="file"
                        accept=".json"
                        onChange={(e) => handleFileInputChange(e.target.files?.[0])}
                        className="sr-only"
                    />
                    <label
                        htmlFor="importFileInput"
                        onDrop={(e) => {
                            e.preventDefault()
                            setImportDragActive(false)
                            processDroppedFile(e.dataTransfer.files?.[0], 'importFile')
                        }}
                        onDragOver={(e) => e.preventDefault()}
                        onDragEnter={() => setImportDragActive(true)}
                        onDragLeave={() => setImportDragActive(false)}
                        className={`w-78 border rounded-[10px] px-ds-md py-ds-xl flex flex-col justify-center items-center gap-ds-sm 
                                    cursor-pointer hover:bg-crema/50 transition ${importDragActive
                                ? 'ring-2 ring-espresso border-transparent'
                                : 'border-dashed border-roast/75'}`}
                    >
                        {hasImportFile ? (
                            <MdDone size="3.5rem" className="text-green-600" />
                        ) : (
                            <PiUploadSimpleBold size="3.5rem" color="#713105" />
                        )}
                        <span className="text-espresso text-xl font-medium text-center break-all px-2">
                            {filesState?.importFile?.file?.name || "No file selected"}
                        </span>
                        {hasImportFile ? (
                            <span className="text-espresso text-base font-semibold">
                                File uploaded
                            </span>
                        ) : null}
                    </label>
                    {filesState?.importFile?.error ? <p className="text-red-600">{filesState.importFile.error}</p> : null}
                </div>
            </div>


            {
                loadError && (
                    <div className="text-red-600 rounded-lg max-w-xl text-center text-2xl">
                        {loadError}
                    </div>
                )
            }

            <button
                type="submit"
                disabled={isSubmitting || hasFileErrors}
                className={`bg-grounds text-foam rounded-4xl py-4 w-50 text-2xl font-semibold  transition duration-300 hover:bg-noir/90
                            mb-ds-lg ${isSubmitting || hasFileErrors
                        ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}`}
            >
                {isSubmitting ? 'Submitting...' : 'Submit'}
            </button>
        </form >
    )
}



export default WorkflowCreationForm