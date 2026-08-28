import { useState } from "react"
import { useNavigate } from "react-router-dom"
import StepBadge from '../../molecules/HomePage/StepBadge'
import { formatNetworkError } from "../../../utils/helpers"
import { FaRegQuestionCircle } from "react-icons/fa"
import Tooltip from "../../molecules/WorkflowLayout/Tooltip"
import { API_BASE_URL } from "../../../config"
import { PiUploadSimpleBold } from "react-icons/pi"

function WorkflowCreationForm({ filesState, filesDispatch, loadError, setLoadError }) {
    const navigate = useNavigate()

    // údaje pro vytvoření workflow
    const [workflowName, setWorkflowName] = useState("")
    const [folderName, setFolderName] = useState("")
    const [reportFileName, setReportFileName] = useState("")
    const [dataFormat, setDataFormat] = useState("cd")

    // pro změnu designu
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [dataDragActive, setDataDragActive] = useState(false)
    const [batchDragActive, setBatchDragActive] = useState(false)
    const [importDragActive, setImportDragActive] = useState(false)

    // volá se po dropu, podle typu souboru zavolá funkci pro jeho získání
    function processDroppedFile(file, key) {
        if (!file) return

        switch (key) {
            case 'data':
                handleDataFileChange(file)
                break
            case 'batchInfo':
                handleBatchInfoFileChange(file)
                break
            case 'importFile':
                handleFileInputChange(file)
                break
            default:
                break
        }
    }

    // submit formuláře
    async function onSubmit(e) {
        e.preventDefault()

        const hasFileErrors = Boolean(
            filesState?.data?.error || filesState?.batchInfo?.error || filesState?.importFile?.error
        )
        if (hasFileErrors) {
            const firstErr = filesState?.data?.error || filesState?.batchInfo?.error || filesState?.importFile?.error
            setLoadError(firstErr)
            return
        }

        // kontrola, jestli uživatel nahrál jak data tak batch info
        const missingFiles = []
        if (!filesState?.data?.file) missingFiles.push('data')
        if (!filesState?.batchInfo?.file) missingFiles.push('batchInfo')
        if (missingFiles.length) {
            missingFiles.forEach((k) => {
                const err = k === 'data' ? 'Data file is required.' : 'Batch info file is required.'
                filesDispatch({ type: 'SET_ERROR', key: k, error: err })
            })

            let erorrMsg = ""
            if (missingFiles.length === 2) {
                erorrMsg = "Data file and Batch info file are required."
            } else {
                erorrMsg = missingFiles[0] === 'data' ? "Data file is required." : "Batch info file is required."
            }

            setLoadError(erorrMsg)
            return
        }

        const formData = new FormData()
        formData.append("workflowName", workflowName)
        formData.append("folderName", folderName)
        formData.append("reportFileName", reportFileName)
        formData.append("dataFormat", dataFormat)

        if (filesState?.data?.file) {
            formData.append("data", filesState.data.file)
        }

        if (filesState?.batchInfo?.file) {
            formData.append("batchInfo", filesState.batchInfo.file)
        }

        if (filesState?.importFile?.file) {
            formData.append("importFile", filesState.importFile.file)
        }

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

            setWorkflowName("")
            setFolderName("")
            setReportFileName("")
            filesDispatch({ type: 'CLEAR' })
            setLoadError(null)

            // přesměrování na layout
            navigate(`/workflow/${responseData.workflowId}`)
        } catch (err) {
            setLoadError(formatNetworkError(err))
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

    // funkce pro získání vstupních dat
    function handleDataFileChange(file) {
        if (!file) return

        filesDispatch({ type: 'SET_FILE', key: 'data', file })
        filesDispatch({ type: 'SET_ERROR', key: 'data', error: null })
    }

    // funkce pro získání batch info
    function handleBatchInfoFileChange(file) {
        if (!file) return

        filesDispatch({ type: 'SET_FILE', key: 'batchInfo', file })
        filesDispatch({ type: 'SET_ERROR', key: 'batchInfo', error: null })
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

            {/* druhá část formuláře - vstupní data a batch info */}
            <div>
                <div className="flex justify-center items-center gap-ds-lg mb-ds-lg">
                    <StepBadge
                        stepNumber={2}
                    />
                    <h2 className="text-3xl font-bold text-noir">Data import</h2>
                </div>

                <div className="flex flex-row gap-ds-xl">
                    <div className="flex flex-col">
                        <label className="mb-[8px] font-medium text-noir text-2xl">
                            <div className="flex items-center gap-ds-sm">
                                <Tooltip
                                    text={"Input data file. The file must be in spreadsheet format."}
                                >
                                    <FaRegQuestionCircle size="1.5rem" color="341100" className="shrink-0" />
                                </Tooltip>
                                Upload data *
                            </div>
                        </label>
                        <input
                            id="dataFileInput"
                            type="file"
                            onChange={(e) => handleDataFileChange(e.target.files?.[0])}
                            className="sr-only"
                        />
                        <label
                            htmlFor="dataFileInput"
                            onDrop={(e) => {
                                e.preventDefault()
                                setDataDragActive(false)
                                processDroppedFile(e.dataTransfer.files?.[0], 'data')
                            }}
                            onDragOver={(e) => e.preventDefault()}
                            onDragEnter={() => setDataDragActive(true)}
                            onDragLeave={() => setDataDragActive(false)}
                            className={`w-78 border rounded-[10px] px-ds-md py-ds-xl flex flex-col justify-center items-center gap-ds-sm 
                                        cursor-pointer hover:bg-crema/50 transition ${dataDragActive ? 'ring-2 ring-espresso border-transparent' : 'border-dashed border-roast/75'}`}
                        >
                            <PiUploadSimpleBold size="3.5rem" color="#713105" />
                            <span className="text-espresso font-medium text-xl text-center break-all px-2">
                                {filesState?.data?.file?.name || "No file selected"}
                            </span>
                        </label>
                        {filesState?.data?.error ? <p className="text-red-600">{filesState.data.error}</p> : null}
                    </div>

                    <div className="flex flex-col">
                        <label className="mb-[8px] font-medium text-noir text-2xl">
                            <div className="flex items-center gap-ds-sm">
                                <Tooltip
                                    text={"Batch information file. The file must be in spreadsheet format."}
                                >
                                    <FaRegQuestionCircle size="1.5rem" color="341100" className="shrink-0" />
                                </Tooltip>
                                Upload batch info *
                            </div>
                        </label>
                        <input
                            id="batchInfoFileInput"
                            type="file"
                            onChange={(e) => handleBatchInfoFileChange(e.target.files?.[0])}
                            className="sr-only"
                        />
                        <label
                            htmlFor="batchInfoFileInput"
                            onDrop={(e) => {
                                e.preventDefault()
                                setBatchDragActive(false)
                                processDroppedFile(e.dataTransfer.files?.[0], 'batchInfo')
                            }}
                            onDragOver={(e) => e.preventDefault()}
                            onDragEnter={() => setBatchDragActive(true)}
                            onDragLeave={() => setBatchDragActive(false)}
                            className={`w-78 border rounded-[10px] px-ds-md py-ds-xl flex flex-col justify-center items-center gap-ds-sm 
                                        cursor-pointer hover:bg-crema/50 transition ${batchDragActive ? 'ring-2 ring-espresso border-transparent' : 'border-dashed border-roast/75'}`}
                        >
                            <PiUploadSimpleBold size="3.5rem" color="#713105" />
                            <span className="text-espresso text-xl font-medium text-center break-all px-2">
                                {filesState?.batchInfo?.file?.name || "No file selected"}
                            </span>
                        </label>
                        {filesState?.batchInfo?.error ? <p className="text-red-600">{filesState.batchInfo.error}</p> : null}
                    </div>
                </div>
            </div>

            {/* třetí část formuláře - import metod */}
            <div>
                <div className="flex justify-center items-center gap-ds-lg mb-ds-lg">
                    <StepBadge
                        stepNumber={3}
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
                                    cursor-pointer hover:bg-crema/50 transition ${importDragActive ? 'ring-2 ring-espresso border-transparent' : 'border-dashed border-roast/75'}`}
                    >
                        <PiUploadSimpleBold size="3.5rem" color="#713105" />
                        <span className="text-espresso text-xl font-medium text-center break-all px-2">
                            {filesState?.importFile?.file?.name || "No file selected"}
                        </span>
                    </label>
                    {filesState?.importFile?.error ? <p className="text-red-600">{filesState.importFile.error}</p> : null}
                </div>
            </div>

            {/* čtvrtá část formuláře - formát vstupních dat */}
            <div>
                <div className="flex justify-center items-center gap-ds-lg mb-ds-lg">
                    <StepBadge
                        stepNumber={4}
                    />
                    <h2 className="text-3xl font-bold text-noir">Data format</h2>
                </div>

                <div className="flex flex-col">
                    <label className="mb-ds-sm font-medium text-noir text-2xl">
                        <div className="flex items-center gap-ds-sm">
                            <Tooltip
                                text={"Format of the input data (e.g., Compound Discoverer)"}
                            >
                                <FaRegQuestionCircle size="1.5rem" color="341100" className="shrink-0" />
                            </Tooltip>
                            Data format *
                        </div>
                    </label>
                    <select
                        id="format"
                        value={dataFormat}
                        onChange={(e) => setDataFormat(e.target.value)}
                        className="border border-roast/75 rounded-lg p-ds-sm h-18 w-80 focus:border-noir focus:outline-none focus:border-2"
                        required
                    >
                        <option value="cd" className="">Compound Discoverer</option>
                    </select>
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
                disabled={isSubmitting || Boolean(filesState?.data?.error || filesState?.batchInfo?.error || filesState?.importFile?.error)}
                className={`bg-grounds text-foam rounded-4xl py-4 w-50 text-2xl font-semibold  transition duration-300 hover:bg-noir/90
                            mb-ds-lg ${isSubmitting || Boolean(filesState?.data?.error || filesState?.batchInfo?.error || filesState?.importFile?.error)
                        ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}`}
            >
                {isSubmitting ? 'Submitting...' : 'Submit'}
            </button>
        </form >
    )
}



export default WorkflowCreationForm