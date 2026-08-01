import { useState } from "react"
import { useNavigate } from "react-router-dom"
import StepBadge from '../molecules/StepBadge'
import { formatNetworkError } from "../../utils/helpers"
import { FaRegQuestionCircle } from "react-icons/fa"
import Tooltip from "../molecules/Tooltip"
import { API_BASE_URL } from "../../config"

function WorkflowCreationForm({ filesState, filesDispatch, loadError, setLoadError }) {
    const navigate = useNavigate()

    const [workflowName, setWorkflowName] = useState("")
    const [folderName, setFolderName] = useState("")
    const [reportFileName, setReportFileName] = useState("")

    const [successMessage, setSuccessMessage] = useState("")
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [dataFormat, setDataFormat] = useState("cd")

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

    function handleFileInputChange(e) {
        const file = e.target.files?.[0]
        if (!file) return
        if (file.name.endsWith('.json')) {
            filesDispatch({ type: 'SET_FILE', key: 'importFile', file })
            filesDispatch({ type: 'SET_ERROR', key: 'importFile', error: null })
        } else {
            filesDispatch({ type: 'SET_ERROR', key: 'importFile', error: 'Please select a JSON file' })
        }
    }

    function handleDataFileChange(e) {
        const file = e.target.files?.[0]
        if (!file) return

        filesDispatch({ type: 'SET_FILE', key: 'data', file })
        filesDispatch({ type: 'SET_ERROR', key: 'data', error: null })
    }

    function handleBatchInfoFileChange(e) {
        const file = e.target.files?.[0]
        if (!file) return

        filesDispatch({ type: 'SET_FILE', key: 'batchInfo', file })
        filesDispatch({ type: 'SET_ERROR', key: 'batchInfo', error: null })
    }

    return (
        <form onSubmit={onSubmit} className="flex flex-col gap-ds-xl items-center">
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
                <div className="flex flex-col">
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
                </div>
            </div >


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
                            type="file"
                            onChange={handleDataFileChange}
                            className="border border-dashed border-roast/75 rounded-[10px] px-3 py-16"
                        />
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
                            type="file"
                            onChange={handleBatchInfoFileChange}
                            className="border border-dashed border-roast/75 rounded-[10px] px-3 py-16"
                        />
                        {filesState?.batchInfo?.error ? <p className="text-red-600">{filesState.batchInfo.error}</p> : null}
                    </div>
                </div>
            </div>

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
                        type="file"
                        accept=".json"
                        onChange={handleFileInputChange}
                        className="border border-dashed border-roast/75 rounded-[10px] px-ds-md py-ds-xl w-full"
                    />
                    {filesState?.importFile?.error ? <p className="text-red-600">{filesState.importFile.error}</p> : null}
                </div>
            </div>

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