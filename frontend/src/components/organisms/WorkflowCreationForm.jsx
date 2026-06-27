import { useState } from "react"
import { useNavigate } from "react-router-dom"
import StepBadge from '../molecules/StepBadge'

const url = "http://127.0.0.1:5000"

function WorkflowCreationForm() {
    const navigate = useNavigate()

    const [workflowName, setWorkflowName] = useState("")
    const [folderName, setFolderName] = useState("")
    const [reportFileName, setReportFileName] = useState("")
    const [uploadData, setUploadData] = useState()
    const [batchInfo, setUploadBatchInfo] = useState()
    const [errorMessage, setErrorMessage] = useState("")
    const [successMessage, setSuccessMessage] = useState("")
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [dataFormat, setDataFormat] = useState("")

    async function onSubmit(e) {
        e.preventDefault()
        setErrorMessage("")

        const formData = new FormData()
        formData.append("workflowName", workflowName)
        formData.append("folderName", folderName)
        formData.append("reportFileName", reportFileName)
        formData.append("dataFormat", dataFormat)

        if (uploadData) {
            formData.append("data", uploadData)
        }
        if (batchInfo) {
            formData.append("batchInfo", batchInfo)
        }

        try {
            const response = await fetch(url + "/new_workflow", {
                method: "POST",
                body: formData,
            })

            const responseData = await response.json()

            if (!response.ok) {
                setErrorMessage(responseData.message || "Workflow se nepodařilo vytvořit.")
                return
            }

            setWorkflowName("")
            setFolderName("")
            setReportFileName("")
            setUploadData(null)
            setUploadBatchInfo(null)

            // přesměrování na layout
            setTimeout(() => {
                navigate(`/workflow/${responseData.workflowId}`)
            }, 1000)
        } catch (error) {
            setErrorMessage("Nastala chyba při komunikaci se serverem.")
        }
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
                <div className="flex flex-col">
                    <label htmlFor="workflowName" className="mb-[8px] font-medium text-noir text-2xl ">Workflow Name *</label>
                    <input
                        type="text"
                        id="workflowName"
                        value={workflowName}
                        onChange={(e) => setWorkflowName(e.target.value)}
                        className="border border-roast/50 rounded-[10px] px-3 py-2 h-14 w-78 focus:border-noir focus:outline-none focus:border-2"
                        required
                    />
                </div>
                <div className="flex flex-col">
                    <label htmlFor="folderName" className="mb-[8px] font-medium text-noir text-2xl">Folder Name *</label>
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
                    <label htmlFor="reportFileName" className="mb-[8px] font-medium text-noir text-2xl">Report File Name *</label>
                    <input
                        type="text"
                        id="reportFileName"
                        value={reportFileName}
                        onChange={(e) => setReportFileName(e.target.value)}
                        className="border border-roast/50 rounded-[10px] px-3 py-2 h-14 w-78 focus:border-noir focus:outline-none focus:border-2"
                        required
                    />
                </div>
            </div>


            <div>
                <div className="flex justify-center items-center gap-ds-lg mb-ds-lg">
                    <StepBadge
                        stepNumber={2}
                    />
                    <h2 className="text-3xl font-bold text-noir">Data import</h2>
                </div>

                <div className="flex flex-row gap-ds-xl">
                    <div className="flex flex-col">
                        <label className="mb-[8px] font-medium text-noir text-2xl">Upload data</label>
                        <input
                            type="file"
                            onChange={(e) => setUploadData(e.target.files?.[0] || null)}
                            className="border border-dashed border-roast/75 rounded-[10px] px-3 py-16"
                        />
                    </div>

                    <div className="flex flex-col">
                        <label className="mb-[8px] font-medium text-noir text-2xl">Upload batch info</label>
                        <input
                            type="file"
                            onChange={(e) => setUploadBatchInfo(e.target.files?.[0] || null)}
                            className="border border-dashed border-roast/75 rounded-[10px] px-3 py-16"
                        />
                    </div>
                </div>
            </div>

            <div>
                <div className="flex justify-center items-center gap-ds-lg mb-ds-lg">
                    <StepBadge
                        stepNumber={3}
                    />
                    <h2 className="text-3xl font-bold text-noir">Data format</h2>
                </div>

                <div className="flex flex-col">
                    <label className="mb-ds-sm font-medium text-noir text-2xl">Format</label>
                    <select
                        id="format"
                        value={dataFormat}
                        onChange={(e) => setDataFormat(e.target.value)}
                        className="border border-roast/75 rounded-lg p-ds-sm h-18 w-80 focus:border-noir focus:outline-none focus:border-2"
                        required
                    >
                        <option value="" disabled>
                            Choose data format
                        </option>
                        <option value="compound_discoverer">
                            Compound Discoverer
                        </option>
                    </select>
                </div>
            </div>


            {errorMessage ? <p className="text-red-600">{errorMessage}</p> : null}

            <button type="submit" className="bg-grounds text-foam rounded-4xl py-4 w-50 text-2xl font-semibold cursor-pointer">
                Submit
            </button>
        </form>
    )
}



export default WorkflowCreationForm