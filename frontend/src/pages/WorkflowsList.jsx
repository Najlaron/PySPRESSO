import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import WorkflowCard from "../components/molecules/WorkflowList/WorkflowCard"
import { formatNetworkError } from "../utils/helpers"

const url = "http://127.0.0.1:5000"

function WorkflowsList() {
    const navigate = useNavigate()
    const [workflows, setWorkflows] = useState(null)
    const [selectedWorkflow, setSelectedWorkflow] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [loadingError, setLoadingError] = useState(null)

    // Import
    const [importModalOpen, setImportModalOpen] = useState(false)
    const [selectedImportFile, setSelectedImportFile] = useState(null)
    const [importError, setImportError] = useState(null)
    const [notJsonFileError, setNotJsonFileError] = useState(null)
    const [isImporting, setIsImporting] = useState(false)

    //Delete
    const [isDeleting, setIsDeleting] = useState(false)
    const [deleteError, setDeleteError] = useState(false)
    const [deleteSucces, setDeleteSuccess] = useState(null)

    useEffect(() => {
        async function loadWorkflows() {
            setIsLoading(true)

            try {
                const response = await fetch(url + "/workflows")
                const data = await response.json()

                if (!response.ok) {
                    setLoadingError(data?.message ?? "Failed to load workflows.")
                    return
                }

                setWorkflows(data)
            } catch (err) {
                setLoadingError(formatNetworkError(err))
            } finally {
                setIsLoading(false)
            }
        }

        loadWorkflows()
    }, [])

    // auto-hide success message after a short delay
    useEffect(() => {
        if (!deleteSucces) return
        const timeout = setTimeout(() => setDeleteSuccess(null), 3000)
        return () => clearTimeout(timeout)
    }, [deleteSucces])

    async function handleDeleteWorkflow(workflowId) {
        setIsDeleting(true)

        try {
            const response = await fetch(url + `/workflow/${workflowId}/delete`, {
                method: "DELETE"
            })

            const data = await response.json()

            if (!response.ok) {
                setDeleteError(data?.message ?? "Failed to delete workflow.")
                return
            }

            // obnovení workflows
            const responseWorkflows = await fetch(url + "/workflows")
            const dataWorkflows = await responseWorkflows.json()
            setWorkflows(dataWorkflows)
            setDeleteSuccess("Workflow was successfully deleted")
        } catch (err) {
            setDeleteError(formatNetworkError(err))
        } finally {
            setIsDeleting(false)
        }
    }

    async function handleImportSubmit(e) {
        e.preventDefault()
        setIsImporting(true)
        if (!selectedImportFile) {
            setImportError("No file selected")
            return
        }

        try {
            const formData = new FormData()
            formData.append("file", selectedImportFile)

            const response = await fetch(url + `/workflow/import`, {
                method: "POST",
                body: formData,
            })

            const data = await response.json()

            if (!response.ok) {
                setImportError(data?.message ?? "Failed to import workflow.")
                setImportModalOpen(false)
                return
            }

            setSelectedImportFile(null)
            setImportModalOpen(false)

            // přesměrování na layout
            setTimeout(() => {
                navigate(`/workflow/${data.workflowId}`)
            }, 500)
        } catch (err) {
            setImportError(formatNetworkError(err))
            setImportModalOpen(false)
        }
        finally {
            setIsImporting(false)
        }
    }

    function handleFileInputChange(e) {
        const file = e.target.files?.[0]
        if (file && file.name.endsWith(".json")) {
            setSelectedImportFile(file)
            setImportError(null)
        } else if (file) {
            setNotJsonFileError("Please select a JSON file")
        }
    }

    function closeImportModal() {
        setImportModalOpen(false)
        setSelectedImportFile(null)
        setImportError(null)
    }

    return (
        <div className="p-8 bg-foam min-h-screen">
            <h1 className="text-4xl font-bold mb-ds-lg text-center text-noir">Choose Workflow</h1>

            {deleteSucces && (
                <div className="fixed top-ds-lg left-1/2 transform -translate-x-1/2 z-40">
                    <div className="mb-ds-md p-ds-md border border-green-200 bg-green-100 text-green-800 rounded-lg max-w-xl mx-auto text-center text-xl">
                        {deleteSucces}
                    </div>
                </div>
            )}

            {importError && (
                <div className="mb-ds-md p-ds-md border border-red-200 bg-red-100 text-red-800 rounded-lg max-w-xl mx-auto text-center text-xl">
                    {importError}
                </div>
            )}

            {deleteError && (
                <div className="mb-ds-md p-ds-md border border-red-200 bg-red-100 text-red-800 rounded-lg max-w-xl mx-auto text-center text-xl">
                    {deleteError}
                </div>
            )}

            {isLoading && (
                <div className="flex justify-center flex-col items-center gap-ds-sm">
                    <div className="loader"></div>
                    <h2 className="text-noir/70 text-2xl font-semibold">Loading workflows</h2>
                </div>
            )}

            {!isLoading && workflows?.length === 0 && (
                <h2 className="text-xl text-noir/50 text-center italic">You don't have any workflows saved</h2>
            )}

            {!isLoading && workflows?.length > 0 && (


                <div className="flex flex-col gap-ds-md items-center">
                    <div className="flex">
                        <button
                            onClick={() => {
                                setImportModalOpen(true)
                                setImportError(null)
                            }}
                            className="bg-noir text-foam rounded-4xl py-ds-md px-ds-lg text-lg font-semibold cursor-pointer transition duration-300 hover:bg-noir/80"
                        >
                            Import Workflow
                        </button>
                    </div>
                    <div className="mb-8 flex flex-col gap-ds-md">
                        {workflows?.map((workflow) => (
                            <WorkflowCard
                                key={workflow.id}
                                workflow={workflow}
                                isSelected={workflow.id === selectedWorkflow}
                                onClick={() => setSelectedWorkflow(workflow.id === selectedWorkflow ? null : workflow.id)} // možnost odkliknutí
                                onDelete={() => handleDeleteWorkflow(workflow.id)}
                                isDeleting={isDeleting}
                            />
                        ))}
                    </div>
                    <button
                        type="submit"
                        disabled={!selectedWorkflow}
                        className={`bg-grounds text-foam rounded-4xl py-ds-md w-50 text-2xl font-semibold ${!selectedWorkflow ? "opacity-85 cursor-not-allowed" : "cursor-pointer transition duration-300 hover:bg-noir/90"}`}
                        onClick={() => {
                            setTimeout(() => {
                                navigate(`/workflow/${selectedWorkflow}`)
                            }, 1000)
                        }}
                    >
                        Continue
                    </button>
                </div>
            )}


            {/* Modální okno */}
            {importModalOpen && (
                <div className="fixed inset-0 bg-noir/25 flex items-center justify-center z-50">
                    <div className="bg-foam rounded-2xl p-ds-lg max-w-md w-full mx-ds-md shadow-xl">
                        <h2 className="text-2xl font-bold mb-ds-lg text-noir text-center">Import Workflow</h2>

                        <form onSubmit={handleImportSubmit}>
                            <div className="mb-ds-md">
                                <label className="block mb-ds-sm text-noir text-lg">Select JSON file</label>
                                <input
                                    type="file"
                                    accept=".json"
                                    onChange={handleFileInputChange}
                                    className="border border-dashed border-roast/75 rounded-[10px] px-ds-md py-ds-xl w-full"
                                />
                            </div>

                            {notJsonFileError && (
                                <p className="text-red-700 text-lg text-medium pb-ds-sm">{notJsonFileError}</p>
                            )}

                            <div className="flex gap-ds-sm">
                                <button
                                    type="button"
                                    onClick={closeImportModal}
                                    className="flex-1 bg-crema/85 text-noir rounded-lg py-ds-sm font-medium cursor-pointer transition hover:bg-crema"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    disabled={!selectedImportFile || isImporting}
                                    className={`flex-1 rounded-lg py-ds-sm font-medium transition bg-grounds text-foam hover:bg-noir/90 
                                        ${isImporting
                                            ? "cursor-not-allowed" : "cursor-pointer"
                                        }`}
                                >
                                    {isImporting ? "Importing" : "Import"}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    )
}

export default WorkflowsList