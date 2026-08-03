import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import WorkflowCard from "../components/molecules/WorkflowList/WorkflowCard"
import ErrorAlert from "../components/molecules/WorkflowLayout/ErrorAlert"
import { formatNetworkError } from "../utils/helpers"
import { API_BASE_URL } from "../config"

function WorkflowsList() {
    const navigate = useNavigate()
    const [workflows, setWorkflows] = useState(null)
    const [selectedWorkflow, setSelectedWorkflow] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [loadingError, setLoadingError] = useState(null)

    //Delete
    const [isDeleting, setIsDeleting] = useState(false)
    const [deleteError, setDeleteError] = useState(false)
    const [deleteSucces, setDeleteSuccess] = useState(null)

    useEffect(() => {
        async function loadWorkflows() {
            setIsLoading(true)

            try {
                const response = await fetch(API_BASE_URL + "/workflows")
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

    useEffect(() => {
        if (isDeleting) {
            document.body.classList.add("app-busy-cursor")
        } else {
            document.body.classList.remove("app-busy-cursor")
        }

        return () => {
            document.body.classList.remove("app-busy-cursor")
        }
    }, [isDeleting])

    // auto-hide u úspěšného odstranění
    useEffect(() => {
        if (!deleteSucces) return
        const timeout = setTimeout(() => setDeleteSuccess(null), 3000)
        return () => clearTimeout(timeout)
    }, [deleteSucces])

    // auto-hide u chyby při odstranění
    useEffect(() => {
        if (!deleteError) return
        const timeout = setTimeout(() => setDeleteError(false), 3000)
        return () => clearTimeout(timeout)
    }, [deleteError])

    async function handleDeleteWorkflow(workflowId) {
        setIsDeleting(true)

        try {
            const response = await fetch(API_BASE_URL + `/workflow/${workflowId}/delete`, {
                method: "DELETE"
            })

            const data = await response.json()

            if (!response.ok) {
                setDeleteError(data?.message ?? "Failed to delete workflow.")
                return
            }

            // obnovení workflows
            const responseWorkflows = await fetch(API_BASE_URL + "/workflows")
            const dataWorkflows = await responseWorkflows.json()
            setWorkflows(dataWorkflows)
            setDeleteSuccess("Workflow was successfully deleted")
        } catch (err) {
            setDeleteError(formatNetworkError(err))
        } finally {
            setIsDeleting(false)
        }
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

            {deleteError && (
                <div className="fixed top-ds-lg left-1/2 transform -translate-x-1/2 z-40">
                    <ErrorAlert
                        message={deleteError}
                        className="mb-ds-md p-ds-md rounded-lg max-w-xl mx-auto text-center"
                        onDismiss={() => setDeleteError(false)}
                    />
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
                        disabled={!selectedWorkflow || isDeleting}
                        className={`bg-grounds text-foam rounded-4xl py-ds-md w-50 text-2xl font-semibold ${!selectedWorkflow ? "opacity-85 cursor-not-allowed" : "cursor-pointer transition duration-300 hover:bg-noir/90"}`}
                        onClick={() => {
                            navigate(`/workflow/${selectedWorkflow}`)
                        }}
                    >
                        Continue
                    </button>
                </div>
            )}
        </div>
    )
}

export default WorkflowsList