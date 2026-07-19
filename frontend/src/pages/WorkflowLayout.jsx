import { useState, useEffect } from "react"
import { useParams, useNavigate } from "react-router-dom"
import WorkflowSidebar from "../components/organisms/Layouts/WorkflowSidebar"
import WorkflowContent from "../components/organisms/Layouts/WorkflowContent"
import WorkflowError from "../components/organisms/WorkflowError"
import { formatNetworkError } from "../utils/helpers"

const url = "http://127.0.0.1:5000"

function WorkflowLayout() {
    const { workflowId } = useParams()
    const navigate = useNavigate()
    const [workflow, setWorkflow] = useState(null)
    const [error, setError] = useState("")
    const [workflowLoadErrror, setWorkflowLoadError] = useState(null)
    const [operations, setOperations] = useState([])
    const [searchQuery, setSearchQuery] = useState("")
    const [selectedStep, setSelectedStep] = useState(null)
    const [isRunning, setIsRunning] = useState(false)
    const [isAllStepsRunning, setIsAllStepsRunning] = useState(false)
    const [isWorkflowLoading, setIsWorkflowLoading] = useState(false)
    const [runningStepId, setRunningStepId] = useState(null)
    const [isStepDeleting, setIsStepDeleting] = useState(false)
    const [addedOperationId, setAddedOperationId] = useState(null)
    const [stepExecutionMessage, setStepExecutionMessage] = useState(null)

    useEffect(() => {
        async function loadWorkflow() {
            setIsWorkflowLoading(true)

            try {
                const response = await fetch(url + `/workflow/id/${workflowId}`)
                const data = await response.json()

                if (!response.ok) {
                    setWorkflowLoadError("not-found")
                    return
                }

                setWorkflow(data)
                setWorkflowLoadError(null)
            } catch (err) {
                setWorkflowLoadError("network")
            } finally {
                setIsWorkflowLoading(false)
            }
        }

        // pro vyhledávání se načtou všechny operace
        async function loadOperations() {
            try {
                const response = await fetch(url + "/operations")
                const data = await response.json()

                if (!response.ok) {
                    setError(data?.message ?? "Failed to load available operations from server.")
                    return
                }

                setOperations(data)
                setError(null)
            } catch (err) {
                setError(err?.message ?? "Network error while fetching operations.")
            }
        }

        loadWorkflow()
        loadOperations()
    }, [workflowId])

    // vrací pouze ty metody, které obsahují zadaný výraz ze search baru
    const filteredOperations = operations.filter((op) => {
        const query = searchQuery.toLowerCase()
        return op.label.toLowerCase().includes(query)
    })



    // funkce pro obnovu workflow po vykonání nějakého kroku
    async function refreshWorkflow() {
        try {
            const resp = await fetch(url + `/workflow/id/${workflowId}`)
            const data = await resp.json()

            if (!resp.ok) {
                setError(data?.message ?? "Failed to refresh workflow from server.")
                return null
            }

            setWorkflow(data)
            setError(null)
            return data
        } catch (err) {
            setError(err?.message ?? "Network error while refreshing workflow.")
            return null
        }
    }


    async function handleAddStep(operation) {
        setAddedOperationId(operation.id)

        const stepData = {
            operationId: operation.id
        }

        try {
            const response = await fetch(url + `/workflow/${workflowId}/step`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(stepData),
            })

            const data = await response.json()

            if (!response.ok) {
                setError(data?.message ?? "Failed to add step to workflow.")
                return
            }

            await refreshWorkflow()
            setSearchQuery("")
        } catch (err) {
            setError(formatNetworkError(err))
        } finally {
            setAddedOperationId(null)
        }
    }

    async function handleExecuteStep(stepId) {
        setIsRunning(true) // krok běží, asi se může odstranit
        setRunningStepId(stepId)

        try {

            const response = await fetch(url + `/workflow/${workflowId}/step/${stepId}/run`, {
                method: "POST",
            })

            const dataResponse = await response.json()

            if (!response.ok) {
                setError(dataResponse?.message ?? "Failed to run the selected step.")
                return
            }

            const stepOperation = operations.find(op => op.id === dataResponse.stepOperationId)
            const operationLabel = stepOperation?.label ?? dataResponse.stepOperationId
            const status = dataResponse?.stepStatus

            setStepExecutionMessage({ operation: operationLabel, message: dataResponse?.stepMessage, status: dataResponse?.stepStatus })
            await refreshWorkflow()
        } catch (err) {
            setError(formatNetworkError(err))
        } finally {
            setIsRunning(false) // krok doběhl
            setRunningStepId(null)
        }
    }

    async function handleExecuteAllSteps() {
        setIsAllStepsRunning(true)

        const steps = workflow?.definition?.steps.filter((step) => {
            return step.status !== "done"
        }) || []


        try {
            if (steps.length === 0) {
                setError("All steps are already done.")
                return
            }

            for (let i = 0; i < steps.length; i++) {
                const step = steps[i]

                await handleExecuteStep(step.step_id)

                if (error) {
                    break
                }
            }
        } catch (err) {
            setError(formatNetworkError(err))
        } finally {
            setIsAllStepsRunning(false)
        }
    }

    async function handleDeleteStep(stepId) {
        setIsStepDeleting(true)

        try {
            const response = await fetch(url + `/workflow/${workflowId}/delete_step/${stepId}`, {
                method: "DELETE"
            })

            const data = await response.json()

            if (!response.ok) {
                setError(data?.message ?? "Failed to delete the step from workflow.")
                return
            }

            await refreshWorkflow()
            setSelectedStep(null)
        } catch (err) {
            setError(formatNetworkError(err))
        } finally {
            setIsStepDeleting(false)
        }
    }

    async function handleCloseParameters() {
        // Zavře form a obnoví workflow data
        setSelectedStep(null)

        try {
            await refreshWorkflow()
        } catch (err) {
            setError(formatNetworkError(err))
        }
    }

    return (
        <div className="min-h-screen flex">
            <WorkflowSidebar
                searchQuery={searchQuery}
                setSearchQuery={setSearchQuery}
                filteredOperations={filteredOperations}
                workflow={workflow}
                operations={operations}
                onAddStep={handleAddStep}
                onDeleteStep={handleDeleteStep}
                onSelectStep={setSelectedStep}
                selectedStep={selectedStep}
                onExecuteStep={handleExecuteStep}
                onExecuteAll={handleExecuteAllSteps}
                isStepRunning={isRunning}
                runningStepId={runningStepId}
                isWorkflowLoading={isWorkflowLoading}
                isStepDeleting={isStepDeleting}
                addedOperationId={addedOperationId}
                isRunningAll={isAllStepsRunning}
            />
            <WorkflowContent
                workflow={workflow}
                selectedStep={selectedStep}
                operations={operations}
                workflowId={workflowId}
                onCloseParameters={handleCloseParameters}
                isLoading={isWorkflowLoading}
                apiBaseUrl={url}
                error={error}
                workflowError={workflowLoadErrror}
                stepExecutionMessage={stepExecutionMessage}
            />
        </div>
    )
}

export default WorkflowLayout
