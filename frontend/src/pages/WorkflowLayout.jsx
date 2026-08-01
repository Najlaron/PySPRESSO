import { useState, useEffect, useRef } from "react"
import { useParams, useNavigate } from "react-router-dom"
import WorkflowSidebar from "../components/organisms/Layouts/WorkflowSidebar"
import WorkflowContent from "../components/organisms/Layouts/WorkflowContent"
import WorkflowError from "../components/organisms/WorkflowError"
import { formatNetworkError } from "../utils/helpers"
import { API_BASE_URL } from "../config"

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
    const [executeFailed, setExecuteFailed] = useState(false)
    const reorderPromiseRef = useRef(Promise.resolve())
    const parameterSubmitPromiseRef = useRef(Promise.resolve())
    const isBusy = Boolean(
        runningStepId || isRunning || isAllStepsRunning || isStepDeleting || addedOperationId
    )

    useEffect(() => {
        if (isBusy) {
            document.body.classList.add("app-busy-cursor")
        } else {
            document.body.classList.remove("app-busy-cursor")
        }

        return () => {
            document.body.classList.remove("app-busy-cursor")
        }
    }, [isBusy])

    useEffect(() => {
        async function loadWorkflow() {
            setIsWorkflowLoading(true)

            try {
                const response = await fetch(API_BASE_URL + `/workflow/id/${workflowId}`)
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
                const response = await fetch(API_BASE_URL + "/operations")
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
            const resp = await fetch(API_BASE_URL + `/workflow/id/${workflowId}`)
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

    function registerParametersSubmitPromise(submitPromise) {
        parameterSubmitPromiseRef.current = Promise.resolve(submitPromise).catch(() => null)
    }


    async function handleAddStep(operation) {
        await parameterSubmitPromiseRef.current

        setAddedOperationId(operation.id)
        await reorderPromiseRef.current // čeká se na dokončení změny pořadí

        const stepData = {
            operationId: operation.id
        }

        try {
            const response = await fetch(API_BASE_URL + `/workflow/${workflowId}/step`, {
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
        await parameterSubmitPromiseRef.current

        setIsRunning(true) // krok běží, asi se může odstranit
        setRunningStepId(stepId)

        await reorderPromiseRef.current // čeká se na dokončení změny pořadí

        try {

            const response = await fetch(API_BASE_URL + `/workflow/${workflowId}/step/${stepId}/run`, {
                method: "POST",
            })

            const dataResponse = await response.json()

            if (!response.ok) {
                setError(dataResponse?.message ?? "Failed to run the selected step.")
                return false
            }

            const stepOperation = operations.find(op => op.id === dataResponse.stepOperationId)
            const operationLabel = stepOperation?.label ?? dataResponse.stepOperationId
            const status = dataResponse?.stepStatus

            setStepExecutionMessage({ operation: operationLabel, message: dataResponse?.stepMessage, status: dataResponse?.stepStatus })
            await refreshWorkflow()

            if (dataResponse.stepStatus !== "done") {
                return false
            }

            return true
        } catch (err) {
            setError(formatNetworkError(err))
            return false
        } finally {
            setIsRunning(false) // krok doběhl
            setRunningStepId(null)
        }
    }

    async function handleExecuteAllSteps() {
        await parameterSubmitPromiseRef.current

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

                const success = await handleExecuteStep(step.step_id)

                if (!success) {
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
        await parameterSubmitPromiseRef.current
        await reorderPromiseRef.current // čeká se na dokončení změny pořadí
        setIsStepDeleting(true)

        try {
            const response = await fetch(API_BASE_URL + `/workflow/${workflowId}/delete_step/${stepId}`, {
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

    async function handleReorderSteps(reorderedStepIds) {
        await parameterSubmitPromiseRef.current

        const reorderPromise = (async () => {
            setWorkflow((prevWorkflow) => {
                if (!prevWorkflow?.definition?.steps) return prevWorkflow

                const stepsById = new Map(prevWorkflow.definition.steps.map((step) => [step.step_id, step]))
                const reorderedSteps = reorderedStepIds
                    .map((stepId) => stepsById.get(stepId))
                    .filter(Boolean)

                return {
                    ...prevWorkflow,
                    definition: {
                        ...prevWorkflow.definition,
                        steps: reorderedSteps
                    }
                }
            })

            try {
                const response = await fetch(API_BASE_URL + `/workflow/${workflowId}/reorder`, {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ stepIds: reorderedStepIds }),
                })

                const data = await response.json()

                if (!response.ok) {
                    setError(data?.message ?? "Failed to reorder workflow steps.")
                    return
                }

                await refreshWorkflow()
            } catch (err) {
                setError(formatNetworkError(err))
            }
        })()

        reorderPromiseRef.current = reorderPromise
        return reorderPromise
    }

    return (
        <div className="h-screen overflow-hidden flex">
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
                onReorderSteps={handleReorderSteps}
            />
            <WorkflowContent
                workflow={workflow}
                selectedStep={selectedStep}
                operations={operations}
                workflowId={workflowId}
                onRefreshWorkflow={refreshWorkflow}
                onCloseParameters={handleCloseParameters}
                onParametersSubmitPromiseChange={registerParametersSubmitPromise}
                onDismissError={() => setError("")}
                isLoading={isWorkflowLoading}
                apiBaseUrl={API_BASE_URL}
                error={error}
                workflowError={workflowLoadErrror}
                stepExecutionMessage={stepExecutionMessage}
                reorderPromise={reorderPromiseRef.current}
            />
        </div>
    )
}

export default WorkflowLayout
