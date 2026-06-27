import { useState, useEffect } from "react"
import { useParams } from "react-router-dom"
import WorkflowSidebar from "../components/organisms/Layouts/WorkflowSidebar"
import WorkflowContent from "../components/organisms/Layouts/WorkflowContent"

const url = "http://127.0.0.1:5000"

function WorkflowLayout() {
    const { workflowId } = useParams()

    const [workflow, setWorkflow] = useState(null)
    const [error, setError] = useState("")
    const [operations, setOperations] = useState([])
    const [searchQuery, setSearchQuery] = useState("")
    const [selectedStep, setSelectedStep] = useState(null)

    const [runningStepId, setRunningStepId] = useState(null)

    const [selectedCategoryTags, setSelectedCategoryTags] = useState([])

    useEffect(() => {
        async function loadWorkflow() {
            try {
                const response = await fetch(url + `/workflow/${workflowId}`)
                const data = await response.json()

                if (!response.ok) {
                    setError(data.message || "Workflow se nepodařilo načíst.")
                    return
                }

                setWorkflow(data)
            } catch (error) {
                setError("Nastala chyba při komunikaci se serverem.")
            }
        }

        async function loadOperations() {
            try {
                const response = await fetch(url + "/operations")
                const data = await response.json()

                if (!response.ok) {
                    console.error("Chyba při načítání operací:", data)
                    return
                }

                setOperations(data)
            } catch (error) {
                console.error("Chyba při načítání operací:", error)
            }
        }

        loadWorkflow()
        loadOperations()
    }, [workflowId])

    const allCategoryTags = Array.from(
        new Set(
            operations.flatMap((operation) => operation.categoryTags || [])
        )
    ).sort()

    function toggleCategoryTag(tag) {
        setSelectedCategoryTags((currentTags) =>
            currentTags.includes(tag)
                ? currentTags.filter((currentTag) => currentTag !== tag)
                : [...currentTags, tag]
        )
    }

    function clearCategoryTags() {
        setSelectedCategoryTags([])
    }

    const filteredOperations = operations.filter((op) => {
        const query = searchQuery.trim().toLowerCase()
        const operationTags = op.categoryTags || []

        const matchesSearch =
            !query ||
            op.label?.toLowerCase().includes(query) ||
            op.id?.toLowerCase().includes(query) ||
            op.description?.toLowerCase().includes(query)

        const matchesTags =
            selectedCategoryTags.length === 0 ||
            selectedCategoryTags.every((tag) => operationTags.includes(tag))

        return matchesSearch && matchesTags
    })

    async function handleAddStep(operation) {
        const stepData = {
            operationId: operation.id,
        }

        try {
            const response = await fetch(url + `/workflow/${workflowId}/step`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(stepData),
            })

            const data = await response.json()

            if (!response.ok) {
                alert("Chyba: " + (data.message || "Nepodařilo se přidat krok"))
                return
            }

            const workflowResponse = await fetch(url + `/workflow/${workflowId}`)
            const updatedWorkflow = await workflowResponse.json()

            setWorkflow(updatedWorkflow)
            setSearchQuery("")
        } catch (error) {
            alert("Chyba: " + error.message)
        }
    }

    async function handleExecuteStep(stepId) {
        setRunningStepId(stepId)

        try {
            const response = await fetch(url + `/workflow/${workflowId}/step/${stepId}/run`, {
                method: "POST",
            })

            const dataResponse = await response.json()

            if (!response.ok) {
                console.log("Chyba: " + (dataResponse.message || "Nepodařilo se spustit krok"))
                return
            }

            const workflowResponse = await fetch(url + `/workflow/${workflowId}`)
            const updatedWorkflow = await workflowResponse.json()

            setWorkflow(updatedWorkflow)
        } catch (error) {
            alert("Chyba: " + error.message)
        } finally {
            setRunningStepId(null)
        }
    }

    async function handleDeleteStep(stepId) {
        try {
            const response = await fetch(url + `/workflow/${workflowId}/delete_step/${stepId}`, {
                method: "DELETE",
            })

            const data = await response.json()

            if (!response.ok) {
                alert("Chyba: " + (data.message || "Nepodařilo se smazat krok"))
                return
            }

            const workflowResponse = await fetch(url + `/workflow/${workflowId}`)
            const updatedWorkflow = await workflowResponse.json()

            setWorkflow(updatedWorkflow)
            setSelectedStep(null)
        } catch (error) {
            alert("Chyba: " + error.message)
        }
    }

    async function handleCloseParameters() {
        setSelectedStep(null)

        try {
            const workflowResponse = await fetch(url + `/workflow/${workflowId}`)
            const updatedWorkflow = await workflowResponse.json()
            setWorkflow(updatedWorkflow)
        } catch (error) {
            console.error("Chyba při obnovení workflow:", error)
        }
    }

    if (error) {
        return (
            <div className="p-8">
                <p className="text-red-600">{error}</p>
            </div>
        )
    }

    return (
        <div className="min-h-screen flex">
            <WorkflowSidebar
                searchQuery={searchQuery}
                setSearchQuery={setSearchQuery}
                filteredOperations={filteredOperations}
                allCategoryTags={allCategoryTags}
                selectedCategoryTags={selectedCategoryTags}
                toggleCategoryTag={toggleCategoryTag}
                clearCategoryTags={clearCategoryTags}
                workflow={workflow}
                operations={operations}
                onAddStep={handleAddStep}
                onDeleteStep={handleDeleteStep}
                onSelectStep={setSelectedStep}
                selectedStep={selectedStep}
                onExecuteStep={handleExecuteStep}
                runningStepId={runningStepId}
            />

            <WorkflowContent
                workflow={workflow}
                selectedStep={selectedStep}
                operations={operations}
                workflowId={workflowId}
                onCloseParameters={handleCloseParameters}
                apiBaseUrl={url}
            />
        </div>
    )
}

export default WorkflowLayout