import { useState, useEffect } from "react"
import { useParams, useNavigate } from "react-router-dom"
import WorkflowSidebar from "../components/organisms/Layouts/WorkflowSidebar"
import WorkflowContent from "../components/organisms/Layouts/WorkflowContent"

const url = "http://127.0.0.1:5000"

function WorkflowLayout() {
    const { workflowId } = useParams()
    const navigate = useNavigate()
    const [workflow, setWorkflow] = useState(null)
    const [error, setError] = useState("")
    const [operations, setOperations] = useState([])
    const [searchQuery, setSearchQuery] = useState("")
    const [loading, setLoading] = useState(false)
    const [selectedStep, setSelectedStep] = useState(null)
    const [isRunning, setIsRunning] = useState(false)

    useEffect(() => {
        async function loadWorkflow() {
            try {
                const response = await fetch(url + `/workflow/${workflowId}`)
                const data = await response.json()

                if (!response.ok) {
                    setError(data.message || "Workflow se nepodařilo načíst.") // rozhodnout se pro nějaký obecný error zobrazení
                    return
                }

                setWorkflow(data)
            } catch (error) {
                setError("Nastala chyba při komunikaci se serverem.")
            }
        }

        // pro vyhledávání se načtou všechny operace
        async function loadOperations() {
            try {
                const response = await fetch(url + "/operations")
                const data = await response.json()
                setOperations(data)
            } catch (error) {
                console.error("Chyba při načítání operací:", error)
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



    async function handleAddStep(operation) {
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
                alert("Chyba: " + (data.message || "Nepodařilo se přidat krok"))
                return
            }

            // musí se aktulizovat workflow
            const workflowResponse = await fetch(url + `/workflow/${workflowId}`)
            const updatedWorkflow = await workflowResponse.json()
            setWorkflow(updatedWorkflow)

            setSearchQuery("")
        } catch (error) {
            alert("Chyba: " + error.message)
        }
    }

    async function handleExecuteStep(stepId) {
        setIsRunning(true) // krok běží

        try {

            const response = await fetch(url + `/workflow/${workflowId}/step/${stepId}/run`, {
                method: "POST",
            })

            const dataResponse = await response.json()

            if (!response.ok) {
                console.log("Chyba: " + (dataResponse.message || "Nepodařilo se spustit krok"))
                return
            }

            // musí se aktulizovat workflow
            const workflowResponse = await fetch(url + `/workflow/${workflowId}`)
            const updatedWorkflow = await workflowResponse.json()

            setWorkflow(updatedWorkflow)

        } catch (error) {
            alert("Chyba: " + error.message)
        } finally {
            setIsRunning(false) // krok doběhl
        }
    }

    async function handleDeleteStep(stepId) {
        try {
            const response = await fetch(url + `/workflow/${workflowId}/delete_step/${stepId}`, {
                method: "DELETE"
            })

            const data = await response.json()

            if (!response.ok) {
                alert("Chyba: " + (data.message || "Nepodařilo se přidat krok"))
                return
            }

            // musí se aktulizovat workflow
            const workflowResponse = await fetch(url + `/workflow/${workflowId}`)
            const updatedWorkflow = await workflowResponse.json()
            setWorkflow(updatedWorkflow)
            setSelectedStep(null)

        } catch (error) {
            alert("Chyba: " + error.message)
        }
    }

    async function handleCloseParameters() {
        // Zavře form a obnoví workflow data
        setSelectedStep(null)

        try {
            const workflowResponse = await fetch(url + `/workflow/${workflowId}`)
            const updatedWorkflow = await workflowResponse.json()
            setWorkflow(updatedWorkflow)
        } catch (error) {
            console.error("Chyba při obnovení workflow:", error)
        }
    }


    // if (error) {
    //     return (
    //         <div className="p-8">
    //             <p className="text-red-600">{error}</p>
    //             <button onClick={() => navigate(-1)} className="mt-4 text-blue-600 underline">
    //                 Zpět
    //             </button>
    //         </div>
    //     )
    // }

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
                isStepRunning={isRunning}
            />
            <WorkflowContent
                workflow={workflow}
                selectedStep={selectedStep}
                operations={operations}
                workflowId={workflowId}
                onCloseParameters={handleCloseParameters}
            />
        </div>
    )
}

export default WorkflowLayout
