import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import WorkflowCard from "../components/molecules/WorkflowCard"


const url = "http://127.0.0.1:5000"

function WorkflowsList() {
    const navigate = useNavigate()
    const [workflows, setWorkflows] = useState(null)
    const [selectedWorkflow, setSelectedWorkflow] = useState(null)


    useEffect(() => {
        async function loadWorkflows() {
            try {
                const response = await fetch(url + "/workflows")
                const data = await response.json()

                if (!response.ok) {
                    console.log("Máš tam chybu")
                    return
                }

                setWorkflows(data)
            } catch (error) {
                console.log("Máš tam chybu")
            }
        }

        loadWorkflows()
    }, [])



    return (
        <div className="p-8 bg-foam min-h-screen">
            <h1 className="text-4xl font-bold mb-8 text-center">Choose Workflow</h1>

            <div className="flex flex-col gap-ds-md items-center">
                <div className="mb-8 flex flex-col gap-ds-md">
                    {workflows?.map((workflow) => (
                        <WorkflowCard
                            key={workflow.id}
                            workflow={workflow}
                            isSelected={workflow.id === selectedWorkflow}
                            onClick={() => setSelectedWorkflow(workflow.id === selectedWorkflow ? null : workflow.id)} // možnost odkliknutí
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
        </div>
    )
}

export default WorkflowsList