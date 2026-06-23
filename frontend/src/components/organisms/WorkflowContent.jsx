import ParametersForm from "./ParametersForm"

function WorkflowContent({ workflow, selectedStep, operations }) {
    if (selectedStep) {
        const operation = operations.find(op => op.id === selectedStep.operation_id)
        return (
            <ParametersForm
                step={selectedStep}
                operation={operation}
            />
        )
    }

    return (
        <main className="flex-1 pt-ds-lg! px-ds-xl bg-foam gap-0!">
            <h1 className="text-4xl font-bold mb-ds-xl">{workflow?.workflow_name}</h1>

            <pre className="bg-white p-4 rounded border overflow-auto">
                {JSON.stringify(workflow, null, 2)}
            </pre>
        </main>
    )
}

export default WorkflowContent
