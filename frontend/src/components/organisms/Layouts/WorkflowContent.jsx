import ParametersForm from "../ParametersForm"
import { useState } from "react"
import WorkflowVisualization from "../WorkflowVisualization"
import DataTabs from "../DataTabs"
import DataFrame from "../DataFrame"

function WorkflowContent({ workflow, selectedStep, operations, workflowId, onCloseParameters }) {
    const [activeTab, setActiveTab] = useState(null)
    const [showJson, setShowJson] = useState(false)

    function formatWorkflowForDisplay(wf) {
        if (!wf) return wf
        const out = { ...wf }
        if (out.state && typeof out.state === "object") {
            const s = { ...out.state }
            const largeFields = [
                "data",
                "variable_metadata",
                "metadata",
                "batch_info",
                "pca_df",
                "pca_loadings",
                "fold_change",
                "plsda_metadata",
                "candidates",
            ]
            largeFields.forEach((k) => {
                if (k in s && s[k] != null) s[k] = "nastaveno"
            })
            out.state = s
        }
        return out
    }

    if (selectedStep) {
        const operation = operations.find(op => op.id === selectedStep.operation_id)
        return (
            <ParametersForm
                step={selectedStep}
                operation={operation}
                workflowId={workflowId}
                onClose={onCloseParameters}
            />
        )
    }

    function getDataFrameForTab(tabName) {
        if (!workflow || !workflow.state) return null

        switch (tabName) {
            case "Data":
                return workflow.state.data
            case "Metadata":
                return workflow.state.metadata
            case "Variables Metadata":
                return workflow.state.variable_metadata
            case "Batch Info":
                return workflow.state.batch_info
            case "Candidates Features":
                return workflow.state.candidates
            default:
                return null
        }
    }

    return (
        // Tohle celé je ta pravá část layoutu
        <main className="flex-1 pt-ds-lg! px-ds-xl bg-foam gap-0!">
            {/* 1. Nadpis */}
            <h1 className="text-4xl font-bold mb-ds-xl">{workflow?.workflow_name}</h1>

            <div className="flex flex-col gap-ds-lg">
                {/* 2. Tohle je ta obrazovka s napísem, že máš spustit tu vizualizaci (v ní se potom bude zobrazovat ta tabulka) */}
                <WorkflowVisualization />

                {/* 3. Tohle jsou ty tlačítka (data, metadata,...) */}
                <DataTabs
                    setActiveTab={setActiveTab}
                    activeTab={activeTab}
                />

                {/* 4. Tohle je potom tabulka, která zobrazuje ty data */}
                {activeTab && workflow?.state && (
                    <DataFrame
                        data={getDataFrameForTab(activeTab)}
                    />
                )}


                {/* 5. Tohle je ten JSON */}
                <div className="mt-ds-lg">
                    <button className="bg-crema p-ds-md text-noir font-semibold"
                        onClick={() => setShowJson(!showJson)}
                    >
                        {showJson ? "Disable JSON" : "Show JSON"}
                    </button>

                    {showJson && (
                        <pre className="bg-white p-4 rounded border overflow-auto mt-ds-md">
                            {JSON.stringify(formatWorkflowForDisplay(workflow), null, 2)}
                        </pre>
                    )}

                </div>


            </div>

        </main>
    )
}

export default WorkflowContent
