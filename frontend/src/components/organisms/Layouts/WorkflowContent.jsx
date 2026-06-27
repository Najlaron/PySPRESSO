import ParametersForm from "../ParametersForm"
import { useEffect, useMemo, useState } from "react"
import WorkflowVisualization from "../WorkflowVisualization"
import DataTabs from "../DataTabs"
import DataFrame from "../DataFrame"

const IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".svg"]

function isImagePath(path) {
    return IMAGE_EXTENSIONS.some((ext) =>
        String(path).toLowerCase().endsWith(ext)
    )
}

function collectImagePaths(value) {
    if (!value) return []

    if (typeof value === "string") {
        return isImagePath(value) ? [value] : []
    }

    if (Array.isArray(value)) {
        return value.flatMap((item) => collectImagePaths(item))
    }

    if (typeof value === "object") {
        return Object.values(value).flatMap((item) => collectImagePaths(item))
    }

    return []
}

function getImagePathsFromStep(step) {
    const summary = step?.output_summary || {}

    const imagePaths = [
        ...collectImagePaths(summary.saved_paths),
        ...collectImagePaths(summary.figure_path),
        ...collectImagePaths(summary.before_plots),
        ...collectImagePaths(summary.after_plots),
        ...collectImagePaths(summary.s_exploration_images),
        ...collectImagePaths(summary.s_exploration_paths),
        ...collectImagePaths(summary.plots),
        ...collectImagePaths(summary.plot_paths),
        ...collectImagePaths(summary.figures),
        ...collectImagePaths(summary.figure_paths),
    ]

    if (summary.figure_base_path) {
        imagePaths.push(`${summary.figure_base_path}.png`)
    }

    return [...new Set(imagePaths)]
}

function getVisualizationSteps(workflow, operations) {
    const steps = workflow?.definition?.steps || []

    return steps
        .map((step, index) => {
            const imagePaths = getImagePathsFromStep(step)

            if (imagePaths.length === 0) return null

            const operation = operations.find(
                (op) => op.id === step.operation_id
            )

            return {
                stepId: step.step_id,
                operationId: step.operation_id,
                stepNumber: index + 1,
                title: operation?.label || step.operation_id || "Visualization",
                imagePaths,
            }
        })
        .filter(Boolean)
}

function WorkflowContent({
    workflow,
    selectedStep,
    operations,
    workflowId,
    onCloseParameters,
    apiBaseUrl,
}) {
    const [activeView, setActiveView] = useState(null)
    const [showJson, setShowJson] = useState(false)

    const visualizations = useMemo(() => {
        return getVisualizationSteps(workflow, operations)
    }, [workflow, operations])

    useEffect(() => {
        if (!workflow) return

        if (!activeView && visualizations.length > 0) {
            const latestVisualization = visualizations[visualizations.length - 1]
            setActiveView({
                type: "visualization",
                stepId: latestVisualization.stepId,
            })
            return
        }

        if (activeView?.type === "visualization") {
            const selectedStillExists = visualizations.some(
                (visualization) => visualization.stepId === activeView.stepId
            )

            if (!selectedStillExists && visualizations.length > 0) {
                const latestVisualization = visualizations[visualizations.length - 1]
                setActiveView({
                    type: "visualization",
                    stepId: latestVisualization.stepId,
                })
            }

            if (!selectedStillExists && visualizations.length === 0) {
                setActiveView(null)
            }
        }
    }, [workflow, visualizations, activeView?.type, activeView?.stepId])

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
        const operation = operations.find(
            (op) => op.id === selectedStep.operation_id
        )

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

    const selectedVisualization =
        activeView?.type === "visualization"
            ? visualizations.find(
                  (visualization) => visualization.stepId === activeView.stepId
              )
            : null

    const selectedDataFrame =
        activeView?.type === "data"
            ? getDataFrameForTab(activeView.tabName)
            : null

    return (
        <main className="flex-1 pt-ds-lg! px-ds-xl bg-foam gap-0!">
            <h1 className="text-4xl font-bold mb-ds-xl">
                {workflow?.workflow_name}
            </h1>

            <div className="flex flex-col gap-ds-lg">
                {activeView?.type === "data" ? (
                    <DataFrame data={selectedDataFrame} />
                ) : (
                    <WorkflowVisualization
                        visualization={selectedVisualization}
                        apiBaseUrl={apiBaseUrl}
                    />
                )}

                {(workflow?.state?.data || visualizations.length > 0) && (
                    <DataTabs
                        visualizations={visualizations}
                        activeView={activeView}
                        setActiveView={setActiveView}
                        hasData={Boolean(workflow?.state?.data)}
                    />
                )}

                <div className="mt-ds-lg">
                    <button
                        className="bg-crema p-ds-md text-noir font-semibold"
                        onClick={() => setShowJson(!showJson)}
                    >
                        {showJson ? "Disable JSON" : "Show JSON"}
                    </button>

                    {showJson && (
                        <pre className="bg-white p-4 rounded border overflow-auto mt-ds-md">
                            {JSON.stringify(
                                formatWorkflowForDisplay(workflow),
                                null,
                                2
                            )}
                        </pre>
                    )}
                </div>
            </div>
        </main>
    )
}

export default WorkflowContent