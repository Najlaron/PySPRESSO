import ParametersForm from "../ParametersForm"
import { useState, useEffect, useMemo } from "react"
import WorkflowVisualization from "../WorkflowVisualization"
import DataTabs from "../DataTabs"
import DataFrame from "../DataFrame"
import VisualizationTabs from "../VisualizationTabs"
import WorkflowError from "../WorkflowError"
import { formatNetworkError } from "../../../utils/helpers"

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


function WorkflowContent({ workflow,
    selectedStep,
    operations,
    workflowId,
    onCloseParameters,
    isLoading,
    apiBaseUrl,
    error,
    workflowError,
    stepExecutionMessage
}) {
    //const [activeTab, setActiveTab] = useState(null)
    const [activeView, setActiveView] = useState(null)
    const [showJson, setShowJson] = useState(false)
    const [showLog, setShowLog] = useState(false)
    const [isExporting, setIsExporting] = useState(false)
    const [exportError, setExportError] = useState(null)

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

    // const logMessages = useMemo(() => {
    //     if (!workflow) return []

    //     const steps = workflow.definition?.steps || []
    //     return steps
    //         .map((step) => {
    //             const operation = operations.find((op) => op.id === step.operation_id)
    //             const name = operation?.label || step.operation_id
    //             const summary = step?.output_summary || {}

    //             const status = summary.status || summary.state || null
    //             const err = summary.error || summary.exception || summary.traceback || summary.message

    //             if (err) {
    //                 return `Operation ${name} failed: ${String(err)}`
    //             }

    //             if (status && String(status).toLowerCase && ["done", "success", "completed", "ok"].includes(String(status).toLowerCase())) {
    //                 return `Operation ${name} was completed.`
    //             }

    //             // fallback: if there's any saved_paths or figure_paths, assume success
    //             if (summary.saved_paths || summary.figure_path || summary.plots || summary.figures) {
    //                 return `Operation ${name} was completed.`
    //             }

    //             return null
    //         })
    //         .filter(Boolean)
    // }, [workflow, operations])


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

    async function handleExportWorkflow() {
        setIsExporting(true)
        try {
            const response = await fetch(`${apiBaseUrl}/workflow/${workflowId}/export`)

            if (!response.ok) {
                setExportError("Failed to export workflow.")
                return
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `workflow_${workflowId}.json`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
        } catch (err) {
            setExportError(formatNetworkError(err))
        } finally {
            setIsExporting(false)
        }
    }

    // status vykonaného kroku
    const execucitonStatus = stepExecutionMessage?.status


    return (
        // Tohle celé je ta pravá část layoutu
        <main className="flex-1 pt-ds-lg! px-ds-xl bg-foam gap-0!">
            {workflowError && (
                <WorkflowError
                    errorType={workflowError}
                />
            )}

            <div className="flex justify-between items-center">
                {/* 1. Nadpis */}
                <h1 className="text-4xl font-bold mb-ds-xl">{workflow?.workflow_name}</h1>



                {error && !workflowError && (
                    <div className="p-ds-lg rounded border border-red-200 bg-red-50 text-red-800 text-xl">
                        {error}
                    </div>
                )}

                {exportError && !workflowError && (
                    <div className="p-ds-lg rounded border border-red-200 bg-red-50 text-red-800 text-xl">
                        {exportError}
                    </div>
                )}
            </div>

            {isLoading
                ? <div className="flex justify-center flex-col items-center gap-ds-sm">
                    <div className="loader"></div>
                    <h2 className="text-noir/70 text-2xl font-semibold">Loading workflow</h2>
                </div>
                : <div className="flex flex-col gap-ds-lg">

                    {visualizations.length > 0 && (
                        <VisualizationTabs
                            visualizations={visualizations}
                            activeView={activeView}
                            setActiveView={setActiveView}
                        />
                    )}

                    {activeView?.type === "data" ? (
                        <DataFrame data={selectedDataFrame} />
                    ) : (
                        <WorkflowVisualization
                            visualization={selectedVisualization}
                            apiBaseUrl={apiBaseUrl}
                        />
                    )}

                    <div className="flex justify-between">
                        {workflow?.state?.data && (
                            <DataTabs
                                visualizations={visualizations}
                                activeView={activeView}
                                setActiveView={setActiveView}
                            />
                        )}
                        {workflow && (
                            <button
                                onClick={handleExportWorkflow}
                                disabled={isExporting}
                                className={`bg-espresso text-foam px-ds-md py-ds-md font-semibold transition hover:bg-noir/90
                                    ${isExporting ? "" : "cursor-pointer"}
                                    `}
                            >
                                {isExporting ? "Exporting" : "Export Workflow"}
                            </button>
                        )}
                    </div>

                    {/* Execution log button and panel */}
                    {/* {workflow && (
                        <div className="mt-ds-md">
                            <div className="bg-white p-4 rounded border overflow-auto mt-ds-md max-h-64">
                                {logMessages.length === 0 ? (
                                    <p className="text-noir/60">No execution log available.</p>
                                ) : (
                                    <ul className="list-disc pl-5">
                                        {logMessages.map((msg, idx) => (
                                            <li key={idx} className="text-noir mb-1">{msg}</li>
                                        ))}
                                    </ul>
                                )}
                            </div>
                        </div>
                    )} */}

                    {workflow && (
                        <div className="mt-ds-md">
                            <div className={`bg-light-foam p-ds-md rounded-lg border-l-6 overflow-auto mt-ds-md max-h-64 shadow-xl
                                ${execucitonStatus === "done" ? "border-l-[#6D8B74]" : execucitonStatus === "failed" ? "border-l-[#ED9C4C]" : "border-transparent "}
                                `}>
                                {stepExecutionMessage ? (
                                    <div className="flex items-start gap-ds-md">
                                        <div>
                                            <div className="font-semibold text-noir mb-2">{stepExecutionMessage.operation}</div>
                                            <ul className="">
                                                {stepExecutionMessage.message.map((msg, idx) => {
                                                    const text = msg?.message ?? msg
                                                    return (
                                                        <li key={idx} className="text-noir mb-1">{text}</li>
                                                    )
                                                })}
                                            </ul>
                                        </div>
                                    </div>
                                ) : (
                                    <p className="text-noir">No output message available.</p>
                                )}
                            </div>
                        </div>
                    )}

                    {/* 4. Tohle jsou ty tlačítka (data, metadata,...) */}
                    {/* {workflow?.state.data && (
                        <DataTabs
                            setActiveTab={setActiveTab}
                            activeTab={activeTab}
                        />
                    )} */}

                    {/* 5. Tohle je ten JSON */}
                    {workflow && (
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
                    )}

                </div>

            }
        </main >
    )
}

export default WorkflowContent
