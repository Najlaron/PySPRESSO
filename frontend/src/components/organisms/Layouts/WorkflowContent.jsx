import ParametersForm from "../ParametersForm"
import { useState, useEffect, useMemo } from "react"
import WorkflowVisualization from "../WorkflowVisualization"
import DataTabs from "../DataTabs"
import DataFrame from "../DataFrame"
import VisualizationTabs from "../VisualizationTabs"
import WorkflowError from "../WorkflowError"
import ErrorAlert from "../../molecules/ErrorAlert"
import StepExecutionResult from "./StepExecutionResult"
import { formatNetworkError } from "../../../utils/helpers"

const IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".svg"]

// kontroluje, jestli cesta končí příponou značící vizualizaci
function isImagePath(path) {
    return IMAGE_EXTENSIONS.some((ext) =>
        String(path).toLowerCase().endsWith(ext)
    )
}

// vezme danou cestu z getImagePathFromStep a zkontroluje, jestli nevede k vizualizaci
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

    // vezme cesty které jsou obsaženy ve výstupu kroku
    // a zkontroluje, jestli vedou k vizualizaci
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

    // set odstraní duplicity
    return [...new Set(imagePaths)]
}

function getVisualizationSteps(workflow, operations) {
    const steps = workflow?.definition?.steps || []

    return steps
        .map((step, index) => {
            // kontroluje, jestli krok vytvořil nějakou vizualizaci
            const imagePaths = getImagePathsFromStep(step)

            if (imagePaths.length === 0) return null

            const operation = operations.find(
                (op) => op.id === step.operation_id
            )

            // objekt poposující vizualizaci
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
    onParametersSubmitPromiseChange,
    onDismissError,
    isLoading,
    apiBaseUrl,
    error,
    workflowError,
    stepExecutionMessage,
    reorderPromise
}) {
    //const [activeTab, setActiveTab] = useState(null)
    const [activeView, setActiveView] = useState(null)
    const [showJson, setShowJson] = useState(false)
    const [showLog, setShowLog] = useState(false)
    const [isExporting, setIsExporting] = useState(false)
    const [exportError, setExportError] = useState(null)
    const [folderError, setFolderError] = useState(null)

    // získává vizualizace z kroků
    const visualizations = useMemo(() => {
        return getVisualizationSteps(workflow, operations)
    }, [workflow, operations])

    useEffect(() => {
        if (!workflow) return

        // pokud existuje nějaká vizualizace ale žádná není aktivní,
        // nastaví se na aktivní poslední vytvořená vizualizace
        if (!activeView && visualizations.length > 0) {
            const latestVisualization = visualizations[visualizations.length - 1]
            setActiveView({
                type: "visualization",
                stepId: latestVisualization.stepId,
            })
            return
        }

        // kontrola, jestli aktivní vizualizace stále existuje,
        // pokud neexistuje (krok byl smazán), vrátí poslední vytvořenou vizaulizaci
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
        const operation = operations.find(op => op.id === selectedStep.operation_id)
        return (
            <ParametersForm
                step={selectedStep}
                operation={operation}
                workflowId={workflowId}
                onClose={onCloseParameters}
                onSubmitPromiseChange={onParametersSubmitPromiseChange}
                reorderPromiseParams={reorderPromise}
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

    async function handleOpenFolder() {
        try {
            const response = await fetch(`${apiBaseUrl}/workflow/${workflowId}/folder`)
            const data = await response.json()

            if (!response.ok) {
                setFolderError("Failed to export workflow.")
                return
            }
        } catch (err) {
            setFolderError(formatNetworkError(err))
        }
    }

    return (
        // Tohle celé je ta pravá část layoutu
        <main className="flex-1 min-h-0 overflow-y-auto pt-ds-lg! px-ds-xl bg-foam gap-0!">
            {workflowError && (
                <WorkflowError
                    errorType={workflowError}
                />
            )}

            <div className="flex justify-between items-center">
                {/* 1. Nadpis */}
                <h1 className="text-4xl font-bold mb-ds-xl">{workflow?.workflow_name}</h1>

                {error && !workflowError && (
                    <ErrorAlert
                        message={error}
                        className="bg-red-50 mb-ds-md"
                        onDismiss={onDismissError}
                    />
                )}

                {exportError && !workflowError && (
                    <ErrorAlert
                        message={exportError}
                        className="bg-red-50 mb-ds-md"
                        onDismiss={() => setExportError(null)}
                    />
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
                        <div className="flex gap-ds-lg">
                            {workflow && (
                                <button
                                    onClick={handleExportWorkflow}
                                    disabled={isExporting}
                                    className={`rounded hover:text-foam px-ds-md py-ds-md font-medium transition hover:bg-espresso
                                    ${isExporting ? "" : "cursor-pointer"}
                                    `}
                                >
                                    {isExporting ? "Exporting" : "Export Workflow"}
                                </button>
                            )}
                            {/* {workflow && (
                                <button
                                    onClick={handleOpenFolder}
                                    disabled={isExporting}
                                    className="p-ds-md font-medium cursor-pointer rounded hover:bg-crema"
                                >
                                    Open folder
                                </button>
                            )} */}
                        </div>
                    </div>


                    {workflow && (
                        <StepExecutionResult stepExecutionMessage={stepExecutionMessage} />
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
