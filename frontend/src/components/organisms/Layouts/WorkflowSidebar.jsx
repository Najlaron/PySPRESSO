import SearchBar from "../../molecules/WorkflowLayout/SearchBar"
import SearchedMethod from "../../molecules/WorkflowLayout/SearchedMethod"
import WorkflowStepCard from "../../molecules/WorkflowLayout/WorkflowStepCard"
import TabDataButton from "../../molecules/WorkflowLayout/TabDataButton"
import { MdPlayArrow } from "react-icons/md"

function WorkflowSidebar({
    searchQuery,
    setSearchQuery,
    filteredOperations,
    workflow,
    operations,
    onAddStep,
    onDeleteStep,
    onSelectStep,
    selectedStep,
    onExecuteStep,
    onExecuteAll,
    isStepRunning,
    runningStepId,
    isWorkflowLoading,
    isStepDeleting,
    addedOperationId,
    isRunningAll
}) {


    return (
        <aside className="w-[30%] bg-light-foam pt-ds-xl pb-ds-lg px-ds-lg">
            <div className="flex flex-col justify-between h-full max-w-150">
                {/* Search bar */}
                <div className="">
                    <SearchBar
                        searchQuery={searchQuery}
                        setSearchQuery={setSearchQuery}
                        isWorkflowLoading={isWorkflowLoading}
                    />

                    {/* zobrazí se metody, odpovídající zadanému výrazu */}
                    {searchQuery && filteredOperations.length > 0 && workflow && (
                        <div className="mb-ds-md rounded max-h-64 overflow-y-auto shadow-md">
                            {filteredOperations.map((op) => (
                                <SearchedMethod
                                    key={op.id}
                                    onClick={() => {
                                        onAddStep(op)
                                    }}
                                    operation={op}
                                    addedId={addedOperationId}
                                />
                            ))}
                        </div>
                    )}

                    {searchQuery && filteredOperations.length === 0 && workflow && (
                        <div className="mb-4 p-ds-md rounded bg-foam shadow-sm">
                            <p className="text-espresso font-medium">No methods were found</p>
                        </div>
                    )}
                </div>

                <div className="mt-ds-xl flex-1 overflow-y-auto">
                    <div className="flex flex-col gap-ds-md">
                        <h3 className="text-xl text-espresso font-semibold">METHODS IN WORKFLOW</h3>
                        {workflow?.definition.steps?.map((step, idx) => {
                            const op = operations.find(o => o.id === step.operation_id)
                            return (
                                <WorkflowStepCard
                                    key={step.step_id}
                                    step={step}
                                    operation={op}
                                    stepNumber={idx + 1}
                                    isSelected={selectedStep?.step_id === step.step_id}
                                    handlers={{
                                        onDelete: () => onDeleteStep(step?.step_id),
                                        onSelectStep: () => onSelectStep(step),
                                        onExecute: () => onExecuteStep(step.step_id)
                                    }}
                                    runningStepId={runningStepId}
                                    isDeleting={isStepDeleting}
                                />
                            )
                        })}
                    </div>
                </div>
                {workflow && (
                    <button onClick={onExecuteAll}
                        disabled={isRunningAll}
                        className={`bg-espresso text-foam rounded-xl shadow-md py-ds-md text-xl font-medium mt-ds-xl
                        flex gap-ds-md items-center justify-center
                        hover:bg-noir/90 transition ${isRunningAll ? "cursor-wait bg-noir/90" : " cursor-pointer"}`}>
                        <MdPlayArrow size="1.5rem" />
                        <span>{isRunningAll ? "Running..." : "Run all methods"}</span>
                    </button>
                )}
            </div>
        </aside>
    )
}

export default WorkflowSidebar
