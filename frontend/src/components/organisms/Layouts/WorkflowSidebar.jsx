import SearchBar from "../../molecules/WorkflowLayout/SearchBar"
import SearchedMethod from "../../molecules/WorkflowLayout/SearchedMethod"
import WorkflowStepCard from "../../molecules/WorkflowLayout/WorkflowStepCard"
import TabDataButton from "../../molecules/WorkflowLayout/TabDataButton"
import { MdPlayArrow } from "react-icons/md"
import { DndContext, closestCorners, MouseSensor, useSensor, useSensors } from "@dnd-kit/core"
import { SortableContext, arrayMove, verticalListSortingStrategy } from "@dnd-kit/sortable"
import { restrictToVerticalAxis } from "@dnd-kit/modifiers"

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
    isRunningAll,
    onReorderSteps
}) {
    const isAddingMethod = Boolean(addedOperationId)
    // search bar je zablokovaný pokud se přidává nějaká metody, pokud běží krok nebo pokud se odstranuje krok
    const isMethodActionsLocked = isAddingMethod || isStepRunning || Boolean(runningStepId) || isStepDeleting
    const isSingleStepRunning = isStepRunning || Boolean(runningStepId)

    const sensors = useSensors(
        useSensor(MouseSensor, {
            activationConstraint: {
                distance: 4,
            },
        })
    )

    const handleDragEnd = (event) => {
        const { active, over } = event

        if (!over || active.id === over.id) return

        const oldIndex = workflow?.definition?.steps?.findIndex((step) => step.step_id === active.id) ?? -1
        const newIndex = workflow?.definition?.steps?.findIndex((step) => step.step_id === over.id) ?? -1

        if (oldIndex < 0 || newIndex < 0) return

        const reorderedIds = arrayMove(
            workflow.definition.steps.map((step) => step.step_id),
            oldIndex,
            newIndex
        )

        onReorderSteps?.(reorderedIds)
    }


    return (
        <aside className="w-[30%] h-screen overflow-hidden bg-light-foam pt-ds-xl pb-ds-lg px-ds-lg shadow-[4px_0_12px_rgba(0,0,0,0.08)] z-10">
            <div className="flex flex-col justify-between h-full max-w-360">
                {/* Search bar */}
                <div className="">
                    <SearchBar
                        searchQuery={searchQuery}
                        setSearchQuery={setSearchQuery}
                        isWorkflowLoading={isWorkflowLoading}
                        isDisabled={isMethodActionsLocked}
                    />

                    {/* zobrazí se metody, odpovídající zadanému výrazu */}
                    {searchQuery && filteredOperations.length > 0 && workflow && (
                        <div className="mb-ds-md rounded max-h-64 overflow-y-auto shadow-md">
                            {filteredOperations.map((op) => (
                                <SearchedMethod
                                    key={op.id}
                                    onClick={() => {
                                        if (isMethodActionsLocked) return
                                        onAddStep(op)
                                    }}
                                    operation={op}
                                    addedId={addedOperationId}
                                    disabled={isMethodActionsLocked}
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

                <div className="mt-ds-xl flex-1 min-h-0 overflow-y-auto">
                    <div className="flex flex-col gap-ds-md">
                        <h3 className="text-xl text-espresso font-semibold">METHODS IN WORKFLOW</h3>
                        <DndContext sensors={sensors} collisionDetection={closestCorners} onDragEnd={handleDragEnd} modifiers={[restrictToVerticalAxis]}>
                            <SortableContext items={workflow?.definition?.steps?.map((step) => step.step_id) ?? []} strategy={verticalListSortingStrategy}>
                                {workflow?.definition.steps?.map((step, idx) => {
                                    const op = operations.find(o => o.id === step.operation_id)
                                    return (
                                        <WorkflowStepCard
                                            key={step.step_id}
                                            id={step.step_id}
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
                                            isRunningAll={isRunningAll}
                                            isDeleting={isStepDeleting}
                                        />
                                    )
                                })}
                            </SortableContext>
                        </DndContext>
                    </div>
                </div>

                {/* nejde spustit, pokud už probíhá spouštění nějaké metody, nebo všech metod nebo je zrovna odstraňován nějaký krok */}
                {workflow && (
                    <button onClick={onExecuteAll}
                        disabled={isRunningAll || isSingleStepRunning || isStepDeleting}
                        className={`bg-espresso text-foam rounded-xl shadow-md py-ds-md text-xl font-medium mt-ds-xl
                        flex gap-ds-md items-center justify-center
                        hover:bg-noir/90 transition duration-300 ${isRunningAll || isSingleStepRunning ? "cursor-wait bg-noir/90" : " cursor-pointer"}`}>
                        <MdPlayArrow size="1.5rem" />
                        <span>{isRunningAll ? "Running..." : "Run all methods"}</span>
                    </button>
                )}
            </div>
        </aside>
    )
}

export default WorkflowSidebar
