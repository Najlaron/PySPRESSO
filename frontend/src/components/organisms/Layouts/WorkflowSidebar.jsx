import SearchBar from "../../molecules/WorkflowLayout/SearchBar"
import SearchedMethod from "../../molecules/WorkflowLayout/SearchedMethod"
import WorkflowStepCard from "../../molecules/WorkflowLayout/WorkflowStepCard"
import { CapitalizeFirstLetter } from "../../../utils/helpers"
import TabDataButton from "../../molecules/WorkflowLayout/TabDataButton"

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
    isStepRunning
}) {
    return (
        <aside className="w-[30%] bg-light-foam pt-ds-xl px-ds-lg">
            {/* Search bar */}
            <div className="">
                <SearchBar
                    searchQuery={searchQuery}
                    setSearchQuery={setSearchQuery}
                />

                {/* zobrazí se metody, odpovídající zadanému výrazu */}
                {searchQuery && filteredOperations.length > 0 && (
                    <div className="mb-ds-md rounded max-h-64 overflow-y-auto shadow-md">
                        {filteredOperations.map((op) => (
                            <SearchedMethod
                                key={op.id}
                                onClick={() => {
                                    onAddStep(op)
                                }}
                                operation={op}
                                isSelected={false}
                            />
                        ))}
                    </div>
                )}

                {searchQuery && filteredOperations.length === 0 && (
                    <div className="mb-4 p-ds-md rounded bg-foam shadow-sm">
                        <p className="text-espresso font-medium">No methods were found</p>
                    </div>
                )}
            </div>

            <div className="mt-ds-xl flex flex-col gap-ds-md mb-ds-xl">
                <h3 className="text-xl text-espresso font-semibold">METHODS IN WORKFLOW</h3>
                {workflow?.definition.steps?.map((step, idx) => {
                    const op = operations.find(o => o.id === step.operation_id)
                    return (
                        <WorkflowStepCard
                            key={step.step_id}
                            stepNumber={idx + 1}
                            title={op?.label}
                            category={CapitalizeFirstLetter(op?.categoryTags?.join(", "))}
                            deleteHandle={() => onDeleteStep(step?.step_id)}
                            onSelectStep={() => onSelectStep(step)}
                            isSelected={selectedStep?.step_id === step.step_id}
                            step={step}
                            operation={op}
                            executeHandle={() => onExecuteStep(step.step_id)}
                            isRunning={isStepRunning}
                        />
                    )
                })}
            </div>
        </aside>
    )
}

export default WorkflowSidebar
