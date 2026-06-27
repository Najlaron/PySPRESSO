import { useState } from "react"
import SearchBar from "../../molecules/WorkflowLayout/SearchBar"
import SearchedMethod from "../../molecules/WorkflowLayout/SearchedMethod"
import WorkflowStepCard from "../../molecules/WorkflowLayout/WorkflowStepCard"
import { CapitalizeFirstLetter } from "../../../utils/helpers"

function WorkflowSidebar({
    searchQuery,
    setSearchQuery,
    filteredOperations,
    allCategoryTags = [],
    selectedCategoryTags = [],
    toggleCategoryTag,
    clearCategoryTags,
    workflow,
    operations,
    onAddStep,
    onDeleteStep,
    onSelectStep,
    selectedStep,
    onExecuteStep,
    runningStepId,
}) {
    const [showTagFilters, setShowTagFilters] = useState(false)

    const hasOperationFilter =
        searchQuery.trim().length > 0 || selectedCategoryTags.length > 0

    function formatTag(tag) {
        return String(tag)
            .replaceAll("_", " ")
            .replace(/\b\w/g, (letter) => letter.toUpperCase())
    }

    return (
        <aside className="w-[30%] bg-light-foam pt-ds-xl px-ds-lg">
            <div>
                <SearchBar
                    searchQuery={searchQuery}
                    setSearchQuery={setSearchQuery}
                />

                <div className="mb-ds-md">
                    <button
                        type="button"
                        onClick={() => setShowTagFilters((current) => !current)}
                        className="text-sm text-espresso underline"
                    >
                        {showTagFilters ? "Hide category tags" : "Show category tags"}
                    </button>

                    {showTagFilters && (
                        <div className="mt-ds-sm flex flex-col gap-ds-sm">
                            {selectedCategoryTags.length > 0 && (
                                <div>
                                    <button
                                        type="button"
                                        onClick={clearCategoryTags}
                                        className="text-xs underline text-espresso/70"
                                    >
                                        Clear selected tags
                                    </button>
                                </div>
                            )}

                            <div className="flex flex-wrap gap-ds-xs">
                                {allCategoryTags.map((tag) => {
                                    const isSelected = selectedCategoryTags.includes(tag)

                                    return (
                                        <button
                                            key={tag}
                                            type="button"
                                            onClick={() => toggleCategoryTag(tag)}
                                            className={
                                                isSelected
                                                    ? "px-ds-sm py-1 rounded-full bg-crema text-noir text-xs font-semibold border border-espresso/40"
                                                    : "px-ds-sm py-1 rounded-full bg-white text-espresso text-xs border border-roast/30"
                                            }
                                        >
                                            {formatTag(tag)}
                                        </button>
                                    )
                                })}
                            </div>
                        </div>
                    )}
                </div>

                {hasOperationFilter && filteredOperations.length > 0 && (
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

                {hasOperationFilter && filteredOperations.length === 0 && (
                    <div className="mb-4 p-ds-md rounded bg-foam shadow-sm">
                        <p className="text-espresso font-medium">
                            No methods were found
                        </p>
                    </div>
                )}
            </div>

            <div className="mt-ds-xl flex flex-col gap-ds-md mb-ds-xl">
                <h3 className="text-xl text-espresso font-semibold">
                    METHODS IN WORKFLOW
                </h3>

                {workflow?.definition?.steps?.map((step, idx) => {
                    const op = operations.find(
                        (operation) => operation.id === step.operation_id
                    )

                    return (
                        <WorkflowStepCard
                            key={step.step_id}
                            stepNumber={idx + 1}
                            title={op?.label}
                            category={CapitalizeFirstLetter(
                                op?.categoryTags?.join(", ")
                            )}
                            deleteHandle={() => onDeleteStep(step?.step_id)}
                            onSelectStep={() => onSelectStep(step)}
                            isSelected={selectedStep?.step_id === step.step_id}
                            step={step}
                            operation={op}
                            executeHandle={() => onExecuteStep(step.step_id)}
                            isRunning={runningStepId === step.step_id}
                        />
                    )
                })}
            </div>
        </aside>
    )
}

export default WorkflowSidebar