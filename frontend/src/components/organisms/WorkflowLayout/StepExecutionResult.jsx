// vizualizace výsledky vykonání metody
function StepExecutionResult({ stepExecutionMessage }) {
    // výsledek vykonání
    const executionStatus = stepExecutionMessage?.status
    // statusy pro selhání vykonání metody
    const failedStatus = ["failed", "warning", "blocked", "needs_parameters"]

    // barva rámečku podle výsledky vykonání
    const borderClass =
        executionStatus === "done"
            ? "border-l-[#6D8B74]"
            : failedStatus.includes(executionStatus)
                ? "border-l-[#ED9C4C]"
                : "border-transparent"

    return (
        <div className="">
            <div className={`bg-light-foam p-ds-md rounded-lg border-l-6 overflow-auto max-h-64 shadow-xl ${borderClass}`}>
                {stepExecutionMessage ? (
                    <div className="flex items-start gap-ds-md">
                        <div>
                            <div className="font-semibold text-noir mb-2 text-lg">{stepExecutionMessage.operation}</div>
                            <ul className="">
                                {(stepExecutionMessage.message || []).map((msg, idx) => {
                                    const text = msg?.message ?? msg
                                    return (
                                        <li key={idx} className="text-noir mb-1 text-lg">{text}</li>
                                    )
                                })}
                            </ul>
                        </div>
                    </div>
                ) : (
                    <p className="text-noir text-lg">No output message available.</p>
                )}
            </div>
        </div>
    )
}

export default StepExecutionResult
