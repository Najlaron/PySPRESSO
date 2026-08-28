import { useEffect, useState } from "react"
import ErrorAlert from "./ErrorAlert"

// modal okno pro zadání popisu workflow
function WorkflowDescriptionModal({
    isOpen,
    currentDescription,
    isSaving,
    error,
    onClose,
    onSave,
    onDismissError,
}) {
    const [descriptionValue, setDescriptionValue] = useState("")

    useEffect(() => {
        if (!isOpen) return
        setDescriptionValue(currentDescription || "")
    }, [isOpen, currentDescription])

    if (!isOpen) {
        return null
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-noir/40 px-ds-md">
            <div className="w-full max-w-2xl rounded-lg bg-white p-ds-lg shadow-xl">
                <hgroup>
                    <h2 className="text-2xl font-bold text-noir">Workflow Description</h2>
                    <p className="text-noir/70">Describe the purpose, assumptions, or notes for this workflow.</p>
                </hgroup>

                <textarea
                    value={descriptionValue}
                    onChange={(e) => setDescriptionValue(e.target.value)}
                    rows={8}
                    disabled={isSaving}
                    className="mt-ds-md w-full rounded border border-noir/20 bg-white p-ds-md outline-none focus:border-espresso"
                    placeholder="Add workflow description..."
                />

                {error && (
                    <div className="text-red-600 text-lg">
                        {error}
                    </div>
                )}

                <div className="mt-ds-md flex justify-end gap-ds-sm">
                    <button
                        onClick={onClose}
                        disabled={isSaving}
                        className={`rounded px-ds-md py-ds-sm font-medium transition hover:bg-crema ${isSaving ? "" : "cursor-pointer"}`}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={() => onSave(descriptionValue)}
                        disabled={isSaving}
                        className={`rounded bg-espresso px-ds-md py-ds-sm font-medium text-foam transition hover:bg-noir ${isSaving ? "" : "cursor-pointer"}`}
                    >
                        {isSaving ? "Saving..." : "Save"}
                    </button>
                </div>
            </div>
        </div>
    )
}

export default WorkflowDescriptionModal
