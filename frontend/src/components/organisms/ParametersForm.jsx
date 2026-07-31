import { useState, useEffect } from "react"
import ParameterInput from "../molecules/WorkflowLayout/ParameterInput"
import { formatNetworkError } from "../../utils/helpers"
import { API_BASE_URL } from "../../config"

function ParametersForm({ step, operation, workflowId, onClose, reorderPromiseParams }) {
    const [error, setError] = useState("")
    const [isAdding, setIsAdding] = useState(false)


    function initializeParams() {
        const params = {}
        if (operation?.parameterSchema) {
            operation.parameterSchema.forEach(param => {
                if (step?.params && step.params[param.name] !== undefined) {
                    params[param.name] = step.params[param.name]
                } else if (param.default !== undefined) {
                    params[param.name] = param.default
                } else {
                    params[param.name] = ""
                }
            })
        }
        return params
    }

    const [parameterValues, setParameterValues] = useState(
        initializeParams()
    )

    useEffect(() => {
        setParameterValues(initializeParams())
    }, [operation, step])

    const handleParameterChange = (paramName, value) => {
        setParameterValues((prev) => ({
            ...prev,
            [paramName]: value,
        }))
        setError("")
    }

    async function handleSubmit(e) {
        e.preventDefault()
        setError("")
        setIsAdding(true)

        try {
            await reorderPromiseParams

            const response = await fetch(
                API_BASE_URL + `/workflow/${workflowId}/step/${step.step_id}/parameters`,
                {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ parameters: parameterValues }),
                }
            )

            const data = await response.json()

            if (!response.ok) {
                setError(data?.message ?? "Failed to save parameters")
                return
            }

            onClose()
        } catch (err) {
            setError(formatNetworkError(err))
        } finally {
            setIsAdding(false)
        }
    }

    if (!operation) {
        return (
            <div className="p-ds-xl">
                <p className="text-noir/60">Operation not found</p>
            </div>
        )
    }

    const parameters = operation.parameterSchema || []

    return (
        <main className="flex-1 pt-ds-lg px-ds-xl bg-foam overflow-y-auto">
            <hgroup>
                <h1 className="text-4xl font-bold text-noir mb-ds-sm">
                    {operation.label}
                </h1>
                <p className="text-espresso">PARAMETERS SETTING</p>
            </hgroup>

            <div className="max-w-lg">
                <form onSubmit={handleSubmit} className="space-y-ds-lg">
                    {parameters.length > 0 ? (
                        <>
                            {parameters.map((param) => (
                                <div
                                    key={param.name}
                                    className=""
                                >
                                    <ParameterInput
                                        parameter={param}
                                        value={parameterValues[param.name] !== undefined ? parameterValues[param.name] : ""}
                                        onChange={(value) =>
                                            handleParameterChange(param.name, value)
                                        }
                                    />
                                </div>
                            ))}
                            <div className="text-base text-noir">* Required parameter</div>

                            <button
                                type="submit"
                                className={`bg-espresso hover:bg-roast disabled:bg-roast/50 text-foam px-ds-lg py-ds-md rounded-lg font-semibold shadow-md text-xl ${isAdding ? "" : "cursor-pointer"}`}
                                disabled={isAdding}
                            >
                                {isAdding ? "Submitting" : "Submit"}
                            </button>
                        </>
                    ) : (
                        <p className="text-noir/60">
                            This operation has no parameters to configure.
                        </p>
                    )}
                </form>
            </div>
            {error && (
                <p className="text-red-700 text-xl text-medium">{error}</p>
            )}

        </main>
    )
}

export default ParametersForm