import { useState } from "react"
import ParameterInput from "../molecules/ParameterInput"


function ParametersForm({ step, operation }) {
    // Initialize parameter values from step or empty
    const [parameterValues, setParameterValues] = useState(
        step?.parameters || {}
    )

    const handleParameterChange = (paramName, value) => {
        setParameterValues((prev) => ({
            ...prev,
            [paramName]: value,
        }))
    }

    const handleSubmit = (e) => {
        e.preventDefault()
        console.log("Parameters to save:", parameterValues)

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
                                        value={parameterValues[param.name] || ""}
                                        onChange={(value) =>
                                            handleParameterChange(param.name, value)
                                        }
                                    />
                                    {param.description && (
                                        <p className="text-sm text-noir/60 mt-ds-sm">
                                            {param.description}
                                        </p>
                                    )}
                                </div>
                            ))}
                            <div className="text-base text-noir">* Required parameter</div>

                            <button
                                type="submit"
                                className="bg-espresso text-foam px-ds-lg py-ds-md rounded-lg font-semibold cursor-pointer shadow-md text-xl"
                            >
                                Submit
                            </button>
                        </>
                    ) : (
                        <p className="text-noir/60">
                            This operation has no parameters to configure.
                        </p>
                    )}
                </form>
            </div>
        </main>
    )
}

export default ParametersForm