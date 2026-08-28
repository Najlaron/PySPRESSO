import { useState, useEffect } from "react"
import ParameterInput from "../../molecules/WorkflowLayout/ParameterInput"
import { formatNetworkError } from "../../../utils/helpers"
import { API_BASE_URL } from "../../../config"

// formulář pro vyplnění parametrů metody
function ParametersForm({ step, operation, workflowId, onClose, reorderPromiseParams, onSubmitPromiseChange }) {
    const [error, setError] = useState("")
    const [isAdding, setIsAdding] = useState(false)


    function isEmptyInput(rawValue) {
        // kontroluje, jestli je parametr prázdný
        return (
            rawValue === undefined
            || rawValue === null
            || (typeof rawValue === "string" && rawValue.trim() === "")
        )
    }

    function toTrimmedString(rawValue) {
        // odstraní z hodnoty parametry mezery na konci a na začátku
        return String(rawValue ?? "").trim()
    }

    function parseBooleanString(rawValue) {
        // kontroluje, jestli se ve stringu nachází boolean hodnota
        const text = toTrimmedString(rawValue).toLowerCase()

        if (text === "true") {
            return { ok: true, value: true }
        }

        if (text === "false") {
            return { ok: true, value: false }
        }

        return { ok: false, error: "Expected 'true' or 'false'." }
    }

    function parseIntValue(rawValue) {
        // kontroluje, jestli je parametr celé číslo
        if (typeof rawValue === "number" && Number.isInteger(rawValue)) {
            return { ok: true, value: rawValue }
        }

        // v případě, že se jedná o string se z něj číslo snaží vytáhnout
        const text = toTrimmedString(rawValue)
        if (!/^-?\d+$/.test(text)) {
            return { ok: false, error: "Expected an integer value." }
        }

        return { ok: true, value: parseInt(text, 10) }
    }

    function parseFloatValue(rawValue) {
        // kontroluje, jestli je parametr float
        if (typeof rawValue === "number" && Number.isFinite(rawValue)) {
            return { ok: true, value: rawValue }
        }

        // opět pokus o vytažení čísla ze stringu
        const text = toTrimmedString(rawValue).replace(",", ".")
        if (!/^-?\d+(\.\d+)?$/.test(text)) {
            return { ok: false, error: "Expected a numeric value." }
        }

        return { ok: true, value: parseFloat(text) }
    }

    function convertListItem(item) {
        // boolean a číslo vrátí
        if (typeof item === "boolean" || typeof item === "number") {
            return item
        }

        const text = toTrimmedString(item)
        const lowered = text.toLowerCase()

        // z "true"/"false" udělá boolean hodnotu
        if (lowered === "true") {
            return true
        }

        if (lowered === "false") {
            return false
        }

        // pokud je ve stringu číslo, tak ho z něj vytáhne
        if (/^-?\d+$/.test(text)) {
            return parseInt(text, 10)
        }

        if (/^-?\d+[.,]\d+$/.test(text)) {
            return parseFloat(text.replace(",", "."))
        }

        return text
    }

    function parseListValue(rawValue, { convertItems = true } = {}) {
        // u stirngu odstraní mezery a prázdné prvky
        const parseItems = (items) => {
            const filtered = items
                .map((item) => (typeof item === "string" ? item.trim() : item))
                .filter((item) => !(typeof item === "string" && item === ""))

            // převede každý prvek pole na string
            if (!convertItems) {
                return filtered.map((item) => String(item))
            }

            // převede každý prvek na jeho skutečnou hodnotu
            return filtered.map((item) => convertListItem(item))
        }

        // pokud je hodnota parametru pole tak ok
        if (Array.isArray(rawValue)) {
            return { ok: true, value: parseItems(rawValue) }
        }

        const text = toTrimmedString(rawValue)

        if (text === "") {
            return { ok: true, value: [] }
        }

        // pokud je paraketr obalen [] (vstup je zapsán jako pole)
        // tak je převeden na JS pole
        if (text.startsWith("[") && text.endsWith("]")) {
            try {
                const parsed = JSON.parse(text.replace(/'/g, '"'))
                if (Array.isArray(parsed)) {
                    return { ok: true, value: parseItems(parsed) }
                }
            } catch {
            }
        }

        // jinak z parametru udělá pole podle prvku oddělených ,
        const splitItems = text.split(",")
        return { ok: true, value: parseItems(splitItems) }
    }

    function parseStrOrListValue(rawValue) {
        // pokud dostane pole, smaže u prvků mezery a vyfiltruje prázdné prvky
        if (Array.isArray(rawValue)) {
            return {
                ok: true,
                value: rawValue
                    .map((item) => String(item).trim())
                    .filter((item) => item !== ""),
            }
        }

        const text = toTrimmedString(rawValue)

        // jinak parsuje pole, prvky nepřevadí (budou string)
        if (text.startsWith("[") && text.endsWith("]")) {
            return parseListValue(text, { convertItems: false })
        }

        if (text.includes(",")) {
            return parseListValue(text, { convertItems: false })
        }

        return { ok: true, value: text }
    }

    // podle typu parametru kontroluje jeho hodnotu
    function parseByType(rawValue, paramType) {
        switch (paramType) {
            case "str":
                return { ok: true, value: toTrimmedString(rawValue) }

            case "str_or_none": {
                // zkontroluje, jestli není hdonota parametru none a jinak vrací string
                const text = toTrimmedString(rawValue)
                const lowered = text.toLowerCase()
                if (lowered === "none" || lowered === "null" || text === "") {
                    return { ok: true, value: null }
                }
                return { ok: true, value: text }
            }

            case "bool":
                if (typeof rawValue === "boolean") {
                    return { ok: true, value: rawValue }
                }
                return { ok: false, error: "Expected a boolean value." }

            case "int":
                return parseIntValue(rawValue)

            case "float":
                return parseFloatValue(rawValue)

            case "list":
                return parseListValue(rawValue, { convertItems: true })

            case "list_or_none": {
                // nejdřív zkontroluje, jestli se nejdná o none, jinak parsuje list
                const text = toTrimmedString(rawValue).toLowerCase()
                if (text === "" || text === "none" || text === "null") {
                    return { ok: true, value: null }
                }
                return parseListValue(rawValue, { convertItems: true })
            }

            case "str_or_list":
                return parseStrOrListValue(rawValue)

            case "list_or_bool": {
                if (typeof rawValue === "boolean") {
                    return { ok: true, value: rawValue }
                }

                // pokud je hodnota boolean (true, false) tak ok, jinak parsuje list
                const boolParsed = parseBooleanString(rawValue)
                if (boolParsed.ok) {
                    return boolParsed
                }

                return parseListValue(rawValue, { convertItems: true })
            }

            case "bool_or_int": {
                if (typeof rawValue === "boolean") {
                    return { ok: true, value: rawValue }
                }

                // pokud je hodnota boolean (true, false) tak ok, jinak parsuje int
                const boolParsed = parseBooleanString(rawValue)
                if (boolParsed.ok) {
                    return boolParsed
                }

                return parseIntValue(rawValue)
            }

            default:
                return { ok: true, value: rawValue }
        }
    }

    function emptyValueForType(paramType) {
        if (paramType === "list") {
            return []
        }

        if (paramType === "str") {
            return ""
        }

        return null
    }

    // kontroluje jestli nechybí povinný parametr a
    // následně kontroluje a parsuje hodnotu parametru podle jeho typu
    function parseAndValidateParameter(param, rawValue) {
        if (isEmptyInput(rawValue)) {
            if (param.required) {
                return {
                    ok: false,
                    error: "This parameter is required.",
                }
            }

            // inicializace prázdné hodnoty parametru
            return {
                ok: true,
                value: emptyValueForType(param.type),
            }
        }

        return parseByType(rawValue, param.type)
    }

    // kontrola parametrů
    function normalizeParameters(values, schema) {
        const normalized = {}
        const params = schema || []

        for (const param of params) {
            const result = parseAndValidateParameter(param, values[param.name])

            if (!result.ok) {
                return {
                    ok: false,
                    error: `${param.label || param.name}: ${result.error}`,
                }
            }

            normalized[param.name] = result.value
        }

        return {
            ok: true,
            value: normalized,
        }
    }


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

        const submitPromise = (async () => {
            setIsAdding(true)

            try {
                await reorderPromiseParams

                // zkontroluje hodnoty parametrů, pokud jsou ok, tak je přidá
                // jinak vypíše co je špatně
                const normalizedParameters = normalizeParameters(
                    parameterValues,
                    operation?.parameterSchema,
                )

                if (!normalizedParameters.ok) {
                    setError(normalizedParameters.error)
                    return
                }

                const response = await fetch(
                    API_BASE_URL + `/workflow/${workflowId}/step/${step.step_id}/parameters`,
                    {
                        method: "PUT",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ parameters: normalizedParameters.value }),
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
        })()

        onSubmitPromiseChange?.(submitPromise)
        await submitPromise
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

                            {error && (
                                <p className="text-red-700 text-xl text-medium">{error}</p>
                            )}

                            <div className="flex gap-ds-lg">
                                <button
                                    type="submit"
                                    className={`bg-espresso hover:bg-noir/90 transition text-foam px-ds-lg py-ds-md rounded-lg 
                                            font-semibold shadow-md text-xl ${isAdding ? "bg-noir/90" : "cursor-pointer"}`}
                                    disabled={isAdding}
                                >
                                    {isAdding ? "Submitting" : "Submit"}
                                </button>
                                <button
                                    type="button"
                                    className={`hover:bg-crema transition text-noir px-ds-lg py-ds-md rounded-lg 
                                            font-medium text-xl border-2 border-crema cursor-pointer`}
                                    disabled={isAdding}
                                    onClick={() => (onClose())}
                                >
                                    Cancel
                                </button>
                            </div>

                        </>
                    ) : (
                        <div>
                            <p className="text-noir/60 mb-ds-lg">
                                This operation has no parameters to configure.
                            </p>
                            <button
                                type="button"
                                className={`hover:bg-crema transition text-noir px-ds-lg py-ds-md rounded-lg 
                                            font-medium text-xl border-2 border-crema cursor-pointer`}
                                disabled={isAdding}
                                onClick={() => (onClose())}
                            >
                                Cancel
                            </button>
                        </div>

                    )}
                </form>
            </div>


        </main>
    )
}

export default ParametersForm