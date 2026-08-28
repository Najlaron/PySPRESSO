import { useState } from "react"
import { FaRegQuestionCircle } from "react-icons/fa"
import Tooltip from "./Tooltip"

// vizualizace jednoho parametru metody
function ParameterInput({ parameter, value, onChange }) {
    const paramType = parameter.type

    function parseNumericValue(rawValue, type) {
        if (rawValue === "") {
            return undefined
        }

        if (type === "float") {
            const normalized = rawValue.replace(",", ".")
            return parseFloat(normalized)
        }

        return parseInt(rawValue, 10)
    }

    // všechny zápisy datových typů parametrů
    const TYPE_LABELS = {
        str: "String",
        string: "String",
        integer: "Integer",
        int: "Integer",
        number: "Integer",
        boolean: "Boolean",
        bool: "Boolean",
        float: "Float",
        list: "List",
        list_or_bool: "List or Boolean",
        bool_or_int: "Boolean or Integer",
        list_or_none: "List or None",
        str_or_none: "String or None",
        list_or_str: "String or List",
        str_or_list: "String or List",
    }

    function getTypeLabel(paramType) {
        return (TYPE_LABELS[paramType] || "Unknown")
    }

    // zobrazení parametru boolean
    if (paramType === "boolean" || paramType === "bool") {
        return (
            <div className="flex items-center gap-ds-md">
                <input
                    type="checkbox"
                    checked={value === true || value === "true"}
                    onChange={(e) => onChange(e.target.checked)}
                    className="w-6 h-6 rounded cursor-pointer accent-espresso shrink-0"
                />
                <label className="flex justify-center gap-ds-sm text-noir font-medium text-xl">
                    <div className="flex items-center gap-ds-sm">
                        {parameter.label}
                        <Tooltip
                            text={parameter.help}
                        >
                            <FaRegQuestionCircle size="1.2rem" color="341100" className="shrink-0" />
                        </Tooltip>
                    </div>
                </label>
            </div>
        )
    }

    // zobrazení čísel (int, float)
    if (paramType === "integer" || paramType === "number" || paramType === "int" || paramType === "float") {
        return (
            <div className="flex flex-col gap-ds-sm">
                <label className="flex flex-col justify-center text-noir font-medium text-xl">
                    <div className="flex items-center gap-ds-sm">
                        {parameter.required ? (
                            <span>*</span>
                        ) : (
                            ""
                        )}
                        {parameter.label}
                        <Tooltip
                            text={parameter.help}
                        >
                            <FaRegQuestionCircle size="1.2rem" color="341100" className="shrink-0" />
                        </Tooltip>
                    </div>
                    <span className="text-base text-espresso/70">{getTypeLabel(parameter.type)}</span>
                </label>
                {/* `Enter ${parameter.label.toLowerCase()}` */}
                <input
                    type="number"
                    value={value}
                    onChange={(e) => {
                        onChange(parseNumericValue(e.target.value, paramType))
                    }}
                    placeholder={parameter.example}
                    className="px-ds-md py-ds-md border border-roast/50 rounded-lg bg-light-foam text-noir focus:outline-none focus:border-espresso"
                    step={paramType === "float" ? "0.01" : "1"}
                />
            </div>
        )
    }

    // ostatní typy
    return (
        <div className="flex flex-col gap-ds-sm">
            <label className="flex justify-center flex-col text-noir font-medium text-xl">
                <div className="flex items-center gap-ds-sm">
                    {parameter.required ? (
                        <span>*</span>
                    ) : (
                        ""
                    )}
                    {parameter.label}
                    <Tooltip
                        text={parameter.help}
                    >
                        <FaRegQuestionCircle size="1.2rem" color="341100" className="shrink-0" />
                    </Tooltip>
                </div>
                <span className="text-base text-espresso/70">{getTypeLabel(parameter.type)}</span>
            </label>
            <input
                type="text"
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={parameter.example}
                className="px-ds-md py-ds-md border border-roast/50 rounded-lg bg-light-foam text-noir focus:outline-none focus:border-espresso"
            />
        </div>
    )
}

export default ParameterInput
