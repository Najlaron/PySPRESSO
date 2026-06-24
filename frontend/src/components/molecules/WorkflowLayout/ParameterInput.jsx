import { useState } from "react"
import { FaRegQuestionCircle } from "react-icons/fa"
import Tooltip from "../Tooltip"


function ParameterInput({ parameter, value, onChange }) {
    const paramType = parameter.type

    const TYPE_LABELS = {
        str: "String",
        string: "String",
        integer: "Integer",
        int: "Integer",
        number: "Integer",
        boolean: "Boolean",
        bool: "Boolean",
        float: "Float"
    }

    function getTypeLabel(paramType) {
        return (TYPE_LABELS[paramType] || "Unknown")
    }

    // nepovinný parametr -> defaultní value v inputu bude default hodnota parametru
    // povinný parametr -> zobrazovat example (bude doděláno) ale pouze jako placeholder

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

    // asi může být stejné pro všechny čísla (int, float, ....)
    if (paramType === "integer" || paramType === "number" || paramType === "int") {
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
                    onChange={(e) => onChange(e.target.value)}
                    placeholder={parameter.default}
                    className="px-ds-md py-ds-md border border-roast/50 rounded-lg bg-light-foam text-noir focus:outline-none focus:border-espresso"
                    step={paramType === "integer" ? "1" : "0.01"}
                />
            </div>
        )
    }

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
