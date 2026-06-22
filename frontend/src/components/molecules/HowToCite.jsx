import { act, useState } from "react";
import { useCopyToClipboard } from "usehooks-ts";
import { IoCopyOutline } from "react-icons/io5"

function HowToCite() {
    const [activeTab, setActiveTab] = useState("tab1")

    const tabs = [
        { id: "tab1", label: "BibTex" },
        { id: "tab2", label: "Plain Text" },
        { id: "tab3", label: "RIS" },
        { id: "tab4", label: "Refworks" }
    ];

    const bibtex = `@software{workflowlab2024,
    author = {Research Team},
    title = {WorkflowLab: A Platform for Reproducible Data Analysis},
    year = {2024},
    url = {https://workflowlab.example.com},
    version = {1.0.0}
}`

    const plainText = `Research Team. WorkflowLab: A Platform for Reproducible Data Analysis. 
Version 1.0.0. 2024. 
Available at: https://workflowlab.example.com`

    const RIS = `TY  - COMP
AU  - Research Team
TI  - WorkflowLab: A Platform for Reproducible Data Analysis
PY  - 2024
UR  - https://workflowlab.example.com
ET  - 1.0.0
ER  -
`

    const Refworks = `RT Software
A1 Research Team
T1 WorkflowLab: A Platform for Reproducible Data Analysis
YR 2024
VO 1.0.0
LK https://workflowlab.example.com`


    const tabContent = {
        tab1: (
            <pre className="whitespace-pre-wrap text-espresso p-ds-md">
                {bibtex}
            </pre>
        ),
        tab2: (
            <pre className="whitespace-pre-wrap text-espresso p-ds-md">
                {plainText}
            </pre>
        ),
        tab3: (
            <pre className="whitespace-pre-wrap text-espresso p-ds-md">
                {RIS}
            </pre>
        ),
        tab4: (
            <pre className="whitespace-pre-wrap text-espresso p-ds-md">
                {Refworks}
            </pre>
        ),
    }

    const citationByTab = {
        tab1: bibtex,
        tab2: plainText,
        tab3: RIS,
        tab4: Refworks
    };

    const [, copy] = useCopyToClipboard()

    const handleCopy = () => {
        copy(citationByTab[activeTab] ?? "")
    }

    return (
        <div className="bg-light-foam shadow-md max-w-2xl w-full rounded-xl border border-roast/50 ">
            <div className="grid grid-cols-4 bg-crema/50 rounded-t-xl rounded-tr-xl border-b border-roast/50">
                {tabs.map((tab) => (
                    <button key={tab.id} className={`group py-ds-md cursor-pointer text-noir ${activeTab === tab.id ? "border-b-3 border-noir text-noir font-semibold" : ""
                        }`}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        <span className={`text-blue ${activeTab === tab.id ? "" : "group-hover:border-b-2"}`}>{tab.label}</span>
                    </button>
                ))}
            </div>

            <div className="relative h-56">
                <div className="h-full overflow-y-auto">{tabContent[activeTab]}</div>
                <button type="button" aria-label="Copy citaion" className="cursor-pointer absolute bottom-4 right-4" onClick={handleCopy}><IoCopyOutline size="1.5rem" color="#341100" /></button>
            </div>

        </div>
    )
}

export default HowToCite

