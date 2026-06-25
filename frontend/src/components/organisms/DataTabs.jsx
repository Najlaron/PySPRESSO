import TabDataButton from "../molecules/WorkflowLayout/TabDataButton"

function DataTabs({ visualizations = [], activeView, setActiveView, hasData }) {
    const dataTabLabels = [
        "Data",
        "Metadata",
        "Variables Metadata",
        "Batch Info",
        "Candidates Features",
    ]

    return (
        <div className="flex gap-ds-md items-center flex-wrap">
            {visualizations.map((visualization) => (
                <TabDataButton
                    key={visualization.stepId}
                    label={`Plot ${visualization.stepNumber}`}
                    active={
                        activeView?.type === "visualization" &&
                        activeView?.stepId === visualization.stepId
                    }
                    onClick={() =>
                        setActiveView({
                            type: "visualization",
                            stepId: visualization.stepId,
                        })
                    }
                />
            ))}

            {hasData &&
                dataTabLabels.map((tabName) => (
                    <TabDataButton
                        key={tabName}
                        label={tabName}
                        active={
                            activeView?.type === "data" &&
                            activeView?.tabName === tabName
                        }
                        onClick={() =>
                            setActiveView({
                                type: "data",
                                tabName,
                            })
                        }
                    />
                ))}
        </div>
    )
}

export default DataTabs