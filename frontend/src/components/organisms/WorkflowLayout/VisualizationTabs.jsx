import TabDataButton from "../../molecules/WorkflowLayout/TabDataButton"

// tlačítka pro zobrazení vytvořených vizualizací
function VisualizationTabs({ visualizations = [], activeView, setActiveView }) {

    return (
        <div className="flex gap-ds-md items-center overflow-x-auto w-full">
            {visualizations.map((visualization, index) => (
                <TabDataButton
                    key={visualization.stepId}
                    // label={`Plot ${visualization.stepNumber}`}
                    label={`Plot ${index + 1}`}
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
        </div>
    )
}

export default VisualizationTabs
