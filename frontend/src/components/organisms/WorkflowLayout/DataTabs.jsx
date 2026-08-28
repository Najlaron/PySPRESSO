import TabDataButton from "../../molecules/WorkflowLayout/TabDataButton"

// tlačítka pro zobrazení tabulkových dat
function DataTabs({ visualizations = [], activeView, setActiveView }) {
    const tabLabels = ["Data", "Metadata", "Variables Metadata", "Batch Info", "Candidates Features"]

    return (
        <div className="flex gap-ds-md items-center flex-wrap">

            {tabLabels.map((tabName) => (
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