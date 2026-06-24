import TabDataButton from "../molecules/WorkflowLayout/TabDataButton"

function DataTabs({ setActiveTab, activeTab }) {
    const tabLabels = ["Data", "Metadata", "Variables Metadata", "Batch Info", "Candidates Features"]

    return (
        <div className="flex gap-ds-md items-center">
            {tabLabels.map((temp) => {
                return (
                    <TabDataButton key={temp} label={temp} active={activeTab === temp} onClick={() => setActiveTab(temp)} />
                )
            })}
        </div>
    )


}

export default DataTabs