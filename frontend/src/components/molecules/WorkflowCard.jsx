function WorkflowCard({ workflow, isSelected, onClick }) {
    return (
        <div className={`shadow-md p-ds-lg rounded-xl w-150 transition duration-300 hover:shadow-xl hover:-translate-y-2 cursor-pointer ${isSelected ? "bg-espresso" : "bg-light-foam"}`} onClick={onClick}>
            <h2 className={`mb-ds-md font-semibold text-2xl ${isSelected ? "text-foam" : "text-noir"}`}>
                {workflow?.workflow_name}
            </h2>
            <p className={`mb-ds-lg ${isSelected ? "text-foam" : "text-noir"}`}>
                Cross-validated cubic spline correction and normalization of multi-batch LC–MS data with outlier detection and PCA overview.
            </p>

            <p className={`text-espresso ${isSelected ? "text-foam" : "text-espresso"}`}>
                <span className="font-medium">Created</span> {new Date(workflow?.created_at).toLocaleDateString("cs-CZ")}
            </p>
            <p className={`text-espresso ${isSelected ? "text-foam" : "text-espresso"}`}>
                <span className="font-medium">Last modify</span> {new Date(workflow?.updated_at).toLocaleDateString("cs-CZ")}
            </p>
        </div >
    )
}

export default WorkflowCard