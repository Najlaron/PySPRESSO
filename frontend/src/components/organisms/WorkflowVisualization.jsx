import { BsGraphUp } from "react-icons/bs";


function WorkflowVisualization() {
    return (
        <div className="w-full flex items-center justify-center flex-col gap-ds-md h-200 shadow-lg bg-light-foam rounded-xl">
            <div className="flex h-14 w-14 items-center justify-center rounded-lg bg-crema/25 shrink-0">
                <BsGraphUp className="h-7 w-7 text-espresso" />
            </div>
            <span className="w-md text-center text-lg text-espresso/75">Run a visualization method method to generate visualization. The result will appear here.</span>
        </div >
    )
}

export default WorkflowVisualization