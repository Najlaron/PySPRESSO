import { VscError } from "react-icons/vsc"
import { HiOutlineExclamationTriangle } from "react-icons/hi2"

// vizualizace erroru při načítání workflow
function WorkflowError({ errorType }) {
    let title = ""
    let message = ""


    if (errorType === "not-found") {
        title = "The workflow does not exist"
        message = "Check to make sure you've entered the correct workflow ID."
    }
    else {
        title = "The workflow cannot be loaded"
        message = "Cannot reach the backend. Try refreshing the page or restarting the backend."
    }

    return (
        <div className="py-ds-md px-ds-lg rounded border border-red-200 bg-red-50 text-red-800 flex items-center gap-ds-lg">
            <div className="shrink-0 flex items-center justify-center">
                <HiOutlineExclamationTriangle size="2rem" />
            </div>
            <hgroup>
                <h3 className="font-medium text-2xl">{title}</h3>
                <p className="text-xl">{message}</p>
            </hgroup>
        </div>
    )
}

export default WorkflowError