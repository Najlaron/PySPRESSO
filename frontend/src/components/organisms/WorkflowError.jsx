import spiledCoffeeSrc from "../../../media/spilled_coffee.png"

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
        <div className="p-ds-md rounded border border-red-200 bg-red-50 text-red-800">
            <hgroup>
                <h3 className="font-medium text-2xl">{title}</h3>
                <p className="text-xl">{message}</p>
            </hgroup>
        </div>
    )
}

export default WorkflowError