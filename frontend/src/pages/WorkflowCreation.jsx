import StepBadge from '../components/molecules/StepBadge'
import WorkflowForm from '../components/organisms/WorkflowCreationForm'


function WorkflowCreation() {
    return (
        <div className="bg-foam min-h-screen flex flex-col">
            <h1 className="text-4xl font-semibold text-center text-noir mt-ds-xl mb-ds-xl">Workflow Initialization</h1>
            <WorkflowForm />
        </div>
    )
}


export default WorkflowCreation