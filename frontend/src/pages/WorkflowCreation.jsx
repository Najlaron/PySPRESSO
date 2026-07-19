import StepBadge from '../components/molecules/StepBadge'
import WorkflowForm from '../components/organisms/WorkflowCreationForm'
import { useReducer, useState, useEffect } from 'react'

const initialFilesState = {
    data: { file: null, error: null },
    batchInfo: { file: null, error: null },
    importFile: { file: null, error: null },
}

function filesReducer(state, action) {
    switch (action.type) {
        case 'SET_FILE':
            return {
                ...state,
                [action.key]: { ...state[action.key], file: action.file, error: null },
            }
        case 'SET_ERROR':
            return {
                ...state,
                [action.key]: { ...state[action.key], error: action.error },
            }
        case 'CLEAR':
            return initialFilesState
        default:
            return state
    }
}

function WorkflowCreation() {
    const [filesState, filesDispatch] = useReducer(filesReducer, initialFilesState)
    const [loadError, setLoadError] = useState(null)

    return (
        <div className="bg-foam min-h-screen flex flex-col gap-ds-xl">
            <h1 className="text-4xl font-semibold text-center text-noir mt-ds-xl">Workflow Initialization</h1>
            <WorkflowForm filesState={filesState} filesDispatch={filesDispatch} loadError={loadError} setLoadError={setLoadError} />
        </div>
    )
}


export default WorkflowCreation