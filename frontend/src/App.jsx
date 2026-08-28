import { BrowserRouter, Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import WorkflowSetup from './pages/WorkflowSetup'
import WorkflowCreation from './pages/WorkflowCreation'
import WorkflowLayout from './pages/WorkflowLayout'
import WorkflowsList from './pages/WorkflowsList'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/create-workflow" element={<WorkflowSetup />} />
        <Route path="/new-workflow" element={<WorkflowCreation />} />
        <Route path="/workflow/:workflowId" element={<WorkflowLayout />} />
        <Route path="/saved-workflows" element={<WorkflowsList />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
