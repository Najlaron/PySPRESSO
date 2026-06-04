import { BrowserRouter, Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import CreateWorkflow from './pages/CreateWorkflow'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/create-workflow" element={<CreateWorkflow />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
