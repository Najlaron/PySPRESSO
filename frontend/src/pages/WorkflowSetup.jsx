import logoLightSrc from "../../media/logo-light.png"
import Footer from "../components/organisms/Layouts/Footer"
import CreateWfButton from "../components/molecules/WorkflowSetup/CreatingWfButton"
import { Link } from 'react-router-dom'


function WorkflowSetup() {
    return (
        <div className="bg-foam min-h-screen flex flex-col">
            <header>
                <nav className="create-wf-nav">
                    <img src={logoLightSrc} alt="application-logo" className="w-30" />

                    <ul className="nav-menu">
                        <li><a href="/" className="text-2xl text-foam">Home</a></li>
                    </ul>
                </nav>
            </header>

            <main className="container flex flex-col items-center py-ds-xl gap-ds-lg">
                <div>
                    <h1 className="text-4xl text-noir font-bold text-center mb-ds-lg">Workflow Setup</h1>
                    <p className="text-xl text-grounds max-w-2xl text-center">
                        Create a new workflow or continue working with an existing one. Select a previously saved configuration to resume your analysis, or start from scratch and define a new processing pipeline tailored to your data.
                    </p>
                </div>

                <div className="flex gap-ds-xl">
                    <Link to="/saved-workflows" className="flex">
                        <CreateWfButton
                            title={"Choose from existing workflow"}
                            description={"Resume a previously saved workflow."}
                            color={"crema"}
                        />
                    </Link>

                    <Link to="/new-workflow" className="flex">
                        <CreateWfButton
                            title={"Create new workflow"}
                            description={"Start a new analysis pipeline from scratch."}
                            color={"espresso"}
                        />
                    </Link>

                </div>
            </main>

            <Footer />
        </div >
    );
}

export default WorkflowSetup
