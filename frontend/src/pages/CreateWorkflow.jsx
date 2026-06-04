import logoLightTextSrc from "../../media/logo-light-text.png"
import logoLightSrc from "../../media/logo-light.png"
import CreateWfButton from "../components/CreatingWfButton"


function CreateWorkflow() {
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

            <main className="container flex-1 flex flex-col items-center py-ds-xl gap-ds-lg">
                <div>
                    <h1 className="text-4xl text-noir font-bold text-center mb-ds-lg">Workflow Setup</h1>
                    <p className="text-xl text-grounds max-w-2xl text-center">
                        Create a new workflow or continue working with an existing one. Select a previously saved configuration to resume your analysis, or start from scratch and define a new processing pipeline tailored to your data.
                    </p>
                </div>

                <div className="flex gap-ds-xl">
                    <CreateWfButton
                        title={"Choose from existing workflow"}
                        description={"Resume a previously saved workflow."}
                        color={"crema"}
                    />
                    <CreateWfButton
                        title={"Create new workflow"}
                        description={"Start a new analysis pipeline from scratch."}
                        color={"espresso"}
                    />
                </div>
            </main>

            {/* Q: Udělat z footeru komponentu, když se používá na dvou místech? */}
            <footer className="bg-grounds">
                <div className="flex justify-between container pt-ds-xl pb-ds-xl">
                    <div className="flex flex-col items-center gap-ds-md">
                        <img src={logoLightTextSrc} alt="application-logo" className="w-52" />
                        <p className="text-foam">Open-source LC–MS data analysis pipeline.</p>
                    </div>

                    <div>
                        <h5 className="text-crema font-semibold text-xl mb-ds-md">Resources</h5>
                        <ul className="flex flex-col gap-ds-md">
                            <li><a href="#" className="text-foam font-light">GitHub</a></li>
                            <li><a href="#" className="text-foam font-light">Documentation</a></li>
                            <li><a href="#" className="text-foam font-light">Contact</a></li>
                        </ul>
                    </div>

                    <div>
                        <div>
                            <h5 className="text-crema font-semibold text-xl mb-ds-md">Developed at</h5>
                            <ul>
                                <li className="text-foam font-light">Institute of Molecular and Translational Medicine</li>
                                <li className="text-foam font-light">Faculty of Medicine and Dentistry</li>
                                <li className="text-foam font-light">Palacký University Olomouc</li>
                            </ul>
                        </div>

                        <div>
                            <h5 className="text-crema font-semibold text-xl mb-ds-md mt-ds-lg">Funding</h5>
                            <ul>
                                <li className="text-foam font-light">National Institute for Cancer Research (EXCELES)</li>
                                <li className="text-foam font-light">Funded by the European Union - Next Generation EU</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div className="container border-t-2 border-roast/40">
                    <p className="text-foam/70 text-center py-ds-lg">&#xA9; Institute Of Moleculal And Translational Medicine</p>
                </div>
            </footer>
        </div >
    );
}

export default CreateWorkflow;
