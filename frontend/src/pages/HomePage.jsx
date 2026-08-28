import FeatureCard from '../components/molecules/HomePage/FeatureCard'
import Step from '../components/molecules/HomePage/Step'
import HowToCite from '../components/molecules/HomePage/HowToCite'
import Footer from '../components/organisms/Layouts/Footer'
import dataIconSrc from "../../media/icons/data-management-icon.svg"
import logoLightSrc from "../../media/logo-light.png"
import { Link } from 'react-router-dom'
import { LuPackagePlus } from "react-icons/lu"
import { LuWorkflow } from "react-icons/lu"
import { LuDatabase } from "react-icons/lu"


// domovská stránka
function HomePage() {
    return (
        <>
            <header>
                <nav className="nav">
                    <ul className="nav-menu">
                        <li><a href="#about-section" className="text-2xl text-foam">About</a></li>
                        <li><a href="#how-to-use-section" className="text-2xl text-foam">How to use</a></li>
                        <li><a href="#how-to-cite-section" className="text-2xl text-foam">How to cite</a></li>
                    </ul>
                </nav>

                <div className="hero-section">
                    <div>
                        <hgroup>
                            <h1 className="text-5xl text-foam font-bold">PYSPRESSO</h1>
                            <p className="text-2xl text-crema max-w-xl">Modular Pipeline for Omics Analysis - build, run, and share reproducible workflows.</p>
                            <Link to="/create-workflow" className="wf-button cursor-pointer transition duration-300 hover:bg-noir">
                                Get Started
                            </Link>
                        </hgroup>
                    </div>

                    <img src={logoLightSrc} alt="application-logo" className="w-sm" />
                </div>
            </header>

            <main className="bg-foam">
                <section id="about-section" className="container flex flex-col items-center">
                    <h2 className="text-4xl font-bold text-noir text-center mb-ds-lg">About</h2>
                    <p className="text-center text-grounds text-xl mb-ds-xl max-w-225">PySPRESSO is an open-source Python framework for reproducible processing of mass spectrometry-based omics data.
                        It provides a modular workflow environment for data filtering, correction, visualization, statistical analysis, and report generation.
                        The graphical interface allows users to build, run, save, and share analysis workflows, while still supporting custom Python operations for advanced use cases.
                    </p>

                    <div className="flex-section">
                        <FeatureCard
                            title="Modular Workflows"
                            description="Build analysis pipelines from individual processing steps and adjust them to different experimental designs. 
                            Each operation can be configured, reordered, and combined into a reproducible workflow."
                            Icon={LuWorkflow}
                            iconSize={64}
                        />
                        <FeatureCard
                            title="Data Analysis"
                            description="Process, correct, visualize, and statistically evaluate MS data in one environment locally on your PC without the need to upload sensitive or large-scale datasets. 
                            PySPRESSO keeps intermediate outputs and generated results connected to the workflow that created them."
                            Icon={LuDatabase}
                            iconSize={64}
                        />
                        <FeatureCard
                            title="Share and Extend"
                            description="Save workflows as reusable templates and share them with collaborators. 
                            Advanced users can also add custom Python operations which are automatically integrated into the system for you to use."
                            Icon={LuPackagePlus}
                            iconSize={64}
                        />
                    </div>

                </section>

                <section id="how-to-use-section" className="container">
                    <h2 className="text-4xl font-bold text-noir text-center mb-ds-xl">How to use</h2>

                    <div className="flex flex-col gap-ds-lg items-center">
                        <Step title="Import Your Data"
                            description="Load peak-picked mass spectrometry data in one of the supported formats. PySPRESSO imports both the feature intensity table and batch information, including sample names, acquisition order, batch assignment, creation times, and optional sample-grouping metadata columns."
                            number={1} />
                        <Step title="Build or Load a Workflow"
                            description="Create a new workflow from available operations or load a saved workflow template. Configure each step using parameters such as filtering thresholds, correction settings, metadata columns for supervised statistics, visualization options, or others."
                            number={2} />
                        <Step title="Run and Inspect"
                            description="Execute the workflow and inspect generated outputs directly in the interface. Intermediate data, diagnostic plots, warnings, statistical results and extensive report file help you follow what happened during the analysis."
                            number={3} />
                        <Step title="Inspect, Export, and Reuse"
                            description="Review generated plots, processed data, outputs, and reports. Save workflows as reusable templates, share them with collaborators, and easily rerun the analysis with adjusted parameters when a specific experiment requires fine-tuning."
                            number={4} />

                    </div>

                </section>

                <section id="how-to-cite-section" className="container flex flex-col items-center gap-ds-lg">
                    <h2 className="text-4xl font-bold text-noir text-center">How to cite</h2>
                    <p className="text-center text-grounds text-xl max-w-195">If you use WorkflowLab in your research, please cite it using the following BibTeX entry. Proper citation helps support continued development and maintenance.</p>

                    <HowToCite />

                </section>

            </main >

            <Footer />
        </>
    );
}

export default HomePage;
