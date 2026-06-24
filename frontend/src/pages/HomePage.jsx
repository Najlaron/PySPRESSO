import FeatureCard from '../components/molecules/HomePage/FeatureCard'
import Step from '../components/molecules/Step'
import HowToCite from '../components/molecules/HomePage/HowToCite'
import Footer from '../components/organisms/Layouts/Footer'
import dataIconSrc from "../../media/icons/data-management-icon.svg"
import logoLightSrc from "../../media/logo-light.png"
import { Link } from 'react-router-dom'

function DataIcon({ className }) {
    return <img src={dataIconSrc} alt="data-management-icon" className={className} />
}

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
                                Create Workflow
                            </Link>
                        </hgroup>
                    </div>

                    <img src={logoLightSrc} alt="application-logo" className="w-sm" />
                </div>
            </header>

            <main className="bg-foam">
                {/* Q: Šířka containeru? */}
                <section id="about-section" className="container flex flex-col items-center">
                    <h2 className="text-4xl font-bold text-noir text-center mb-ds-lg">About</h2>
                    <p className="text-center text-grounds text-xl mb-ds-xl max-w-225">PySPRESSO is an open-source, Python-based pipeline designed to simplify and standardize the processing of multi-batch LC-MS data. It provides an all-in-one, modular framework that allows users to customize data filtering, correction, visualization, and statistical analysis workflows.</p>

                    <div className="flex-section">
                        <FeatureCard
                            title="Data Management"
                            description="Organize, version, and share your datasets securely within your research team. Organize, version, and share your datasets securely within your research team."
                            Icon={DataIcon}
                        />
                        <FeatureCard
                            title="Data Management"
                            description="Organize, version, and share your datasets securely within your research team. Organize, version, and share your datasets securely within your research team."
                            Icon={DataIcon}
                        />
                        <FeatureCard
                            title="Data Management"
                            description="Organize, version, and share your datasets securely within your research team. Organize, version, and share your datasets securely within your research team."
                            Icon={DataIcon}
                        />
                    </div>

                </section>

                <section id="how-to-use-section" className="container">
                    <h2 className="text-4xl font-bold text-noir text-center mb-ds-xl">How to use</h2>

                    <div className="flex flex-col gap-ds-lg items-center">
                        <Step title="Upload Your Data"
                            description="Import your datasets in various formats including CSV, Excel, JSON, or connect directly to your database. Our system automatically detects data types and suggests appropriate preprocessing steps."
                            number={1} />
                        <Step title="Configure Your Pipeline"
                            description="Use our visual pipeline builder to design your analysis workflow. Select from a library of preprocessing, transformation, and analysis modules, or create custom components."
                            number={2} />
                        <Step title="Execute Analysis"
                            description="Run your configured pipeline with real-time progress monitoring. View intermediate results, debug any issues, and iterate on your analysis in a controlled environment."
                            number={3} />
                        <Step title="Export Results"
                            description="Generate comprehensive reports with visualizations, statistical summaries, and reproducibility information. Export in multiple formats suitable for publication or further processing."
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
