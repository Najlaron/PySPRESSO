import logoLightTextSrc from "../../../media/logo-light-text.png"

function Footer() {
    return (
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
    );
}

export default Footer
