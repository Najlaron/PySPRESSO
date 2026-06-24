import { useState, useRef, useEffect } from "react"
import { MdExpandMore, MdPlayArrow } from "react-icons/md"
import { GiTestTubes } from "react-icons/gi"
import { ImLab } from "react-icons/im"
import { PiSliders } from "react-icons/pi"
import { HiOutlineDotsHorizontal } from "react-icons/hi"
import { RiDeleteBinLine } from "react-icons/ri"


function WorkflowStepCard({ stepNumber, title, category, deleteHandle, onSelectStep, isSelected }) {
    const [menuOpen, setMenuOpen] = useState(false)
    const menuRef = useRef(null)

    // po kliknutí jinam se menu zabalí
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                setMenuOpen(false)
            }
        }

        document.addEventListener('click', handleClickOutside)
        return () => document.removeEventListener('click', handleClickOutside)
    }, [])

    return (
        <div className={`flex w-full box-border overflow-hidden rounded-xl bg-foam border-2 shadow-sm transition duration-200 ${isSelected ? "border-espresso" : "border-roast/20"}`}>

            {/* číslo - levá strana */}
            <div className="flex w-1/10 items-center justify-center bg-espresso text-xl font-semibold text-foam">
                {stepNumber}
            </div>

            {/* pravá strana */}
            <div className="flex flex-1 flex-col gap-ds-md min-w-0 w-full">
                {/* ikona + název, typ metody */}
                <div className="flex items-start justify-between mt-ds-md px-ds-md w-full">
                    <div className="flex items-center gap-ds-md min-w-0">
                        <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-crema/25 shrink-0">
                            <ImLab className="h-6 w-6 text-espresso" />
                        </div>

                        <div className="min-w-0">
                            <h2 className="text-base font-medium text-noir truncate">
                                {title}
                            </h2>
                            <p className="mt-1 text-sm text-noir/60 truncate">{category}</p>
                        </div>
                    </div>

                    <div className="relative ml-ds-md" ref={menuRef}>
                        <button
                            onClick={() => setMenuOpen((s) => !s)}
                            aria-haspopup="true"
                            aria-expanded={menuOpen}
                            className="cursor-pointer shrink-0"
                        >
                            <HiOutlineDotsHorizontal size="1.5rem" color="341100" />
                        </button>

                        {menuOpen && (
                            <div className="absolute right-0 w-42 bg-foam border border-roast/20 rounded shadow-md z-20 p-1">
                                <button
                                    onClick={() => {
                                        setMenuOpen(false)
                                        deleteHandle()
                                    }}
                                    className="w-full p-ds-sm hover:bg-crema/65 flex gap-ds-sm items-center rounded cursor-pointer"
                                >
                                    <RiDeleteBinLine size="1.25rem" color="341100" /> {/*musí se zarovnat */}
                                    <span className="text-base text-noir">Delete method</span>
                                </button>
                            </div>
                        )}
                    </div>
                </div>

                {/* tlačítka */}
                <div className="flex items-center gap-ds-md px-ds-md mb-ds-md flex-wrap w-full">
                    <button className="cursor-pointer">
                        <MdExpandMore size="1.5rem" color="341100" />
                    </button>

                    <button className="cursor-pointer flex items-center gap-2 px-ds-md py-1 text-espresso border-2 border-transparent hover:border-espresso hover:text-noir rounded-lg transition" onClick={onSelectStep}>
                        <PiSliders size="1.5rem" color="713105" />
                        <span className="text-base">Parameters</span>
                    </button>

                    <button className="cursor-pointer flex items-center gap-2 px-ds-md py-1  text-espresso rounded-lg transition border-2 border-crema hover:bg-crema">
                        <MdPlayArrow size="1.5rem" />
                        <span className="text-base">Run</span>
                    </button>
                </div>


            </div>


        </div>
    );
}

export default WorkflowStepCard