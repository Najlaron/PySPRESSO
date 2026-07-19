import { HiOutlineDotsHorizontal } from "react-icons/hi"
import { RiDeleteBinLine } from "react-icons/ri"
import { useState, useRef, useEffect } from "react"


function WorkflowCard({ workflow, isSelected, onClick, onDelete, isDeleting }) {
    const [menuOpen, setMenuOpen] = useState(false)
    const [isExpanded, setIsExpanded] = useState(false)
    const containerRef = useRef(null)

    // po kliknutí jinam se menu zabalí
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (containerRef.current && !containerRef.current.contains(event.target)) {
                setMenuOpen(false)
            }
        }

        document.addEventListener('click', handleClickOutside)
        return () => document.removeEventListener('click', handleClickOutside)
    }, [])

    // zavře menu, pokud se klikne na tři tečky v jiné kartě
    useEffect(() => {
        const onOtherOpen = (event) => {
            try {
                const otherId = event?.detail?.id
                if (!otherId) return
                if (otherId !== workflow?.id) setMenuOpen(false)
            } catch (e) {
                // ignore
            }
        }

        document.addEventListener('workflow-menu-open', onOtherOpen)
        return () => document.removeEventListener('workflow-menu-open', onOtherOpen)
    }, [workflow?.id])

    return (
        <div ref={containerRef} className={`relative shadow-md p-ds-lg rounded-xl w-150 transition duration-300 hover:shadow-xl hover:-translate-y-2 cursor-pointer ${isSelected ? "bg-espresso" : "bg-light-foam"}`} onClick={onClick}>

            <div className="flex items-center justify-between mb-ds-md ">
                <h2 className={`font-semibold text-2xl ${isSelected ? "text-foam" : "text-noir"}`}>
                    {workflow?.workflow_name}
                </h2>
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        setMenuOpen((s) => {
                            const next = !s
                            if (next) {
                                try {
                                    document.dispatchEvent(new CustomEvent('workflow-menu-open', { detail: { id: workflow?.id } }))
                                } catch (err) {
                                    // ignore
                                }
                            }
                            return next
                        })
                    }}
                    aria-haspopup="true"
                    aria-expanded={menuOpen}
                    className="cursor-pointer shrink-0"
                >
                    <HiOutlineDotsHorizontal size="1.5rem" color="341100" />
                </button>

                {menuOpen && (
                    <div className="absolute right-10 top-15 w-42 bg-foam border border-roast/20 rounded shadow-md z-20 p-1">
                        <button
                            onClick={async (e) => { e.stopPropagation(); await onDelete(); setMenuOpen(false); }}
                            className={`w-full p-ds-sm hover:bg-crema/65 flex gap-ds-sm items-center rounded
                                                    ${isDeleting ? "cursor-wait bg-crema/65" : "cursor-pointer"}`}
                        >
                            <RiDeleteBinLine size="1.25rem" color="341100" /> {/*musí se zarovnat */}
                            <span className="text-base text-noir">Delete method</span>
                        </button>
                    </div>
                )}
            </div>

            <p className={`mb-ds-lg ${isSelected ? "text-foam" : "text-noir"}`}>
                Cross-validated cubic spline correction and normalization of multi-batch LC–MS data with outlier detection and PCA overview.
            </p>

            <p className={`text-espresso ${isSelected ? "text-foam" : "text-espresso"}`}>
                <span className="font-medium">Created</span> {new Date(workflow?.created_at).toLocaleDateString("cs-CZ")}
            </p>
            <p className={`text-espresso ${isSelected ? "text-foam" : "text-espresso"}`}>
                <span className="font-medium">Last modify</span> {new Date(workflow?.updated_at).toLocaleDateString("cs-CZ")}
            </p>
        </div >
    )
}

export default WorkflowCard