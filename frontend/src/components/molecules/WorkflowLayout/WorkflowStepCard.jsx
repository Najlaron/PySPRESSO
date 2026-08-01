import { useState, useRef, useEffect } from "react"
import { MdExpandMore, MdExpandLess, MdPlayArrow } from "react-icons/md"
import { GiTestTubes } from "react-icons/gi"
import { ImLab } from "react-icons/im"
import { PiSliders } from "react-icons/pi"
import { HiOutlineDotsHorizontal } from "react-icons/hi"
import { RiDeleteBinLine } from "react-icons/ri"
import { CapitalizeFirstLetter } from "../../../utils/helpers"
import { MdDone } from "react-icons/md"
import { useSortable } from "@dnd-kit/sortable"
import { CSS } from "@dnd-kit/utilities"
import { LuClock3 } from "react-icons/lu"



function WorkflowStepCard({
    id,
    step,
    operation,
    stepNumber,
    isSelected,
    handlers: { onDelete, onSelectStep, onExecute },
    runningStepId,
    isRunningAll,
    isDeleting
}) {
    const [menuOpen, setMenuOpen] = useState(false)
    const [isExpanded, setIsExpanded] = useState(false)
    const menuRef = useRef(null)
    const alreadyRun = step.status === "done"
    const isCurrentStepRunning = runningStepId === step.step_id
    const isAnyStepRunning = Boolean(runningStepId)
    const isCorrectionCategory = operation?.categoryTags?.some((tag) => String(tag).toLowerCase() === "correction")
    const takeLong = isCorrectionCategory || operation?.id === "visualizer_violin_plots"

    // drag and drop část
    const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id })
    const style = {
        transform: transform ? CSS.Translate.toString(transform) : undefined,
        transition,
    }

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

    useEffect(() => {
        if (isAnyStepRunning) {
            setMenuOpen(false)
        }
    }, [isAnyStepRunning])

    //const takeLong = operation.categoryTags.

    return (
        <div ref={setNodeRef} style={style}
            className={`flex w-full box-border overflow-hidden rounded-xl bg-foam shadow-md transition duration-200
            ${isSelected ? "border-2 border-espresso" : "border-2 border-espresso/25"} 
            ${alreadyRun ? "opacity-70" : ""}`}

        >
            {/* číslo - levá strana */}
            <div {...attributes} {...listeners} className="cursor-grab flex w-1/10 max-w-15 items-center justify-center bg-espresso text-xl font-semibold text-foam">
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
                            <div className="flex items-center gap-2 min-w-0">
                                <h2 className="text-base font-medium text-noir truncate">
                                    {operation?.label}
                                </h2>
                                {takeLong && (
                                    <LuClock3 className="h-4 w-4 shrink-0 text-espresso" title="This operation may take longer than usual." />
                                )}
                            </div>
                            <p className="mt-1 text-sm text-noir/60 truncate">
                                {CapitalizeFirstLetter(operation?.categoryTags?.join(", "))}
                            </p>
                        </div>
                    </div>

                    <div className="relative ml-ds-md" ref={menuRef}>
                        <button
                            onClick={() => {
                                if (isAnyStepRunning) return
                                setMenuOpen((s) => !s)
                            }}
                            aria-haspopup="true"
                            aria-expanded={menuOpen}
                            disabled={isAnyStepRunning}
                            className={`shrink-0 ${isAnyStepRunning ? "opacity-60" : "cursor-pointer"}`}
                        >
                            <HiOutlineDotsHorizontal size="1.5rem" color="341100" />
                        </button>

                        {menuOpen && (
                            <div className="absolute right-0 w-42 bg-foam border border-roast/20 rounded shadow-md z-20 p-1">
                                <button
                                    onClick={async () => {
                                        if (isAnyStepRunning) return
                                        await onDelete()
                                        setMenuOpen(false)
                                    }}
                                    className={`w-full p-ds-sm flex gap-ds-sm items-center rounded
                                        ${isAnyStepRunning ? "opacity-60" : "hover:bg-crema/65"}
                                        ${isDeleting ? "cursor-wait bg-crema/65" : ""}
                                        ${!isAnyStepRunning && !isDeleting ? "cursor-pointer" : ""}`}
                                    disabled={isAnyStepRunning || isDeleting}
                                >
                                    <RiDeleteBinLine size="1.25rem" color="341100" /> {/*musí se zarovnat */}
                                    <span className="text-base text-noir">Delete method</span>
                                </button>
                            </div>
                        )}
                    </div>
                </div>

                {/* tlačítka */}
                <div className="flex items-center justify-between px-ds-md mb-ds-md flex-wrap w-full">
                    <div className="flex items-center gap-ds-md flex-wrap">
                        <button
                            className="cursor-pointer"
                            onClick={() => setIsExpanded(!isExpanded)}
                        >
                            {isExpanded ? (
                                <MdExpandLess size="1.5rem" color="341100" />
                            ) : (
                                <MdExpandMore size="1.5rem" color="341100" />
                            )}
                        </button>

                        <button className={`flex items-center gap-2 px-ds-md py-1 text-espresso border-2 border-transparent  rounded-lg transition
                                        ${isCurrentStepRunning || alreadyRun || isDeleting ? "" : "cursor-pointer hover:border-espresso hover:text-noir"}`}
                            onClick={onSelectStep}
                            disabled={isCurrentStepRunning || alreadyRun || isDeleting || isRunningAll}
                        >
                            <PiSliders size="1.5rem" color="713105" />
                            <span className="text-base">Parameters</span>
                        </button>

                        {/* nejde spustit, pokud už krok proběhl, nebo zrovna probíhá, nebo je zrovna odstraňován */}
                        <button className={`flex items-center gap-2 px-ds-md py-1  text-espresso rounded-lg transition border-2 border-crema
                    ${(!alreadyRun && !isDeleting && !isRunningAll) ? "cursor-pointer" : ""} ${isCurrentStepRunning || isRunningAll ? "cursor-wait" : ""} ${(alreadyRun || isDeleting || isRunningAll) ? "" : "hover:bg-crema"} `}
                            onClick={onExecute}
                            disabled={isCurrentStepRunning || alreadyRun || isDeleting || isRunningAll}
                        >
                            <MdPlayArrow size="1.5rem" />
                            <span className="text-base">{isCurrentStepRunning ? "Running" : alreadyRun ? "Done" : "Run"}</span>
                        </button>
                    </div>

                    {/* {alreadyRun && (
                        <MdDone size="1.75rem" title="The method was successfully completed." color="#713105" />
                    )} */}
                </div>

                {/* expandovaná sekce s parametry */}
                {isExpanded && (
                    <div className="px-ds-md pb-ds-md">
                        {operation?.parameterSchema && operation.parameterSchema.length > 0 ? (
                            <div className="space-y-ds-sm">
                                {operation.parameterSchema.map((param) => (
                                    <div key={param.name} className="text-sm">
                                        <span className="text-noir">{param.label || param.name}</span>
                                        {step?.params && step.params[param.name] !== undefined ? (
                                            <span className="text-noir ml-2">= {String(step.params[param.name])}</span>
                                        ) : (
                                            <span className="text-noir/60 ml-2">(not set)</span>
                                        )}
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-sm text-noir/60 italic">Method has no parameters</p>
                        )}
                    </div>
                )}


            </div>


        </div>
    );
}

export default WorkflowStepCard