import { IoSearchOutline } from "react-icons/io5"
import { useRef } from "react"

function SearchBar({ searchQuery, setSearchQuery, isWorkflowLoading, isDisabled = false }) {
    const inputRef = useRef(null)
    const isInputDisabled = isWorkflowLoading || isDisabled

    return (
        <div
            className={`mb-ds-md flex items-center bg-foam py-ds-sm px-ds-md gap-ds-md rounded-[10px] h-14 border border-roast/50
                ${isInputDisabled ? "opacity-70" : "focus-within:border-espresso focus-within:border-2 cursor-text"}`}
            onClick={() => {
                if (isInputDisabled) return
                inputRef.current?.focus()
            }}
        >
            <IoSearchOutline color="713105" size="1.5rem" />
            <input
                ref={inputRef}
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search for method"
                className="text-espresso text-lg font-medium border-none outline-none bg-transparent w-full"
                disabled={isInputDisabled}
            />
        </div>
    )
}

export default SearchBar
