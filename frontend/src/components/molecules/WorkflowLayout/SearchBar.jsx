import { IoSearchOutline } from "react-icons/io5"
import { useRef } from "react"

function SearchBar({
    searchQuery,
    setSearchQuery,
    selectedCategory,
    setSelectedCategory,
    categories = [],
    isWorkflowLoading,
    isDisabled = false
}) {
    const inputRef = useRef(null)
    const isInputDisabled = isWorkflowLoading || isDisabled

    return (
        <div className="flex flex-col">
            <div
                className={`mb-ds-md flex items-center bg-foam py-ds-sm px-ds-md gap-ds-md rounded-[10px] h-14 border border-roast/50
                ${isInputDisabled ? "opacity-70" : "focus-within:border-espresso focus-within:border-2 cursor-text"}`}
                onClick={(e) => {
                    if (isInputDisabled) return
                    if (e.target instanceof Element && e.target.closest("select")) return
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
            <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="text-espresso text-base self-end font-medium bg-foam border
                border-roast/50 rounded-md px-ds-sm py-ds-sm outline-none min-w-36 mb-ds-md"
                disabled={isInputDisabled}
            >
                <option value="">All categories</option>
                {categories.map((category) => (
                    <option key={category} value={category}>
                        {category}
                    </option>
                ))}
            </select>
        </div>

    )
}

export default SearchBar
