import { IoSearchOutline } from "react-icons/io5"
import { useRef, useState } from "react"
import { MdExpandMore, MdExpandLess } from "react-icons/md"
import { CapitalizeFirstLetter } from "../../../utils/helpers"

function SearchBar({
    searchQuery,
    setSearchQuery,
    selectedCategories,
    setSelectedCategories,
    categories = [],
    isWorkflowLoading,
    isDisabled = false
}) {
    const inputRef = useRef(null)
    const isInputDisabled = isWorkflowLoading || isDisabled
    const [isCategoryMenuOpen, setIsCategoryMenuOpen] = useState(false)
    const selectedCategoryValues = Array.isArray(selectedCategories) ? selectedCategories : []
    const selectedCount = selectedCategoryValues.length

    function toggleCategory(category) {
        const isSelected = selectedCategoryValues.includes(category)
        if (isSelected) {
            setSelectedCategories(selectedCategoryValues.filter((item) => item !== category))
            return
        }

        setSelectedCategories([...selectedCategoryValues, category])
    }

    return (
        <div className="flex flex-col">
            <div
                className={`mb-ds-md flex items-center bg-foam py-ds-sm px-ds-md gap-ds-md rounded-[10px] h-14 border border-roast/50
                ${isInputDisabled ? "opacity-70" : "focus-within:border-espresso focus-within:border-2 cursor-text"}`}
                onClick={(e) => {
                    if (isInputDisabled) return
                    if (e.target instanceof Element && e.target.closest("button, input, label")) return
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
            <div className="self-end min-w-80 mb-ds-md">
                <button
                    type="button"
                    onClick={() => setIsCategoryMenuOpen((prev) => !prev)}
                    disabled={isInputDisabled}
                    className="w-full flex items-center justify-between text-espresso text-xl font-medium bg-foam border border-roast/50 rounded-md px-ds-md py-ds-sm outline-none disabled:opacity-70"
                >
                    <span>
                        {selectedCount === 0
                            ? "All categories"
                            : `${selectedCount} categor${selectedCount === 1 ? "y" : "ies"} selected`}
                    </span>
                    {isCategoryMenuOpen ? <MdExpandLess size="1.2rem" /> : <MdExpandMore size="1.2rem" />}
                </button>

                {isCategoryMenuOpen ? (
                    <div className="mt-2 bg-foam border border-roast/50 rounded-md shadow-sm p-ds-md max-h-72 overflow-y-auto">
                        <div className="flex justify-between items-center mb-ds-sm">
                            <p className="text-espresso text-base font-medium">Select categories</p>
                            <button
                                type="button"
                                onClick={() => setSelectedCategories([])}
                                disabled={isInputDisabled || selectedCount === 0}
                                className="text-base text-roast hover:text-espresso disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Clear
                            </button>
                        </div>
                        <div className="flex flex-col gap-2">
                            {categories.map((category) => (
                                <label key={category} className="flex items-center gap-2 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={selectedCategoryValues.includes(category)}
                                        onChange={() => toggleCategory(category)}
                                        disabled={isInputDisabled}
                                        className="h-4 w-4"
                                    />
                                    <span className="text-espresso text-base">
                                        {CapitalizeFirstLetter(category)}
                                    </span>
                                </label>
                            ))}
                        </div>
                    </div>
                ) : null}
            </div>
        </div>

    )
}

export default SearchBar
