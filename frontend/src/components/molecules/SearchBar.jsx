import { IoSearchOutline } from "react-icons/io5"
import { useRef } from "react"

function SearchBar({ searchQuery, setSearchQuery }) {
    const inputRef = useRef(null)

    return (
        <div
            className="mb-ds-md flex items-center bg-foam py-ds-sm px-ds-md gap-ds-md rounded-[10px] h-14 border border-roast/50 focus-within:border-espresso focus-within:border-2 cursor-text"
            onClick={() => inputRef.current?.focus()}
        >
            <IoSearchOutline color="713105" size="1.5rem" />
            <input
                ref={inputRef}
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search for method"
                className="text-espresso text-lg font-medium border-none outline-none bg-transparent w-full"
            />
        </div>
    )
}

export default SearchBar
