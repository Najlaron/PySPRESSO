// import { CiSquarePlus } from "react-icons/ci";
import { FaP, FaPlus } from "react-icons/fa6";
import { CapitalizeFirstLetter } from "../../../utils/helpers"
import { IoAddCircleOutline } from "react-icons/io5";
import { MdOutlineAddBox } from "react-icons/md";


function SearchedMethod({ operation, onClick, addedId, disabled = false }) {
    const isCurrentOpAdded = operation.id === addedId
    const isDisabled = disabled && !isCurrentOpAdded

    return (
        <div className={`flex items-center justify-between gap-ds-md border-l-8 border-l-transparent p-ds-md bg-foam
            ${isCurrentOpAdded ? "bg-crema/65 border-l-espresso cursor-wait" : ""}
            ${isDisabled ? "opacity-60" : "cursor-pointer hover:bg-crema/65 hover:border-l-espresso"}`}
            onClick={() => {
                if (isCurrentOpAdded || isDisabled) return;
                onClick();
            }}
        >
            <div>
                <p className="text-noir font-medium">{operation.label}</p>
                <p className="text-espresso">{CapitalizeFirstLetter(operation?.categoryTags?.join(", "))}</p>
            </div>
            {/* <IoAddCircleOutline size="2.5rem" color="#713105" /> */}
            <MdOutlineAddBox size="2.5rem" color="#713105" />
        </div>
    )
}


export default SearchedMethod