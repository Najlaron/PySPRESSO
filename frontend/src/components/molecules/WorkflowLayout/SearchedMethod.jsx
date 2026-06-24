// import { CiSquarePlus } from "react-icons/ci";
import { FaP, FaPlus } from "react-icons/fa6";
import { CapitalizeFirstLetter } from "../../../utils/helpers"


function SearchedMethod({ operation, onClick, isSelected }) {

    return (
        <div className={`flex items-center gap-ds-md cursor-pointer border-l-8 border-l-transparent hover:bg-crema/65 hover:border-l-espresso p-ds-md ${isSelected ? "bg-foam border-2 border-espresso rounded-2xl mb-ds-md" : "bg-foam"}`} onClick={onClick} >
            {/* <FaPlus size="1rem" color="#713105" /> */}
            <div>
                <p className="text-noir font-medium">{operation.label}</p>
                <p className="text-espresso">{CapitalizeFirstLetter(operation?.categoryTags?.join(", "))}</p>
            </div>
        </div>
    )
}


export default SearchedMethod