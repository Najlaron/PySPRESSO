import { FaArrowRight } from "react-icons/fa6";


function CreatingWfButton({ title, description, color }) {
    return (
        <div className={`bg-light-foam border-l-4 ${color === "crema" ? "border-crema" : "border-espresso"} rounded-3xl shadow-md p-ds-lg flex items-center`}>
            <div className="pr-ds-sm">
                <h4 className="text-2xl font-semibold mb-ds-md max-w-75">{title}</h4>
                <p className="text-xl max-w-75">{description}</p>
            </div>
            <FaArrowRight size="1.5rem" color="#7F5E35" />
        </div>
    )
}

export default CreatingWfButton