import { TiDeleteOutline } from "react-icons/ti"
import { MdErrorOutline } from "react-icons/md"
import { HiOutlineX } from "react-icons/hi"
import { HiOutlineExclamationTriangle } from "react-icons/hi2"


function ErrorAlert({ message, className = "", onDismiss }) {
    if (!message) return null

    return (
        <div className={`relative p-ds-lg pr-12 rounded border border-red-200 bg-red-50 text-red-800 
                        text-xl text-medium flex items-center gap-ds-md ${className}`}>
            <div className="shrink-0 flex items-center justify-center">
                <HiOutlineExclamationTriangle size="2rem" />
            </div>
            <span className="flex-1 mr-ds-md">{message}</span>
            {onDismiss ? (
                <button
                    type="button"
                    onClick={onDismiss}
                    aria-label="Dismiss error"
                    className="absolute top-2 right-2 cursor-pointer text-red-700 
                                transition hover:text-red-900"
                >
                    <HiOutlineX size="1.5rem" />
                </button>
            ) : null}
        </div>
    )
}

export default ErrorAlert
