import "../../../../css/tooltip.css"

function Tooltip({ children, text, placement = "top" }) {
    return (
        <div className="tooltip-container">
            <div className="tooltip-children">
                {children}
                <div
                    className={`tooltip ${
                        placement === "bottom" ? "tooltip-bottom" : ""
                    }`}
                    role="tooltip"
                >
                    <span>{text}</span>
                </div>
            </div>
        </div>
    )
}

export default Tooltip