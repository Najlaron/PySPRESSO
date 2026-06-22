import "../../../css/tooltip.css"

function Tooltip({ children, text }) {


    return (
        <div className="tooltip-container">
            <div className="tooltip-children">
                {children}
                <div className="tooltip">
                    <span>{text}</span>
                </div>
            </div>
        </div>
    )
}

export default Tooltip