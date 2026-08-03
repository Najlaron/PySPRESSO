function StepBadge({ stepNumber }) {
    return (
        <div className="w-18 h-18 rounded-full bg-espresso flex items-center justify-center">
            <span className="font-bold text-foam text-3xl">{stepNumber}</span>
        </div>
    )
}

export default StepBadge