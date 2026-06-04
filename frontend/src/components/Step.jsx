import StepBadge from './StepBadge'

function Step({ title, description, number }) {
    return (
        <div className="flex gap-ds-xl">
            <StepBadge stepNumber={number} />
            <div className="max-w-lg">
                <h3 className="text-3xl font-bold text-noir mb-ds-md">{title}</h3>
                <p className="text-xl text-grounds">{description}</p>
            </div>
        </div>
    )
}

export default Step