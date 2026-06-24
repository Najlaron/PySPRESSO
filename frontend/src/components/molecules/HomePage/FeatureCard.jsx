function FeatureCard({ title, description, Icon }) {
    return (
        <article className="bg-light-foam shadow-lg p-ds-lg rounded-lg">
            <div className="inline-flex items-center justify-center rounded-xl bg-crema/75 mb-ds-lg">
                <Icon className="w-25" />
            </div>

            <h3 className="text-3xl font-semibold text-noir mb-ds-md">{title}</h3>
            <p className="text-xl text-espresso">{description}</p>
        </article>
    )
}

export default FeatureCard