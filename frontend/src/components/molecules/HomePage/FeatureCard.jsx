// karta pro zobrazení vlasnosti modulu
function FeatureCard({ title, description, Icon, iconSize }) {
    return (
        <article className="bg-light-foam shadow-lg p-ds-lg rounded-lg flex-1">
            <div className="inline-flex items-center justify-center rounded-xl bg-crema/75 mb-ds-lg w-20 h-20">
                <Icon className="object-contain p-2" size={iconSize} color="#713105" />
            </div>

            <h3 className="text-3xl font-semibold text-noir mb-ds-md">{title}</h3>
            <p className="text-xl text-espresso">{description}</p>
        </article>
    )
}

export default FeatureCard