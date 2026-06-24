function TabDataButton({ label, active, onClick }) {
    return (
        <button
            className={`border border-crema px-ds-md py-2 rounded-md text-lg cursor-pointer ${active ? "bg-espresso text-light-foam" : "hover:bg-crema  text-noir"}`}
            onClick={onClick}
        >
            {label}
        </button >
    )
}

export default TabDataButton