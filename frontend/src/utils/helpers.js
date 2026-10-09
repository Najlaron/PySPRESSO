export function CapitalizeFirstLetter(string) {
    if (!string) return ""
    return string
        .split(",")
        .map((category) => {
            const normalized = category.trim().replace(/_/g, " ")
            if (!normalized) return ""
            return normalized.charAt(0).toUpperCase() + normalized.slice(1)
        })
        .filter(Boolean)
        .join(", ")
}

export function formatNetworkError(err) {
    if (!err || err.message === "Failed to fetch") {
        return "Cannot reach the backend. Try refreshing the page or restarting the backend."
    }
    else {
        return err.message
    }
}
