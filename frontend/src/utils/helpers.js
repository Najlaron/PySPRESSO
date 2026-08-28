export function CapitalizeFirstLetter(string) {
    if (!string) return ""
    return string.charAt(0).toUpperCase() + string.slice(1)
}

export function formatNetworkError(err) {
    if (!err || err.message === "Failed to fetch") {
        return "Cannot reach the backend. Try refreshing the page or restarting the backend."
    }
    else {
        return err.message
    }
}
