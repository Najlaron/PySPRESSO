import { AgGridProvider, AgGridReact } from 'ag-grid-react'
import { AllCommunityModule, themeQuartz } from 'ag-grid-community'
import { useMemo } from 'react'

const tableTheme = themeQuartz
    .withParams({
        backgroundColor: '#ffffff',
        foregroundColor: '#341100',
        headerBackgroundColor: '#faf8f5',
        spacing: 12,
        fontSize: 16,
        headerFontSize: 18,
        wrapperBorder: false,
        headerRowBorder: false,
    })

// funkce pro správné vypisování řádků
function DataFrame({ data }) {
    const normalizeFieldName = (name) => {
        return name
            .replace(/[()[\]]/g, '')
            .replace(/\s+/g, '_')
            .replace(/\./g, '_')
    }

    const rowData = useMemo(() => {
        return data.data.map((row) =>
            Object.fromEntries(
                data.columns.map((column, index) => [
                    normalizeFieldName(column),
                    row[index] !== undefined ? row[index] : null
                ])
            )
        )
    }, [data])

    const columnDefs = useMemo(() => {
        return data.columns.map(column => ({
            field: normalizeFieldName(column),
            headerName: column,
            sortable: true,
            resizable: true
        }))
    }, [data])

    const modules = [AllCommunityModule]

    return (
        <AgGridProvider modules={modules}>
            <div
                className="h-175 w-full shadow-lg"
            >
                <AgGridReact
                    modules={modules}
                    rowData={rowData}
                    columnDefs={columnDefs}
                    pagination={true}
                    paginationPageSize={100}
                    theme={tableTheme}
                />
            </div>
        </AgGridProvider >
    )
}

export default DataFrame