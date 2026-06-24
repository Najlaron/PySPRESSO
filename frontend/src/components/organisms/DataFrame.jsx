import { AgGridProvider, AgGridReact } from 'ag-grid-react'
import { AllCommunityModule } from 'ag-grid-community';
import { useMemo } from 'react'

function DataFrame({ data }) {
    const rowData = useMemo(() => {
        return data.data.map(row =>
            Object.fromEntries(
                data.columns.map((column, index) => [
                    column,
                    row[index]
                ])
            )
        )
    }, [data])

    const columnDefs = useMemo(() => {
        return data.columns.map(column => ({
            field: column,
            sortable: true,
            resizable: true
        }))
    }, [data])

    const modules = [AllCommunityModule]

    return (


        <AgGridProvider modules={modules}>
            <div
                className="ag-theme-quartz"
                style={{
                    height: '700px',
                    width: '100%'
                }}
            >
                <AgGridReact
                    modules={modules}
                    rowData={rowData}
                    columnDefs={columnDefs}
                    pagination={true}
                    paginationPageSize={100}
                />
            </div>
        </AgGridProvider >
    )
}

export default DataFrame