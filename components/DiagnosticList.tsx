import { Diagnostic } from "@/lib/services/PatientsTypes";
import { diagnosticTableMockData } from "./utils/diagnosticTableMockData";

const DiagnosticList = ({ diagnostics } : { diagnostics : Diagnostic[]}) => {
    return (        
        <section role="list" className="rounded-3xl p-5 bg-white divide-y divide-gray-100">
            <div className="flex items-center pb-10">
                <h2 className="text-2xl font-medium" >Diagnostic List</h2>
            </div> 
            <div className="flex">
                <table className="w-full table-fixed">
                    <thead>
                        <tr>
                        <th className="bg-gray-100 font-medium rounded-l-3xl text-left p-4">Problem / Diagnosis</th>
                        <th className="bg-gray-100 font-medium text-left p-4">Description</th>
                        <th className="bg-gray-100 font-medium rounded-r-3xl text-left p-4">Status</th>
                        </tr>
                    </thead>
                <tbody>
                    
                    {diagnostics.map( (diagnostic, index) => {
                    return (
                    <tr key={index}>
                       <td className="break-normal py-4 pl-4 font-light text-pretty">{diagnostic.name}</td>
                       <td className="py-4 pl-4 font-light">{diagnostic.description}</td>
                       <td className="py-4 pl-4 font-light">{diagnostic.status}</td>
                    </tr>
                  )})}

                </tbody>
                </table>
            </div>
        </section>
)}

export default DiagnosticList