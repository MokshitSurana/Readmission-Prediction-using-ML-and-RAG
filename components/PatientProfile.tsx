import CalendarTodayIcon from "./icons/CalendarIcon";
import FemaleIcon from "./icons/FemaleIcon";
import InsuranceIcon from "./icons/InsuranceIcon";
import PhoneIcon from "./icons/PhoneIcon";
import { formatDate } from "./utils/formatDate";
import { patient } from './utils/patientsMockData';
import type { PatientProfile } from "@/lib/services/PatientsTypes";

const PatientProfile = ({ patient }: { patient: PatientProfile}) => {

    const formattedDate = formatDate(new Date(patient.date_of_birth))

    return (        
        <section role="list" className="rounded-3xl p-5 bg-white divide-y divide-gray-100">
            <div className="grid grid-cols-1 justify-items-center gap-6 pb-4">
                <img className="pt-4 w-3/4" src={patient.profile_picture} alt="" />
                <h2 className="text-2xl font-medium" >{patient.name}</h2>
                <div className="w-full grid grid-cols-6 gap-8 grid-rows-1" >
                    <div className="col-span-1 text-center m-auto bg-gray-100 p-4 rounded-full">
                        <CalendarTodayIcon />
                    </div>
                    <div className="col-span-5" >
                        <h3 className=" font-light" >Date Of Birth</h3>
                        <p className=" font-semibold">{formattedDate}</p>
                    </div>
                    <div className="col-span-1 text-center m-auto bg-gray-100 p-0.5 rounded-full">
                        <FemaleIcon />
                    </div>
                    <div className="col-span-5" >
                        <h3 className=" font-light" >Gender</h3>
                        <p className=" font-semibold">{patient.gender}</p>
                    </div>
                    <div className="col-span-1 text-center m-auto bg-gray-100 p-0.5 rounded-full">
                        <PhoneIcon />
                    </div>
                    <div className="col-span-5" >
                        <h3 className=" font-light" >Contact Info.</h3>
                        <p className=" font-semibold">{patient.phone_number}</p>
                    </div>
                    <div className="col-span-1 text-center m-auto bg-gray-100 p-0.5 rounded-full">
                        <PhoneIcon />
                    </div>
                    <div className="col-span-5" >
                        <h3 className=" font-light" >Emergency Contacts</h3>
                        <p className=" font-semibold">{patient.emergency_contact}</p>
                    </div>
                    <div className="col-span-1 text-center m-auto bg-gray-100 p-0.5 rounded-full">
                        <InsuranceIcon />
                    </div>
                    <div className="col-span-5" >
                        <h3 className=" font-light" >Insurance Provider</h3>
                        <p className=" font-semibold">{patient.insurance_type}</p>
                    </div>
                </div>
                <button className="mt-8 justify-self-center bg-teal-300 hover:bg-teal-500 text-black font-semibold px-12 py-4 rounded-full">
                    Show All Information
                </button>
            </div> 
        </section>
)}

export default PatientProfile