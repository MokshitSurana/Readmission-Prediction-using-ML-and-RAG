import PatientList from "@/components/PatientList";
import DiagnosisHistory from '@/components/DiagnosisHistory';
import DiagnosticList from "@/components/DiagnosticList";
import PatientProfile from "@/components/PatientProfile";
import LabResultsList from "@/components/LabResultList";

import getAllPatients from "@/lib/services/Patients";
import type { DiagnosisHistory as DHistory, Diagnostic, Patient, PatientProfile as PProfile } from "@/lib/services/PatientsTypes";


function getProfileData<Patient>(patient: Patient) {
  let profile: { 
    [key : string] : NonNullable<NonNullable<Patient>[Extract<keyof Patient, string>]> 
  } = {}

  for (const key in patient) {
    if (patient) {
      const value = patient[key]
      if(value && !Array.isArray(value) ) {
        profile[key] = value
      }
    }
  }

  return profile
}

export default async function Home() {
  const initialData = await getAllPatients()

  const patient : Patient | undefined = initialData.find( (patient) => patient.name.match("Jessica Taylor") )
  let profile!: PProfile
  let diagnosisHistory! : DHistory[]
  let diagnoticList! : Diagnostic[]
  let labResults! : Array<string>

  if (patient) {
    profile = getProfileData<Patient>(patient) as PProfile

    if (patient.diagnosis_history)
      diagnosisHistory = patient.diagnosis_history

    if (patient.diagnostic_list)
      diagnoticList = patient.diagnostic_list

    if (patient.lab_results)
      labResults = patient.lab_results
  }

  return (
    <main className="flex flex-wrap justify-center lg:grid lg:grid-rows-1 lg:grid-flow-col lg:gap-x-8 lg:grid-cols-4 min-h-screen mx-4 mb-8">
      <section className="mb-8 lg:mb-0" > <PatientList/> </section>
      <section className="mb-8 lg:mb-0 grid grid-cols-1 col-start-2 col-end-4 gap-8">
        <DiagnosisHistory diagnosisHistory={diagnosisHistory} />
        <DiagnosticList diagnostics={diagnoticList} />
      </section>
      <section className="mb-8 lg:mb-0 grid grid-cols-1 gap-8">
        <PatientProfile patient={profile} />
        <LabResultsList labResults={labResults} />
      </section>
    </main>
  );
}
