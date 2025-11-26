export interface Patient {
    name: string;
    gender: string;
    age: number;
    profile_picture: string;
    date_of_birth: string;
    phone_number: string;
    emergency_contact: string;
    insurance_type: string;
    diagnosis_history?: DiagnosisHistory[];
    diagnostic_list?: Diagnostic[];
    lab_results?: string[];
}

export type PatientProfile = Pick<Patient, "name" | "gender" | "age" | "profile_picture" | "date_of_birth" | "phone_number" | "emergency_contact" | "insurance_type">

export interface DiagnosisHistory {
    month: string;
    year: number;
    blood_pressure: BloodPressure;
    heart_rate: HealthMetric;
    respiratory_rate: HealthMetric;
    temperature: HealthMetric;
}

export interface BloodPressure {
    systolic: HealthMetric;
    diastolic: HealthMetric;
}

export interface HealthMetric {
    value: number;
    levels: string;
}

export interface Diagnostic {
    name: string;
    description: string;
    status: string;
}
