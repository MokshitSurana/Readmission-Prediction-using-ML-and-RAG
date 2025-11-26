const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface PatientData {
  // Demographics
  age: number;
  gender: string;
  ethnicityEncoded: number;

  // Stay Info
  hospSeq: number;
  losHospital: number;
  losIcu: number;
  isFirstStay: boolean;
  hospitalExpireFlag: boolean;

  // Diagnosis
  diseaseName: string;
  dischargeSummary?: string;

  // Vitals
  heartRate: number;
  respiratoryRate: number;
  temperature: number;
  map: number;
  pao2Fio2: number;
  urineOutput: number;

  // Labs
  wbcCount: number;
  plateletCount: number;
  creatinine: number;
  bilirubin: number;
  bun: number;
  glucose: number;

  // Neuro
  gcsVerbal: number;

  // Severity Scores
  sofa: number;
  apsiii: number;
  oasis: number;
  elixhauserSID30: number;
  apsiiiProb: number;
  elixhauserQuan: number;
}

export interface SimilarCase {
  rank: number;
  similarity: number;
  readmit: number;
  age: number;
  sofa: number;
  apsiii: number;
  elixhauser: number;
  text: string;
  disease: string;
}

export interface ClinicalScores {
  sofa: number;
  apsiii: number;
  oasis: number;
  elixhauserSID30: number;
}

export interface PredictionResult {
  willBeReadmitted: boolean;
  probability: number;
  mlProbability: number;
  textProbability: number;
  riskCategory: string;
  cohort: string;
  reasoning: string;
  scores: ClinicalScores;
  similarCases: SimilarCase[];
}

export class APIError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'APIError';
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    const data = await response.json();
    return data.status === 'healthy';
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
}

export async function predictReadmission(
  patientData: PatientData
): Promise<PredictionResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(patientData),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        response.status,
        errorData.detail || `Prediction failed with status ${response.status}`
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    console.error('Prediction request failed:', error);
    throw new Error('Failed to connect to prediction service. Please ensure the backend is running.');
  }
}