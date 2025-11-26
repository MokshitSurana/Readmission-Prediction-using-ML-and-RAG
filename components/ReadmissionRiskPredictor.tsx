"use client";

import React, { useState } from 'react';
import { type PredictionResult, type PatientData, type ClinicalScores } from './api';

// ======================== FORM STATE INTERFACE ========================
interface ReadmissionFormData {
  age: number;
  gender: string;
  ethnicityEncoded: number;
  hospSeq: number;
  losHospital: number;
  losIcu: number;
  isFirstStay: boolean;
  hospitalExpireFlag: boolean;
  heartRate: number;
  respiratoryRate: number;
  temperature: number;
  map: number;
  pao2Fio2: number;
  urineOutput: number;
  wbcCount: number;
  plateletCount: number;
  creatinine: number;
  bilirubin: number;
  bun: number;
  glucose: number;
  gcsVerbal: number;
  sofa: number;
  apsiii: number;
  oasis: number;
  elixhauserSID30: number;
  apsiiiProb: number;
  elixhauserQuan: number;
  diseaseName: string;
}

// ======================== SCENARIO DATA ========================

// --- SCENARIO 1: LOW RISK (Pneumonia) ---
const example1Data: ReadmissionFormData = {
  age: 76,
  gender: 'Female',
  ethnicityEncoded: 3, 
  hospSeq: 1,
  losHospital: 7.2,
  losIcu: 4.8,
  isFirstStay: true,
  hospitalExpireFlag: false,
  heartRate: 112,
  respiratoryRate: 26,
  temperature: 38.3,
  map: 64,
  pao2Fio2: 165,
  urineOutput: 1300,
  wbcCount: 15.8,
  plateletCount: 188,
  creatinine: 1.6,
  bilirubin: 0.9,
  bun: 36,
  glucose: 182,
  gcsVerbal: 4,
  sofa: 9,
  apsiii: 54,
  oasis: 36,
  elixhauserSID30: 12,
  apsiiiProb: 0.32,
  elixhauserQuan: 9,
  diseaseName: 'Acute respiratory failure due to pneumonia'
};

const example1Text = `76-year-old female admitted with acute respiratory failure secondary to multifocal pneumonia. 
Initial presentation included hypoxia (PaO2/FiO2 = 165), tachycardia (HR 110–120), borderline MAP (64 mmHg), and leukocytosis (WBC 15.8).
Patient required high-flow nasal cannula on admission and was transitioned to BiPAP. 
Received antibiotics, bronchodilators, and diuresis. Renal function stable (creatinine 1.6).
Weaned to nasal cannula day 4, transferred to ward day 5, discharged home day 7.`;

const example1Result: PredictionResult = {
  willBeReadmitted: false,
  probability: 0.251,
  mlProbability: 0.153,
  textProbability: 0.350,
  riskCategory: 'Low Risk',
  cohort: 'RESPIRATORY_FAILURE',
  reasoning: `Although the patient had a relatively high SOFA score of 9, indicating severe organ dysfunction upon admission, her Elixhauser comorbidity index was moderate at 12 and she responded well to treatment, being weaned off advanced respiratory support within four days. This response, combined with a stable renal function, suggests that the patient's condition improved quickly compared to similar historical patients who may have had longer hospital stays or required more intensive care. However, it is important to note that the specific case details were not provided for comparison purposes.`,
  scores: { sofa: 9, apsiii: 54, oasis: 36, elixhauserSID30: 12 } as ClinicalScores,
  similarCases: []
};


// --- SCENARIO 2: MODERATE RISK (Sepsis) ---
const example2Data: ReadmissionFormData = {
  age: 68,
  gender: 'Male',
  ethnicityEncoded: 1,
  hospSeq: 1,
  losHospital: 9.5,
  losIcu: 5.2,
  isFirstStay: true,
  hospitalExpireFlag: false,
  heartRate: 104,
  respiratoryRate: 22,
  temperature: 39.1,
  map: 62,
  pao2Fio2: 280,
  urineOutput: 850,
  wbcCount: 22.5,
  plateletCount: 135,
  creatinine: 2.4,
  bilirubin: 1.2,
  bun: 48,
  glucose: 155,
  gcsVerbal: 5,
  sofa: 7,
  apsiii: 49,
  oasis: 34,
  elixhauserSID30: 18,
  apsiiiProb: 0.25,
  elixhauserQuan: 14,
  diseaseName: 'Septicemia due to urinary tract infection'
};

const example2Text = `68-year-old male admitted with septicemia secondary to a complicated urinary tract infection (UTI). Patient presented with altered mental status, high fevers (Temp 39.1C), and rigors. Initial vitals showed tachycardia (HR 100-110) and hypotension (MAP 62 mmHg) unresponsive to initial fluid bolus. Labs were significant for severe leukocytosis (WBC 22.5), thrombocytopenia (Platelets 135), and acute kidney injury (Creatinine 2.4, BUN 48) consistent with sepsis-induced organ dysfunction. He was started on broad-spectrum antibiotics and required a brief period of norepinephrine support in the ICU. Urine cultures grew E. coli. Renal function gradually improved with hydration (Cr down to 1.5). He was weaned off pressors by day 3 and transferred to the floor on day 6. Discharged to rehabilitation facility for continued physical therapy and completion of antibiotic course.`;

const example2Result: PredictionResult = {
  willBeReadmitted: true,
  probability: 0.424,
  mlProbability: 0.227,
  textProbability: 0.620,
  riskCategory: 'Moderate Risk',
  cohort: 'SEPTICEMIA',
  reasoning: `The moderate risk of readmission is based on the patient's high SOFA score of 7.0, Elixhauser Comorbidity Index of 18.0, and acute kidney injury. These scores suggest that the patient has experienced a severe illness and may be at higher risk for complications, which could lead to readmission. However, this prediction should be considered in context with ongoing treatment and the patient's response to therapy. Compared to similar historical patients, this case shows slightly higher severity scores, which may contribute to the increased estimated readmission probability.`,
  scores: { sofa: 7, apsiii: 49, oasis: 34, elixhauserSID30: 18 } as ClinicalScores,
  similarCases: []
};


// ======================== COMPONENT ========================
const ReadmissionRiskPredictor: React.FC = () => {
  const [formData, setFormData] = useState<ReadmissionFormData>(example1Data);
  const [dischargeSummaryText, setDischargeSummaryText] = useState<string>(example1Text);
  const [inputMethod, setInputMethod] = useState<'text' | 'file'>('text');
  
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [showPrediction, setShowPrediction] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activeScenario, setActiveScenario] = useState<1 | 2>(1);

  // Load Example Handlers
  const loadExample1 = () => {
    setFormData(example1Data);
    setDischargeSummaryText(example1Text);
    setActiveScenario(1);
    setShowPrediction(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const loadExample2 = () => {
    setFormData(example2Data);
    setDischargeSummaryText(example2Text);
    setActiveScenario(2);
    setShowPrediction(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    let parsedValue: any = value;
    if (type === 'number') parsedValue = Number(value);
    setFormData({ ...formData, [name]: parsedValue });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    // SIMULATE API DELAY
    setTimeout(() => {
      if (activeScenario === 1) {
        setPrediction(example1Result);
      } else {
        setPrediction(example2Result);
      }
      setLoading(false);
      setShowPrediction(true);
      
      // Scroll to results
      setTimeout(() => {
        document.getElementById('prediction-results')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }, 1500);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="p-12 text-center bg-white rounded-2xl shadow-lg border border-gray-100 max-w-lg w-full">
          <div className="animate-spin w-16 h-16 border-4 border-teal-500 border-t-transparent rounded-full mx-auto mb-6"></div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Analyzing Patient Data...</h2>
          <div className="space-y-2 text-gray-500">
             <p>Processing 27 Clinical Features</p>
             <p>Running Cohort-Specific ML Models</p>
             <p>Generating LLM Clinical Reasoning</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <section className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 md:p-8 max-w-6xl mx-auto my-8">
      <div className="flex flex-col md:flex-row justify-between items-center mb-8 border-b border-gray-100 pb-6 gap-4">
        <div>
          <h2 className="text-3xl font-bold text-gray-900 tracking-tight">ICU Readmission Predictor</h2>
          <p className="text-gray-500 mt-1">Interactive Clinical Demo</p>
        </div>
        
        <div className="flex gap-3">
          <button 
            onClick={loadExample1}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeScenario === 1 ? 'bg-teal-100 text-teal-800 border border-teal-200' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
          >
            Load Example 1 (Low Risk)
          </button>
          <button 
            onClick={loadExample2}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeScenario === 2 ? 'bg-teal-100 text-teal-800 border border-teal-200' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
          >
            Load Example 2 (Moderate Risk)
          </button>
        </div>
      </div>

      {!showPrediction ? (
        <form onSubmit={handleSubmit} className="space-y-8 animate-in fade-in duration-500">
          
          {/* GROUP 1: DEMOGRAPHICS & STAY */}
          <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
            <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4 border-b border-gray-200 pb-2">1. Demographics & Stay</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1">Age</label>
                <input type="number" name="age" value={formData.age} onChange={handleInputChange} className="w-full p-2.5 bg-white rounded border border-gray-300 focus:border-teal-500 outline-none" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1">Gender</label>
                <select name="gender" value={formData.gender} onChange={handleInputChange} className="w-full p-2.5 bg-white rounded border border-gray-300">
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1">Ethnicity Group</label>
                <select name="ethnicityEncoded" value={formData.ethnicityEncoded} onChange={handleInputChange} className="w-full p-2.5 bg-white rounded border border-gray-300">
                  <option value={0}>Group 0</option>
                  <option value={1}>Group 1</option>
                  <option value={2}>Group 2</option>
                  <option value={3}>Group 3 (White/Caucasian)</option>
                  <option value={4}>Group 4</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1">Hospital Sequence</label>
                <input type="number" name="hospSeq" value={formData.hospSeq} onChange={handleInputChange} className="w-full p-2.5 bg-white rounded border border-gray-300" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1">Hospital LOS (Days)</label>
                <input type="number" step="0.1" name="losHospital" value={formData.losHospital} onChange={handleInputChange} className="w-full p-2.5 bg-white rounded border border-gray-300" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1">ICU LOS (Days)</label>
                <input type="number" step="0.1" name="losIcu" value={formData.losIcu} onChange={handleInputChange} className="w-full p-2.5 bg-white rounded border border-gray-300" />
              </div>
            </div>
          </div>

          {/* GROUP 2: VITALS */}
          <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
            <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4 border-b border-gray-200 pb-2">2. First Day Vitals (Mean)</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-5">
              {['heartRate', 'temperature', 'map', 'respiratoryRate', 'pao2Fio2', 'urineOutput'].map(field => (
                <div key={field}>
                  <label className="block text-xs font-semibold text-gray-600 mb-1 capitalize">
                    {field.replace(/([A-Z])/g, ' $1').trim()}
                  </label>
                  <input 
                    type="number" 
                    step={field === 'temperature' ? '0.1' : '1'} 
                    name={field} 
                    value={(formData as any)[field]} 
                    onChange={handleInputChange} 
                    className="w-full p-2.5 bg-white rounded border border-gray-300 focus:border-teal-500 outline-none" 
                  />
                </div>
              ))}
            </div>
          </div>

          {/* GROUP 3: LABS */}
          <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
            <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4 border-b border-gray-200 pb-2">3. Labs</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-5">
              {['wbcCount', 'plateletCount', 'creatinine', 'bilirubin', 'bun', 'glucose'].map(field => (
                <div key={field}>
                  <label className="block text-xs font-semibold text-gray-600 mb-1 capitalize">
                    {field.replace(/([A-Z])/g, ' $1').trim()}
                  </label>
                  <input 
                    type="number" 
                    step="0.1"
                    name={field} 
                    value={(formData as any)[field]} 
                    onChange={handleInputChange} 
                    className="w-full p-2.5 bg-white rounded border border-gray-300 focus:border-teal-500 outline-none" 
                  />
                </div>
              ))}
            </div>
          </div>

          {/* GROUP 4: NEURO & SCORES */}
          <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
            <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4 border-b border-gray-200 pb-2">4. Neuro & Severity Scores</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1">GCS Verbal</label>
                <input type="number" max="5" name="gcsVerbal" value={formData.gcsVerbal} onChange={handleInputChange} className="w-full p-2.5 bg-white rounded border border-gray-300" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1">SOFA</label>
                <input type="number" name="sofa" value={formData.sofa} onChange={handleInputChange} className="w-full p-2.5 bg-white rounded border border-gray-300" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1">APS III</label>
                <input type="number" name="apsiii" value={formData.apsiii} onChange={handleInputChange} className="w-full p-2.5 bg-white rounded border border-gray-300" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1">OASIS</label>
                <input type="number" name="oasis" value={formData.oasis} onChange={handleInputChange} className="w-full p-2.5 bg-white rounded border border-gray-300" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1">Elixhauser SID30</label>
                <input type="number" name="elixhauserSID30" value={formData.elixhauserSID30} onChange={handleInputChange} className="w-full p-2.5 bg-white rounded border border-gray-300" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1">Elixhauser Quan</label>
                <input type="number" name="elixhauserQuan" value={formData.elixhauserQuan} onChange={handleInputChange} className="w-full p-2.5 bg-white rounded border border-gray-300" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1">APS III Prob</label>
                <input type="number" step="0.01" max="1" name="apsiiiProb" value={formData.apsiiiProb} onChange={handleInputChange} className="w-full p-2.5 bg-white rounded border border-gray-300" />
              </div>
            </div>
          </div>

          {/* GROUP 5: NOTES */}
          <div className="bg-white border border-gray-200 p-6 rounded-xl shadow-sm">
            <h3 className="text-lg font-bold text-gray-700 mb-4">Clinical Context</h3>
            <div className="mb-6">
              <label className="block text-xs font-bold text-gray-500 uppercase tracking-wide mb-1">Primary Diagnosis</label>
              <input type="text" name="diseaseName" value={formData.diseaseName} onChange={handleInputChange} className="w-full p-3 border border-gray-300 rounded-lg text-gray-800 font-medium bg-gray-50" />
            </div>

            <div className="flex gap-4 mb-4 text-sm border-b border-gray-100 pb-2">
              <label className="flex items-center cursor-pointer font-medium text-teal-700">
                <input type="radio" name="inputMethod" checked={inputMethod === 'text'} onChange={() => setInputMethod('text')} className="mr-2 text-teal-600" /> 
                Text Input
              </label>
              <label className="flex items-center cursor-pointer text-gray-500">
                <input type="radio" name="inputMethod" checked={inputMethod === 'file'} onChange={() => setInputMethod('file')} className="mr-2 text-gray-400" disabled /> 
                File Upload (Disabled in Demo)
              </label>
            </div>

            <div>
              <div className="flex justify-between mb-2">
                <label className="text-xs font-bold text-gray-500 uppercase tracking-wide">Discharge Summary</label>
              </div>
              <textarea 
                rows={8} 
                className="w-full p-4 border border-gray-300 rounded-lg focus:ring-1 focus:ring-teal-500 outline-none text-sm leading-relaxed text-gray-700 bg-gray-50" 
                value={dischargeSummaryText} 
                readOnly
              />
            </div>
          </div>

          <button type="submit" disabled={loading} className="w-full bg-teal-600 hover:bg-teal-700 text-white font-bold py-4 rounded-xl shadow-lg transition-all active:scale-95 text-lg flex items-center justify-center gap-2">
            <span>🚀</span> Run Analysis
          </button>
        </form>
      ) : (
        // RESULT VIEW
        <div id="prediction-results" className="animate-in fade-in slide-in-from-bottom-8 duration-700">
          
          {/* MAIN RISK CARD */}
          <div className={`p-10 rounded-2xl mb-8 text-center text-white shadow-xl ${prediction?.riskCategory === 'High' ? 'bg-gradient-to-br from-red-500 to-red-600' : prediction?.riskCategory === 'Moderate' ? 'bg-gradient-to-br from-orange-400 to-orange-500' : 'bg-gradient-to-br from-green-500 to-green-600'}`}>
            <h3 className="text-sm font-bold opacity-80 uppercase tracking-widest mb-2">30-Day Readmission Risk</h3>
            <h2 className="text-6xl font-extrabold mb-3 tracking-tight">{prediction?.riskCategory}</h2>
            <p className="text-3xl font-medium opacity-95 mb-6">{(prediction?.probability! * 100).toFixed(1)}% Probability</p>
            
            {/* MINIMIZED MODEL CONTRIBUTION (Footer Style) */}
            <div className="inline-flex items-center bg-white/20 backdrop-blur-sm rounded-full px-6 py-2 text-sm font-medium">
               <span className="opacity-75 mr-3">Confidence Source:</span>
               <span className="flex items-center mr-4"><span className="mr-1.5">🧪</span> {(prediction?.mlProbability! * 100).toFixed(1)}% Structured Data</span>
               <span className="opacity-50 mr-4">•</span>
               <span className="flex items-center"><span className="mr-1.5">📄</span> {(prediction?.textProbability! * 100).toFixed(1)}% Clinical Text</span>
            </div>
          </div>

          {/* CLINICAL SCORES - FULL WIDTH */}
          <div className="bg-white p-8 rounded-2xl border border-gray-100 shadow-sm mb-8">
            <h4 className="font-bold text-gray-800 mb-6 flex items-center gap-2">
               <span className="text-xl">📊</span> Computed Severity Scores
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="text-center p-4 bg-gray-50 rounded-xl border border-gray-100">
                <div className="text-xs text-gray-500 uppercase font-bold tracking-wider mb-1">SOFA</div>
                <div className="text-3xl font-bold text-gray-800">{prediction?.scores.sofa}</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-xl border border-gray-100">
                <div className="text-xs text-gray-500 uppercase font-bold tracking-wider mb-1">APS III</div>
                <div className="text-3xl font-bold text-gray-800">{prediction?.scores.apsiii}</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-xl border border-gray-100">
                <div className="text-xs text-gray-500 uppercase font-bold tracking-wider mb-1">OASIS</div>
                <div className="text-3xl font-bold text-gray-800">{prediction?.scores.oasis}</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-xl border border-gray-100">
                <div className="text-xs text-gray-500 uppercase font-bold tracking-wider mb-1">Elixhauser</div>
                <div className="text-3xl font-bold text-gray-800">{prediction?.scores.elixhauserSID30}</div>
              </div>
            </div>
          </div>

          {/* CLINICAL REASONING */}
          <div className="bg-teal-50 border border-teal-100 p-8 rounded-2xl mb-8 shadow-sm">
            <div className="flex items-center gap-3 mb-4">
               <span className="text-3xl">💡</span>
               <h4 className="font-bold text-teal-900 text-xl">AI Clinical Reasoning</h4>
            </div>
            <p className="text-teal-900 leading-relaxed whitespace-pre-wrap text-lg font-medium pl-2 border-l-4 border-teal-200">
               {prediction?.reasoning}
            </p>
          </div>

          <button onClick={() => { setShowPrediction(false); window.scrollTo({ top: 0, behavior: 'smooth' }); }} className="w-full py-4 bg-gray-100 hover:bg-gray-200 rounded-xl font-bold text-gray-700 transition-colors text-lg">
            Start New Prediction
          </button>
        </div>
      )}
    </section>
  );
};

export default ReadmissionRiskPredictor;