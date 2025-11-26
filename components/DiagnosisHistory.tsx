"use client"
import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import ExpandMoreIcon from './icons/ExpandMore';
import ArrowUpIcon from './icons/ArrowUp';
import ArrowDownIcon from './icons/ArrowDown';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

import { options, data } from './utils/chartMockData';
import { diagnosticHistoryCards } from "./utils/diagnosisMockData";
import BaseCardDiagnosis, { BaseCard } from './BaseCardDiagnosis';

import type { DiagnosisHistory } from '@/lib/services/PatientsTypes';

const styleSystolic = {
    borderColor: 'rgb(230, 111, 210)',
    backgroundColor: 'rgba(230, 111, 210)',
    tension: 0.4,
    pointBorderWidth: 2,
    pointBorderColor: 'rgba(255,255,255)',
    pointRadius: 7,
    borderWidth: 2
}

const styleDiastolic = {
    borderColor: 'rgb(140, 111, 230)',
      backgroundColor: 'rgba(140, 111, 230)',
      tension: 0.4,
      pointBorderWidth: 2,
      pointBorderColor: 'rgba(255,255,255)',
      pointRadius: 7,
      borderWidth: 2
}


const DiagnosisHistory = ({ diagnosisHistory }: {diagnosisHistory : DiagnosisHistory[]}) => {
    
    const diastolics = diagnosisHistory.map( (history) => history.blood_pressure.diastolic.value ).reverse()
    const systolics = diagnosisHistory.map( (history) => history.blood_pressure.systolic.value ).reverse()
    const respiratoryRates = diagnosisHistory.map( (history) => history.respiratory_rate.value ).reverse()
    const heartRates = diagnosisHistory.map( (history) => history.heart_rate.value ).reverse()
    const temperatureRates = diagnosisHistory.map( (history) => history.temperature.value ).reverse()

    const systolicAverage = Math.floor( systolics.reduce( (acc, systolic) => acc + systolic, 0 ) / systolics.length )
    const diastolicAverage = Math.floor( diastolics.reduce( (acc, diastolic) => acc + diastolic, 0 ) / diastolics.length )
    const respiratoryRateAverage = Math.floor(respiratoryRates.reduce( (acc, rate) => acc + rate , 0) / respiratoryRates.length)
    const heartRateAverage = Math.floor(heartRates.reduce( (acc, rate) => acc + rate , 0) / heartRates.length)
    const temperatureRateAverage = Math.floor(temperatureRates.reduce( (acc, rate) => acc + rate , 0) / temperatureRates.length)

    const historyCards = diagnosticHistoryCards.map( (card) => {
        if (card.title.match("Respiratory Rate"))
            return {...card, value: respiratoryRateAverage}
        if (card.title.match("Heart Rate"))
            return {...card, value: heartRateAverage}
        if (card.title.match("Temperature"))
            return {...card, value: temperatureRateAverage}
    }).filter( (element) => element !== undefined) as BaseCard[]

    const labels = diagnosisHistory.map( (history) => `${history.month.slice(0,3)}, ${history.year}` ).reverse()
    const dataChart = {
        labels : labels,
        datasets: [
            {...styleDiastolic, data: diastolics},
            {...styleSystolic, data: systolics}
        ]
    }

    return (
        <section role="list" className="rounded-3xl p-5 bg-white divide-y divide-gray-100">
            <div className="flex items-center pb-10">
                <h2 className="text-2xl font-medium" >Diagnosis History</h2>
            </div> 
            {/* <section className='grid grid-cols-1 grid-rows-1 lg:grid-cols-3 gap-y-12 bg-blue-50 px-5 pt-2 rounded-xl' >
              <div className='grid col-span-2 row-span-1 gap-y-8 lg:gap-0'>
                <div className='grid grid-cols-2 grid-rows-1' >
                    <h2 className="flex items-center text-xl font-medium">Blood Pressure</h2>
                    <div className='grid grid-cols-1 text-sm md:mr-0 lg:mr-8'>
                        <p className='flex justify-end items-center'>
                            Last 6 Months <span className='ml-3'><ExpandMoreIcon /></span>
                        </p>
                    </div>
                </div>
                <div className='flex items-center w-10/12 lg:w-full'>
                    <Line options={options} data={dataChart} /> 
                </div>
              </div>
              <div className='flex flex-wrap'>
                <div className='w-full grid grid-cols-1 grid-rows-3 text-md font-medium mb-4'>
                    <div className='flex items-center'>
                        <div className=' w-4 h-4 mr-2 bg-pink-400 rounded-full' ></div>
                        Systolic
                    </div>
                    <h3 className="py-1 flex items-center text-2xl font-medium">{systolicAverage}</h3>
                    <div className='flex items-center'>
                        <ArrowUpIcon /> 
                        <p className='ml-1 font-extralight'>Higher than Average</p>
                    </div>
                </div>
                <div className='h-px w-full mr-2 bg-gray-400 mb-4' ></div>
                <div className='w-full grid grid-cols-1 grid-rows-3 text-md font-medium'>
                    <div className='flex items-center'>
                        <div className='w-4 h-4 mr-2 bg-purple-400 rounded-full' ></div>
                        Diastolic
                    </div>
                    <h3 className="py-1 flex items-center text-2xl font-medium">{diastolicAverage}</h3>
                    <div className='flex items-center'>
                        <ArrowDownIcon /> 
                        <p className='ml-1 font-extralight'>Lower than Average</p>
                    </div>
                </div>
                <div className='hidden lg:flex w-full h-14'>
                </div>
              </div>
            </section> */}
            <section className='grid grid-cols-1 grid-rows-1 mt-5 gap-5 justify-center lg:grid-cols-3 ' >
            { historyCards.map( (cardPros, index) => {
                return (<BaseCardDiagnosis key={index} cardProps={cardPros} />)
            })}
            </section>
        </section>
        )
}

export default DiagnosisHistory