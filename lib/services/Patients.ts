import { API_URL, API_AUTHORIZATION_USERNAME, API_AUTHORIZATION_PASSWORD } from "./const";
import { Patient } from './PatientsTypes';

const auth = btoa(`${API_AUTHORIZATION_USERNAME}:${API_AUTHORIZATION_PASSWORD}`)

const getAllPatients : () => Promise<Patient[]> = async function () {

    const res = await fetch(API_URL, {
        method: 'GET',
        headers: {
            "Content-Type": "application/json",
            Authorization: `Basic ${auth}`
        }
    })

    console.log("Authorization", `Basic ${auth}`)

    if (!res.ok)
        throw new Error(`Error fetching Patient API Data ${res.status, res.statusText}`, );

    return res.json()
} 

export default getAllPatients