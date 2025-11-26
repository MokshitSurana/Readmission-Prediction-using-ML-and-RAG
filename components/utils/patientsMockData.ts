export const patients = [
    {
      name: "Emily Williams",
      gender: "Female",
      age: 18,
      image: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80",
      isActive: false
    },
    {
      name: "Ryan Johnson",
      gender: "Male",
      age: 45,
      image: "https://images.unsplash.com/photo-1519244703995-f4e0f30006d5?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80",
      isActive: false
    },
    {
      name: "Brandon Mitchell",
      gender: "Male",
      age: 36,
      image: "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80",
      isActive: false
    },
    {
      name: "Jessica Taylor",
      gender: "Female",
      age: 74,
      image: "images/patientprofile.png",
      isActive: true
    },
    {
      name: "Samantha Johnson",
      gender: "Female",
      age: 56,
      image: "https://images.unsplash.com/photo-1517841905240-472988babdf9?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80",
      isActive: false
    },
    {
      name: "Olivia Brown",
      gender: "Female",
      age: 32,
      image: "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80",
      isActive: false
    }
  ];

export const patient = [
	{
		"name": "Jessica Taylor",
		"gender": "Female",
		"age": 74,
		"profile_picture": "https://fedskillstest.ct.digital/4.png",
		"date_of_birth": "1951-06-12",
		"phone_number": "(415) 555-1234",
		"emergency_contact": "(415) 555-5678",
		"insurance_type": "Sunrise Health Assurance",
		"diagnosis_history": [
			{
				"month": "March",
				"year": 2024,
				"blood_pressure": {
					"systolic": {
						"value": 160,
						"levels": "Higher than Average"
					},
					"diastolic": {
						"value": 78,
						"levels": "Lower than Average"
					}
				},
				"heart_rate": {
					"value": 110,
					"levels": "Higher than Average"
				},
				"respiratory_rate": {
					"value": 28,
					"levels": "Normal"
				},
				"temperature": {
					"value": 98,
					"levels": "Normal"
				}
			}
			// Additional months of diagnostic history
		],
		"diagnostic_list": [
			{
				"name": "Hypertension",
				"description": "Chronic high blood pressure",
				"status": "Under Observation"
			}
			// Additional diagnostics
		],
		"lab_results": [
			"Blood Tests",
			"CT Scans"
			// More lab results
		]
	}
]
