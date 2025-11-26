import faker from 'faker';

export const options = {
    responsive: true,
    plugins: {
      legend: {
          display: false,
      },
    },
    scales: {
      x: {
        grid: {
            display: false
        }
    },
    }
  };
  
const labels = ['Oct, 2023', 'Nov, 2023', 'Dec, 2023', 'Jan, 2024', 'Feb, 2024', 'Mar, 2024'];
  
export const data = {
  labels,
  datasets: [
    {
      data: labels.map(() => faker.datatype.number({ min: 120, max: 180 })),
      borderColor: 'rgb(230, 111, 210)',
      backgroundColor: 'rgba(230, 111, 210)',
      tension: 0.4,
      pointBorderWidth: 2,
      pointBorderColor: 'rgba(255,255,255)',
      pointRadius: 7,
      borderWidth: 2
    },
    {
      data: labels.map(() => faker.datatype.number({ min: 60, max: 100 })),
      borderColor: 'rgb(140, 111, 230)',
      backgroundColor: 'rgba(140, 111, 230)',
      tension: 0.4,
      pointBorderWidth: 2,
      pointBorderColor: 'rgba(255,255,255)',
      pointRadius: 7,
      borderWidth: 2
    },
  ],
};