import Image from "next/image"
import ArrowDownIcon from "./icons/ArrowDown"

export type BaseCard = {
  src : string,
  width: number,
  height: number,
  alt: string
  title: string 
  value: number | undefined
  levels: string | undefined
}

const BaseCardDiagnosis = ({cardProps} : { cardProps: BaseCard}) => {

    if (cardProps.title.match('Heart Rate'))
        return (
            <div className='pl-4 pt-4 bg-red-50 rounded-xl'>
                <Image
                    src={cardProps.src}
                    width={cardProps.width}
                    height={cardProps.height}
                    alt={cardProps.alt}
                />
                <h2 className='pt-4 text-lg font-light'>{cardProps.title}</h2>
                <p className='text-3xl font-bold'>{cardProps.value} bpm</p>
                <div className='flex items-center my-5'>
                        <ArrowDownIcon /> 
                        <p className='ml-1 font-extralight'>{cardProps.levels}</p>
                    </div>
            </div>
        )

    if (cardProps.title.match('Temperature'))
        return (
            <div className='pl-4 pt-4 bg-red-50 rounded-xl'>
                <Image
                    src={cardProps.src}
                    width={cardProps.width}
                    height={cardProps.height}
                    alt={cardProps.alt}
                />
                <h2 className='pt-4 text-lg font-light'>{cardProps.title}</h2>
                <p className='text-3xl font-bold'>{cardProps.value} °F</p>
                <p className='my-5 font-extralight'>{cardProps.levels}</p>
            </div>
        )
    
        return (
            <div className='pl-4 pt-4 bg-blue-50 rounded-xl'>
                <Image
                    src={cardProps.src}
                    width={cardProps.width}
                    height={cardProps.height}
                    alt={cardProps.alt}
                />
                <h2 className='pt-4 text-lg font-light'>{cardProps.title}</h2>
                <p className='text-3xl font-bold'>{cardProps.value} bpm</p>
                <p className='my-5 font-extralight'>{cardProps.levels}</p>
            </div>
        )        
}

export default BaseCardDiagnosis