import { protocolData } from "./sample-data"

export interface PatientData {
  protocolName: string
  patientNumber: number
  planName: string
  totalDose: number
  numFractions: number
  planningApproved: string
  ptvD95: number
  ptvMin: number
  ptvD99: number
  spinalCordD003cc: number
  heartV50Gy: number
  heartMean: number
  lungsGTVV20Gy: number
  lungsGTVV5Gy: number
  lungsGTVMean: number
  esophagusD003cc: number
  esophagusMean: number
  esophagusV60GyPercent: number
  esophagusV60Gy: number
  brachialPlexD003cc: number
  planScore: number
}

export interface DVHData {
  protocolName: string
  patientNumber: number
  planName: string
  totalDose: number
  numFractions: number
  planningApproved: string
  ptvD95: string
  ptvMin: string
  ptvD99: string
  spinalCordD003cc: string
  heartV50Gy: string
  heartMean: string
  lungsGTVV20Gy: string
  lungsGTVV5Gy: string
  lungsGTVMean: string
  esophagusD003cc: string
  esophagusMean: string
  esophagusV60GyPercent: string
  esophagusV60Gy: string
  brachialPlexD003cc: string
}

export interface PatientDataWithDVH {
  percentiles: PatientData
  dvh: DVHData
}

export interface ProtocolItem {
  structureId: string
  dvhObjective: string
  evaluator: string
  variation: string
  priority: string
}

export type ProtocolData = ProtocolItem[]

// Calculate percentiles for a given metric across all patients
export function calculatePercentile(value: number, allValues: number[]): number {
  if (allValues.length === 0) return 0

  const sortedValues = [...allValues].sort((a, b) => a - b)
  const index = sortedValues.findIndex((v) => v >= value)

  if (index === -1) return 100
  if (index === 0) return 0

  return Math.round((index / sortedValues.length) * 100)
}

// Calculate percentiles for a patient based on population data
export function calculatePatientPercentiles(patient: PatientData, population: PatientData[]) {
  const validPopulation = population.filter((p) => p.ptvD95 > 0 && p.planScore > 0)

  return {
    ptvD95: calculatePercentile(
      patient.ptvD95,
      validPopulation.map((p) => p.ptvD95),
    ),
    ptvMin: calculatePercentile(
      patient.ptvMin,
      validPopulation.map((p) => p.ptvMin),
    ),
    ptvD99: calculatePercentile(
      patient.ptvD99,
      validPopulation.map((p) => p.ptvD99),
    ),
    spinalCordD003cc:
      100 -
      calculatePercentile(
        patient.spinalCordD003cc,
        validPopulation.map((p) => p.spinalCordD003cc),
      ), // Lower is better
    heartV50Gy:
      100 -
      calculatePercentile(
        patient.heartV50Gy,
        validPopulation.map((p) => p.heartV50Gy),
      ), // Lower is better
    heartMean:
      100 -
      calculatePercentile(
        patient.heartMean,
        validPopulation.map((p) => p.heartMean),
      ), // Lower is better
    lungsGTVV20Gy:
      100 -
      calculatePercentile(
        patient.lungsGTVV20Gy,
        validPopulation.map((p) => p.lungsGTVV20Gy),
      ), // Lower is better
    lungsGTVV5Gy:
      100 -
      calculatePercentile(
        patient.lungsGTVV5Gy,
        validPopulation.map((p) => p.lungsGTVV5Gy),
      ), // Lower is better
    lungsGTVMean:
      100 -
      calculatePercentile(
        patient.lungsGTVMean,
        validPopulation.map((p) => p.lungsGTVMean),
      ), // Lower is better
    esophagusD003cc:
      100 -
      calculatePercentile(
        patient.esophagusD003cc,
        validPopulation.map((p) => p.esophagusD003cc),
      ), // Lower is better
    esophagusMean:
      100 -
      calculatePercentile(
        patient.esophagusMean,
        validPopulation.map((p) => p.esophagusMean),
      ), // Lower is better
    esophagusV60GyPercent:
      100 -
      calculatePercentile(
        patient.esophagusV60GyPercent,
        validPopulation.map((p) => p.esophagusV60GyPercent),
      ), // Lower is better
    esophagusV60Gy:
      100 -
      calculatePercentile(
        patient.esophagusV60Gy,
        validPopulation.map((p) => p.esophagusV60Gy),
      ), // Lower is better
    brachialPlexD003cc:
      100 -
      calculatePercentile(
        patient.brachialPlexD003cc,
        validPopulation.map((p) => p.brachialPlexD003cc),
      ), // Lower is better
  }
}

// Get protocol data from the sample data
export function getProtocolData(): ProtocolData {
  return protocolData
}

export const parseCSVData = (csvText: string): any => {
  // Basic CSV parsing logic (can be improved)
  const lines = csvText.split("\n")
  const headers = lines[0].split(",")
  const data = []

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(",")
    const row: any = {}
    for (let j = 0; j < headers.length; j++) {
      row[headers[j]] = values[j]
    }
    data.push(row)
  }
  return data
}

export const parseExcelData = async (file: File): Promise<any> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.onload = (e: any) => {
      const data = new Uint8Array(e.target.result)
      // @ts-expect-error
      import("xlsx")
        .then((xlsx) => {
          const workbook = xlsx.read(data, { type: "array" })
          const sheetName = workbook.SheetNames[0]
          const worksheet = workbook.Sheets[sheetName]
          const jsonData = xlsx.utils.sheet_to_json(worksheet)
          resolve(jsonData)
        })
        .catch(reject)
    }

    reader.onerror = (error) => {
      reject(error)
    }

    reader.readAsArrayBuffer(file)
  })
}
