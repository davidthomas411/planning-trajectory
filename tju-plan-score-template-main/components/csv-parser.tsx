"use client"

import { useEffect, useState } from "react"
import type { PatientData } from "@/lib/data"

interface CSVParserProps {
  fileUrl: string
  onDataParsed: (data: PatientData[]) => void
  onError: (error: string) => void
}

export function CSVParser({ fileUrl, onDataParsed, onError }: CSVParserProps) {
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchAndParseCSV = async () => {
      try {
        const response = await fetch(fileUrl)
        if (!response.ok) {
          throw new Error(`Failed to fetch CSV: ${response.status}`)
        }

        const text = await response.text()
        const lines = text.split("\n")

        // Skip header row and parse data
        const headers = lines[0].split(",")
        const patientData: PatientData[] = []

        for (let i = 1; i < lines.length; i++) {
          if (!lines[i].trim()) continue

          const values = lines[i].split(",")
          if (values.length !== headers.length) {
            console.warn(`Line ${i} has incorrect number of values`)
            continue
          }

          // Map CSV values to PatientData structure
          // This is a simplified example - would need to be adapted to actual CSV structure
          const patient: Partial<PatientData> = {
            ptvD95: Number.parseFloat(values[0]),
            ptvMin: Number.parseFloat(values[1]),
            ptvD99: Number.parseFloat(values[2]),
            spinalCordD003cc: Number.parseFloat(values[3]),
            heartV50Gy: Number.parseFloat(values[4]),
            heartMean: Number.parseFloat(values[5]),
            lungsGTVV20Gy: Number.parseFloat(values[6]),
            lungsGTVV5Gy: Number.parseFloat(values[7]),
            lungsGTVMean: Number.parseFloat(values[8]),
            esophagusD003cc: Number.parseFloat(values[9]),
            esophagusMean: Number.parseFloat(values[10]),
            esophagusV60GyPercent: Number.parseFloat(values[11]),
            esophagusV60Gy: Number.parseFloat(values[12]),
            brachialPlexD003cc: Number.parseFloat(values[13]),
            planScore: Number.parseFloat(values[14]),
          }

          patientData.push(patient as PatientData)
        }

        onDataParsed(patientData)
      } catch (error) {
        console.error("Error parsing CSV:", error)
        onError(error instanceof Error ? error.message : "Failed to parse CSV file")
      } finally {
        setIsLoading(false)
      }
    }

    fetchAndParseCSV()
  }, [fileUrl, onDataParsed, onError])

  if (isLoading) {
    return <div className="text-center py-4">Loading CSV data...</div>
  }

  return null
}
