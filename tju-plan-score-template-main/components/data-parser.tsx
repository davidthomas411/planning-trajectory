"use client"

import { useEffect, useState } from "react"
import type { PatientData } from "@/lib/data"

interface DataParserProps {
  onDataLoaded: (patients: PatientData[]) => void
}

export function DataParser({ onDataLoaded }: DataParserProps) {
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [debugInfo, setDebugInfo] = useState<string>("")

  useEffect(() => {
    const loadPatientData = async () => {
      try {
        console.log("Attempting to load patient data...")
        const response = await fetch("/data/patient-data.txt")

        if (!response.ok) {
          throw new Error(`Failed to load patient data: ${response.status} ${response.statusText}`)
        }

        const text = await response.text()
        console.log(`Loaded text data, length: ${text.length} characters`)
        setDebugInfo(`Loaded ${text.length} characters`)

        if (!text || text.length < 100) {
          throw new Error("Data file appears to be empty or too small")
        }

        // Split by lines and filter empty lines
        const lines = text.split("\n").filter((line) => line.trim().length > 0)
        console.log(`Found ${lines.length} lines in the data`)
        setDebugInfo((prev) => `${prev}, ${lines.length} lines found`)

        // Skip header line
        const dataLines = lines.slice(1)

        const patients: PatientData[] = []

        for (const line of dataLines) {
          // Handle tab-separated values
          const columns = line.split("\t").map((col) => col.trim())

          // Skip lines with insufficient data
          if (columns.length < 21) {
            console.log(`Skipping line with insufficient columns: ${columns.length}`, line)
            continue
          }

          // Parse the data according to the column structure
          const patient: PatientData = {
            protocolName: columns[0] || "",
            patientNumber: Number.parseInt(columns[1], 10) || 0,
            planName: columns[2] || "",
            totalDose: Number.parseFloat(columns[3]) || 0,
            numFractions: Number.parseInt(columns[4], 10) || 0,
            planningApproved: columns[5] || "",
            ptvD95: Number.parseFloat(columns[6]) || 0,
            ptvMin: Number.parseFloat(columns[7]) || 0,
            ptvD99: Number.parseFloat(columns[8]) || 0,
            spinalCordD003cc: Number.parseFloat(columns[9]) || 0,
            heartV50Gy: Number.parseFloat(columns[10]) || 0,
            heartMean: Number.parseFloat(columns[11]) || 0,
            lungsGTVV20Gy: Number.parseFloat(columns[12]) || 0,
            lungsGTVV5Gy: Number.parseFloat(columns[13]) || 0,
            lungsGTVMean: Number.parseFloat(columns[14]) || 0,
            esophagusD003cc: Number.parseFloat(columns[15]) || 0,
            esophagusMean: Number.parseFloat(columns[16]) || 0,
            esophagusV60GyPercent: Number.parseFloat(columns[17]) || 0,
            esophagusV60Gy: Number.parseFloat(columns[18]) || 0,
            brachialPlexD003cc: Number.parseFloat(columns[19]) || 0,
            planScore: Number.parseFloat(columns[20]) || 0,
          }

          // Only include patients with valid plan scores and patient numbers
          if (patient.planScore > 0 && patient.patientNumber > 0) {
            patients.push(patient)
          }
        }

        console.log(`Successfully parsed ${patients.length} patients`)
        setDebugInfo((prev) => `${prev}, ${patients.length} patients parsed`)

        if (patients.length === 0) {
          throw new Error("No valid patient data found in the file")
        }

        onDataLoaded(patients)
      } catch (err) {
        console.error("Error loading patient data:", err)
        setError(err instanceof Error ? err.message : "Failed to load data")
        setDebugInfo((prev) => `${prev}, ERROR: ${err instanceof Error ? err.message : "Unknown error"}`)
      } finally {
        setIsLoading(false)
      }
    }

    loadPatientData()
  }, [onDataLoaded])

  if (isLoading) {
    return <div className="text-center py-4">Loading patient database...</div>
  }

  if (error) {
    return (
      <div className="space-y-2">
        <div className="text-red-600 text-center py-4">Error: {error}</div>
        <div className="text-sm text-gray-600 bg-gray-100 p-2 rounded">Debug info: {debugInfo}</div>
      </div>
    )
  }

  return null
}
