"use client"

import { useEffect, useState } from "react"
import type { DVHData } from "@/lib/data"

interface DVHDataParserProps {
  onDataLoaded: (dvhData: DVHData[]) => void
}

export function DVHDataParser({ onDataLoaded }: DVHDataParserProps) {
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [debugInfo, setDebugInfo] = useState<string>("")

  useEffect(() => {
    const loadDVHData = async () => {
      try {
        console.log("Attempting to load DVH data...")
        const response = await fetch("/data/dvh-data.txt")

        if (!response.ok) {
          throw new Error(`Failed to load DVH data: ${response.status} ${response.statusText}`)
        }

        const text = await response.text()
        console.log(`Loaded DVH text data, length: ${text.length} characters`)
        setDebugInfo(`Loaded ${text.length} characters`)

        if (!text || text.length < 100) {
          throw new Error("DVH data file appears to be empty or too small")
        }

        // Split by lines and filter empty lines
        const lines = text.split("\n").filter((line) => line.trim().length > 0)
        console.log(`Found ${lines.length} lines in the DVH data`)
        setDebugInfo((prev) => `${prev}, ${lines.length} lines found`)

        // Skip header line
        const dataLines = lines.slice(1)

        const dvhDataArray: DVHData[] = []

        for (let lineIndex = 0; lineIndex < dataLines.length; lineIndex++) {
          const line = dataLines[lineIndex]

          // Handle tab-separated values
          const columns = line.split("\t").map((col) => col.trim())

          console.log(`Line ${lineIndex + 1}: ${columns.length} columns`, columns.slice(0, 10))

          // Skip lines with insufficient data - the DVH file has many columns
          if (columns.length < 15) {
            console.log(`Skipping line ${lineIndex + 1} with insufficient columns: ${columns.length}`)
            continue
          }

          // Check if this line contains "Achieved" data (should be in column 6, index 6)
          if (columns.length > 6 && columns[6] !== "Achieved") {
            console.log(`Skipping line ${lineIndex + 1} - not an 'Achieved' record: ${columns[6]}`)
            continue
          }

          // Parse the data according to the column structure
          // Based on the header: Protocol Name, Patient Name, Plan Name, Total Dose, Num Fractions, PlanningApproved, Structure ID DVH Objective, then the actual values
          const dvhData: DVHData = {
            protocolName: columns[0] || "",
            patientNumber: Number.parseInt(columns[1], 10) || 0,
            planName: columns[2] || "",
            totalDose: Number.parseFloat(columns[3]) || 0,
            numFractions: Number.parseInt(columns[4], 10) || 0,
            planningApproved: columns[5] || "",
            ptvD95: columns[7] || "",
            ptvMin: columns[8] || "",
            ptvD99: columns[9] || "",
            spinalCordD003cc: columns[10] || "",
            heartV50Gy: columns[11] || "",
            heartMean: columns[12] || "",
            lungsGTVV20Gy: columns[13] || "",
            lungsGTVV5Gy: columns[14] || "",
            lungsGTVMean: columns[15] || "",
            esophagusD003cc: columns[16] || "",
            esophagusMean: columns[17] || "",
            esophagusV60GyPercent: columns[18] || "",
            esophagusV60Gy: columns[19] || "",
            brachialPlexD003cc: columns[20] || "",
          }

          // Only include patients with valid data
          if (dvhData.patientNumber > 0 && dvhData.planName) {
            dvhDataArray.push(dvhData)
            console.log(`Added DVH record for patient ${dvhData.patientNumber} - ${dvhData.planName}`)
          } else {
            console.log(`Skipping invalid DVH record: patient ${dvhData.patientNumber}, plan ${dvhData.planName}`)
          }
        }

        console.log(`Successfully parsed ${dvhDataArray.length} DVH records`)
        setDebugInfo((prev) => `${prev}, ${dvhDataArray.length} DVH records parsed`)

        if (dvhDataArray.length === 0) {
          console.log("No valid DVH data found, using sample data")
          // Don't throw an error, just use empty array and let the app use default DVH data
          onDataLoaded([])
        } else {
          onDataLoaded(dvhDataArray)
        }
      } catch (err) {
        console.error("Error loading DVH data:", err)
        setError(err instanceof Error ? err.message : "Failed to load DVH data")
        setDebugInfo((prev) => `${prev}, ERROR: ${err instanceof Error ? err.message : "Unknown error"}`)
        // Don't fail completely, just use empty array
        onDataLoaded([])
      } finally {
        setIsLoading(false)
      }
    }

    loadDVHData()
  }, [onDataLoaded])

  if (isLoading) {
    return <div className="text-center py-4">Loading DVH data...</div>
  }

  if (error) {
    return (
      <div className="space-y-2">
        <div className="text-yellow-600 text-center py-4">Warning: {error}</div>
        <div className="text-sm text-gray-600 bg-gray-100 p-2 rounded">Debug info: {debugInfo}</div>
        <div className="text-sm text-blue-600">Using default DVH values instead.</div>
      </div>
    )
  }

  return null
}
