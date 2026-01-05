"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { PatientScorecard } from "@/components/patient-scorecard"
import { ProtocolTable } from "@/components/protocol-table"
import { PatientSelector } from "@/components/patient-selector"
import type { DVHData, PatientDataWithDVH } from "@/lib/data"
import { samplePatients, sampleDVHData } from "@/lib/sample-data"

export default function Home() {
  const [combinedData, setCombinedData] = useState<PatientDataWithDVH[]>([])
  const [selectedPatient, setSelectedPatient] = useState<PatientDataWithDVH | null>(null)
  const [selectedProtocolIndex, setSelectedProtocolIndex] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [leftPanelWidth, setLeftPanelWidth] = useState(25) // Percentage width of left panel
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Initialize with sample data
  useEffect(() => {
    const combined: PatientDataWithDVH[] = []

    for (const pData of samplePatients) {
      // Find matching DVH data
      const matchingDVH = sampleDVHData.find(
        (d) => d.patientNumber === pData.patientNumber && d.planName === pData.planName,
      )

      if (matchingDVH) {
        combined.push({
          percentiles: pData,
          dvh: matchingDVH,
        })
      } else {
        // Create a default DVH data object if no match is found
        const defaultDVH: DVHData = {
          protocolName: pData.protocolName,
          patientNumber: pData.patientNumber,
          planName: pData.planName,
          totalDose: pData.totalDose,
          numFractions: pData.numFractions,
          planningApproved: pData.planningApproved,
          ptvD95: `${pData.ptvD95}%`,
          ptvMin: `${pData.ptvMin}%`,
          ptvD99: `${pData.ptvD99} Gy`,
          spinalCordD003cc: `${pData.spinalCordD003cc} Gy`,
          heartV50Gy: `${pData.heartV50Gy}%`,
          heartMean: `${pData.heartMean} Gy`,
          lungsGTVV20Gy: `${pData.lungsGTVV20Gy}%`,
          lungsGTVV5Gy: `${pData.lungsGTVV5Gy}%`,
          lungsGTVMean: `${pData.lungsGTVMean} Gy`,
          esophagusD003cc: `${pData.esophagusD003cc} Gy`,
          esophagusMean: `${pData.esophagusMean} Gy`,
          esophagusV60GyPercent: `${pData.esophagusV60GyPercent}%`,
          esophagusV60Gy: `${pData.esophagusV60Gy} cc`,
          brachialPlexD003cc: `${pData.brachialPlexD003cc} Gy`,
        }
        combined.push({
          percentiles: pData,
          dvh: defaultDVH,
        })
      }
    }

    setCombinedData(combined)

    // Set the first patient as selected by default
    if (combined.length > 0) {
      setSelectedPatient(combined[0])
    }

    setIsLoading(false)
  }, [])

  const handlePatientSelected = (patient: PatientDataWithDVH) => {
    setSelectedPatient(patient)
    // Reset protocol selection when patient changes
    setSelectedProtocolIndex(null)
  }

  const handleProtocolRowSelect = (index: number | null) => {
    setSelectedProtocolIndex(index)
  }

  const handleSectorClick = (index: number | null) => {
    setSelectedProtocolIndex(index)
  }

  // Handle mouse events for resizing
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true)
    e.preventDefault()
  }

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging || !containerRef.current) return

    const containerRect = containerRef.current.getBoundingClientRect()
    const newLeftWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100

    // Constrain between 20% and 60%
    const constrainedWidth = Math.max(20, Math.min(60, newLeftWidth))
    setLeftPanelWidth(constrainedWidth)
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  // Add global mouse event listeners
  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove)
      document.addEventListener("mouseup", handleMouseUp)
      document.body.style.cursor = "col-resize"
      document.body.style.userSelect = "none"

      return () => {
        document.removeEventListener("mousemove", handleMouseMove)
        document.removeEventListener("mouseup", handleMouseUp)
        document.body.style.cursor = ""
        document.body.style.userSelect = ""
      }
    }
  }, [isDragging])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 p-4 md:p-8 flex items-center justify-center">
        <div className="text-xl font-medium">Loading patient data...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4 md:p-8">
      <div className="max-w-[1600px] mx-auto">
        <h1 className="text-3xl font-bold mb-6">Radiation Plan Scoring Dashboard</h1>

        <div ref={containerRef} className="flex gap-0 h-[calc(100vh-120px)]">
          {/* Left Panel - Resizable */}
          <div className="bg-white rounded-lg shadow flex flex-col" style={{ width: `${leftPanelWidth}%` }}>
            <div className="p-6 border-b">
              <h2 className="text-xl font-semibold mb-4">Patient Selection</h2>
              <PatientSelector patients={combinedData} onPatientSelected={handlePatientSelected} />
            </div>

            <div className="p-6 flex-1 overflow-hidden">
              <h2 className="text-xl font-semibold mb-4">Protocol Values</h2>
              <p className="text-sm text-gray-600 mb-3">
                Click a row to highlight the corresponding sector in the daisy plot
              </p>
              <div className="h-full overflow-auto">
                <ProtocolTable onRowSelect={handleProtocolRowSelect} selectedRow={selectedProtocolIndex} />
              </div>
            </div>
          </div>

          {/* Resizer */}
          <div
            className="w-2 bg-gray-200 hover:bg-gray-300 cursor-col-resize flex items-center justify-center group transition-colors"
            onMouseDown={handleMouseDown}
          >
            <div className="w-1 h-8 bg-gray-400 group-hover:bg-gray-500 rounded-full transition-colors"></div>
          </div>

          {/* Right Panel */}
          <div className="bg-white rounded-lg shadow flex-1" style={{ width: `${100 - leftPanelWidth - 0.2}%` }}>
            <div className="p-6 h-full">
              <h2 className="text-xl font-semibold mb-4">Patient Scorecard</h2>
              {selectedPatient ? (
                <div className="h-full overflow-auto">
                  <PatientScorecard
                    patientData={selectedPatient}
                    populationData={combinedData}
                    selectedProtocolIndex={selectedProtocolIndex}
                    onSectorClick={handleSectorClick}
                  />
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">Select a patient to view scorecard</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
