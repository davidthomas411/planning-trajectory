"use client"

import { useState } from "react"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { PatientDataWithDVH } from "@/lib/data"

interface PatientSelectorProps {
  patients: PatientDataWithDVH[]
  onPatientSelected: (patient: PatientDataWithDVH) => void
}

export function PatientSelector({ patients, onPatientSelected }: PatientSelectorProps) {
  const [selectedPatientId, setSelectedPatientId] = useState("")

  const handlePatientSelect = (patientId: string) => {
    setSelectedPatientId(patientId)
    const patient = patients.find((p) => `${p.percentiles.patientNumber}-${p.percentiles.planName}` === patientId)
    if (patient) {
      onPatientSelected(patient)
    }
  }

  // Group patients by approval status for better organization
  const approvedPlans = patients.filter((p) => p.percentiles.planningApproved === "PlanningApproved")
  const unapprovedPlans = patients.filter((p) => p.percentiles.planningApproved === "UnApproved")

  return (
    <div className="space-y-4">
      <div className="grid w-full items-center gap-1.5">
        <Label htmlFor="patientSelect">Select Patient Plan</Label>
        <Select value={selectedPatientId} onValueChange={handlePatientSelect}>
          <SelectTrigger>
            <SelectValue placeholder="Select a patient plan" />
          </SelectTrigger>
          <SelectContent className="max-h-60">
            <div className="px-2 py-1 text-sm font-medium text-gray-500">Approved Plans</div>
            {approvedPlans.map((patient) => (
              <SelectItem
                key={`${patient.percentiles.patientNumber}-${patient.percentiles.planName}`}
                value={`${patient.percentiles.patientNumber}-${patient.percentiles.planName}`}
              >
                Patient {patient.percentiles.patientNumber} - {patient.percentiles.planName} (Score:{" "}
                {patient.percentiles.planScore}%)
              </SelectItem>
            ))}
            <div className="px-2 py-1 text-sm font-medium text-gray-500 border-t mt-1 pt-2">Unapproved Plans</div>
            {unapprovedPlans.map((patient) => (
              <SelectItem
                key={`${patient.percentiles.patientNumber}-${patient.percentiles.planName}`}
                value={`${patient.percentiles.patientNumber}-${patient.percentiles.planName}`}
              >
                Patient {patient.percentiles.patientNumber} - {patient.percentiles.planName} (Score:{" "}
                {patient.percentiles.planScore}%)
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs text-gray-500">
          Sample database contains {patients.length} patient plans ({approvedPlans.length} approved,{" "}
          {unapprovedPlans.length} unapproved)
        </p>
      </div>
    </div>
  )
}
