"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { parseCSVData, parseExcelData } from "@/lib/data"

export function FileUpload() {
  const [file, setFile] = useState<File | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setIsLoading(true)
    setError(null)

    try {
      // In a real implementation, this would process the file
      // and update the application state with the parsed data
      if (file.name.endsWith(".csv")) {
        const text = await file.text()
        const data = parseCSVData(text)
        console.log("Parsed CSV data:", data)
      } else if (file.name.endsWith(".xlsx") || file.name.endsWith(".xls")) {
        const data = await parseExcelData(file)
        console.log("Parsed Excel data:", data)
      } else {
        setError("Unsupported file format. Please upload a CSV or Excel file.")
      }
    } catch (err) {
      setError("Failed to process file. Please check the file format.")
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid w-full items-center gap-1.5">
        <Label htmlFor="patientFile">Upload Patient Data</Label>
        <Input id="patientFile" type="file" accept=".csv,.xlsx,.xls" onChange={handleFileChange} />
        <p className="text-xs text-gray-500">Upload CSV or Excel file with patient radiation plan data</p>
      </div>

      {error && <div className="bg-red-50 text-red-700 p-2 rounded-md text-sm">{error}</div>}

      <Button className="w-full" onClick={handleUpload} disabled={!file || isLoading}>
        {isLoading ? "Processing..." : "Upload and Process"}
      </Button>
    </div>
  )
}
