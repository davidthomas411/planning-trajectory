"use client"

import { useState } from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { getProtocolData } from "@/lib/data"
import { cn } from "@/lib/utils"

interface ProtocolTableProps {
  onRowSelect?: (index: number | null) => void
  selectedRow?: number | null
}

export function ProtocolTable({ onRowSelect, selectedRow }: ProtocolTableProps) {
  const [protocol] = useState(getProtocolData())

  const handleRowClick = (index: number) => {
    const newSelection = selectedRow === index ? null : index
    onRowSelect?.(newSelection)
  }

  return (
    <div className="w-full">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="min-w-[120px] whitespace-nowrap">Structure ID</TableHead>
            <TableHead className="min-w-[140px] whitespace-nowrap">DVH Objective</TableHead>
            <TableHead className="min-w-[100px] whitespace-nowrap">Evaluator</TableHead>
            <TableHead className="min-w-[80px] whitespace-nowrap">Variation</TableHead>
            <TableHead className="min-w-[80px] whitespace-nowrap">Priority</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {protocol.map((item, index) => (
            <TableRow
              key={index}
              className={cn(
                "cursor-pointer transition-colors hover:bg-blue-50",
                selectedRow === index && "bg-blue-100 border-l-4 border-l-blue-500",
              )}
              onClick={() => handleRowClick(index)}
            >
              <TableCell className="font-medium whitespace-nowrap">{item.structureId}</TableCell>
              <TableCell className="whitespace-nowrap">{item.dvhObjective}</TableCell>
              <TableCell className="whitespace-nowrap">{item.evaluator}</TableCell>
              <TableCell className="whitespace-nowrap">{item.variation}</TableCell>
              <TableCell className="whitespace-nowrap">{item.priority}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
