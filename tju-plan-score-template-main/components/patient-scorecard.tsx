"use client"

import { useEffect, useRef } from "react"
import * as d3 from "d3"
import { type PatientDataWithDVH, type ProtocolData, getProtocolData } from "@/lib/data"

interface PatientScorecardProps {
  patientData: PatientDataWithDVH
  populationData: PatientDataWithDVH[]
  selectedProtocolIndex?: number | null
  onSectorClick?: (index: number | null) => void
}

export function PatientScorecard({
  patientData,
  populationData,
  selectedProtocolIndex,
  onSectorClick,
}: PatientScorecardProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const protocol = getProtocolData()

  useEffect(() => {
    if (!svgRef.current) return

    renderDaisyPlot(svgRef.current, patientData, populationData, protocol, selectedProtocolIndex, onSectorClick)
  }, [patientData, populationData, protocol, selectedProtocolIndex, onSectorClick])

  // Function to determine plan acceptability and color based on score - updated thresholds
  const getPlanAcceptability = (score: number) => {
    if (score < 35) {
      return {
        category: "Unacceptable",
        color: "#dc2626", // Red 600
        bgColor: "#fef2f2", // Red 50
        borderColor: "#dc2626",
      }
    } else if (score < 75) {
      return {
        category: "Minor Edits Recommended",
        color: "#d97706", // Amber 600
        bgColor: "#fffbeb", // Amber 50
        borderColor: "#d97706",
      }
    } else {
      return {
        category: "Acceptable",
        color: "#16a34a", // Green 600
        bgColor: "#f0fdf4", // Green 50
        borderColor: "#16a34a",
      }
    }
  }

  const planAcceptability = getPlanAcceptability(patientData.percentiles.planScore)

  return (
    <div className="flex flex-col items-center">
      <div className="mb-4 text-center">
        <h3 className="text-lg font-semibold">
          Patient {patientData.percentiles.patientNumber} - {patientData.percentiles.planName}
        </h3>
        <p className="text-sm text-gray-600">
          {patientData.percentiles.totalDose} cGy in {patientData.percentiles.numFractions} fractions | Status:{" "}
          {patientData.percentiles.planningApproved}
        </p>

        {/* Plan Acceptability Indicator - without title */}
        <div
          className="mt-2 inline-block px-3 py-1 rounded-full text-sm font-medium"
          style={{
            backgroundColor: planAcceptability.bgColor,
            color: planAcceptability.color,
            border: `2px solid ${planAcceptability.borderColor}`,
          }}
        >
          {planAcceptability.category}
        </div>
      </div>

      <div className="w-full max-w-4xl aspect-square">
        <svg ref={svgRef} width="100%" height="100%" viewBox="0 0 900 900" style={{ cursor: "pointer" }} />
      </div>

      <div className="mt-6 w-full max-w-4xl">
        <h3 className="text-lg font-medium mb-2">Absolute Percentile Performance vs Population</h3>

        {/* Plan Score Ranges - updated thresholds */}
        <div className="mb-4 p-3 bg-gray-50 rounded-md">
          <h4 className="text-sm font-medium mb-2">Plan Score Recommendations:</h4>
          <div className="flex flex-wrap gap-4">
            <div className="flex items-center">
              <div className="w-6 h-4 bg-red-600 mr-2 rounded"></div>
              <span className="text-sm">Unacceptable (&lt;35%)</span>
            </div>
            <div className="flex items-center">
              <div className="w-6 h-4 bg-amber-600 mr-2 rounded"></div>
              <span className="text-sm">Minor Edits Recommended (35-75%)</span>
            </div>
            <div className="flex items-center">
              <div className="w-6 h-4 bg-green-600 mr-2 rounded"></div>
              <span className="text-sm">Acceptable (&gt;75%)</span>
            </div>
          </div>
        </div>

        {selectedProtocolIndex !== null && selectedProtocolIndex !== undefined && (
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-md">
            <p className="text-sm font-medium text-blue-800">
              Selected: {protocol[selectedProtocolIndex]?.structureId} - {protocol[selectedProtocolIndex]?.dvhObjective}
              <span className="ml-2 px-2 py-1 bg-blue-200 rounded text-xs">
                Priority {protocol[selectedProtocolIndex]?.priority}
              </span>
            </p>
          </div>
        )}
        <div className="mt-2 text-sm text-gray-600">
          Click on sectors in the daisy plot or rows in the protocol table to highlight constraints
        </div>
      </div>
    </div>
  )
}

function renderDaisyPlot(
  svg: SVGSVGElement,
  patientData: PatientDataWithDVH,
  populationData: PatientDataWithDVH[],
  protocol: ProtocolData,
  selectedProtocolIndex?: number | null,
  onSectorClick?: (index: number | null) => void,
) {
  // Clear previous content
  d3.select(svg).selectAll("*").remove()

  const width = 900
  const height = 900
  const margin = { top: 120, right: 120, bottom: 120, left: 120 }
  const innerRadius = 140
  const outerRadius = Math.min(width, height) / 2 - margin.top

  const svgEl = d3.select(svg)

  // Create the main group element and center it
  const g = svgEl.append("g").attr("transform", `translate(${width / 2}, ${height / 2})`)

  // Add legends in the corners using white space

  // Top Left - Priority Legend (updated with transparency examples)
  const topLeftLegend = svgEl.append("g").attr("transform", "translate(20, 20)")

  topLeftLegend
    .append("text")
    .attr("x", 0)
    .attr("y", 0)
    .attr("font-size", "14px")
    .attr("font-weight", "bold")
    .attr("fill", "#374151")
    .text("Priority Levels:")

  const priorityItems = [
    { priority: "1", label: "Priority 1 (Critical) - Full opacity", opacity: 1.0, color: "#bbf7d0" },
    { priority: "2", label: "Priority 2 (Important) - Medium opacity", opacity: 0.7, color: "#bbf7d0" },
    { priority: "3", label: "Priority 3 (Standard) - Light opacity", opacity: 0.4, color: "#bbf7d0" },
    { priority: "Report", label: "Report Only - White", opacity: 1.0, color: "white" },
  ]

  priorityItems.forEach((item, i) => {
    const yPos = 20 + i * 20

    topLeftLegend
      .append("rect")
      .attr("x", 0)
      .attr("y", yPos - 8)
      .attr("width", 16)
      .attr("height", 12)
      .attr("fill", item.color)
      .attr("opacity", item.opacity)
      .attr("stroke", "#d1d5db")

    topLeftLegend
      .append("text")
      .attr("x", 22)
      .attr("y", yPos)
      .attr("font-size", "12px")
      .attr("fill", "#374151")
      .text(item.label)
  })

  // Top Right - Performance Legend (moved further right)
  const topRightLegend = svgEl.append("g").attr("transform", "translate(720, 20)")

  topRightLegend
    .append("text")
    .attr("x", 0)
    .attr("y", 0)
    .attr("font-size", "14px")
    .attr("font-weight", "bold")
    .attr("fill", "#374151")
    .text("Percentile Performance:")

  const performanceItems = [
    { color: "#dc2626", label: "0-20th percentile (poor)" },
    { color: "#f87171", label: "20-40th percentile (below avg)" },
    { color: "#fde047", label: "40-60th percentile (average)" },
    { color: "#bbf7d0", label: "60-80th percentile (good)" },
    { color: "#4ade80", label: "80-90th percentile (very good)" },
    { color: "#16a34a", label: "90-95th percentile (excellent)" },
    { color: "#065f46", label: "95-100th percentile (outstanding)" },
  ]

  performanceItems.forEach((item, i) => {
    const yPos = 20 + i * 18

    topRightLegend
      .append("rect")
      .attr("x", 0)
      .attr("y", yPos - 8)
      .attr("width", 16)
      .attr("height", 12)
      .attr("fill", item.color)

    topRightLegend
      .append("text")
      .attr("x", 22)
      .attr("y", yPos)
      .attr("font-size", "11px")
      .attr("fill", "#374151")
      .text(item.label)
  })

  // Bottom Left - Dot Legend
  const bottomLeftLegend = svgEl.append("g").attr("transform", `translate(20, ${height - 80})`)

  bottomLeftLegend
    .append("text")
    .attr("x", 0)
    .attr("y", 0)
    .attr("font-size", "14px")
    .attr("font-weight", "bold")
    .attr("fill", "#374151")
    .text("Dot Indicators:")

  const dotItems = [
    { color: "black", label: "Achieved Value" },
    { color: "#888", label: "Requested Value", opacity: 0.7 },
  ]

  dotItems.forEach((item, i) => {
    const yPos = 20 + i * 20

    bottomLeftLegend
      .append("circle")
      .attr("cx", 8)
      .attr("cy", yPos - 4)
      .attr("r", 5)
      .attr("fill", item.color)
      .attr("opacity", item.opacity || 1)

    bottomLeftLegend
      .append("text")
      .attr("x", 22)
      .attr("y", yPos)
      .attr("font-size", "12px")
      .attr("fill", "#374151")
      .text(item.label)
  })

  // Function to get plan score color based on acceptability - updated thresholds
  const getPlanScoreColor = (score: number) => {
    if (score < 35) {
      return "#dc2626" // Red 600 - Unacceptable
    } else if (score < 75) {
      return "#d97706" // Amber 600 - Minor Edits
    } else {
      return "#16a34a" // Green 600 - Acceptable
    }
  }

  // Add title in the center with color based on plan score
  const planScoreColor = getPlanScoreColor(patientData.percentiles.planScore)
  g.append("text")
    .attr("text-anchor", "middle")
    .attr("dy", "0.35em")
    .attr("font-size", "52px")
    .attr("font-weight", "bold")
    .attr("fill", planScoreColor)
    .text(`${patientData.percentiles.planScore}%`)

  // Use the eCDF percentile values directly from the patient data
  const percentiles = patientData.percentiles

  // Create metrics array with eCDF percentile values in protocol order
  const metrics = [
    // PTV metrics first
    {
      name: "PTV D95%",
      value: patientData.dvh.ptvD95,
      percentile: percentiles.ptvD95,
      requested: 95,
      priority: protocol[0]?.priority || "3",
      unit: "",
    },
    {
      name: "PTV Min%",
      value: patientData.dvh.ptvMin,
      percentile: percentiles.ptvMin,
      requested: 90,
      priority: protocol[1]?.priority || "3",
      unit: "",
    },
    {
      name: "PTV D99%",
      value: patientData.dvh.ptvD99,
      percentile: percentiles.ptvD99,
      requested: 95,
      priority: protocol[2]?.priority || "3",
      unit: "",
    },
    // Then spinal cord
    {
      name: "SpinalCord D0.03cc",
      value: patientData.dvh.spinalCordD003cc,
      percentile: percentiles.spinalCordD003cc,
      requested: 80,
      priority: protocol[3]?.priority || "3",
      unit: "",
    },
    // Heart metrics
    {
      name: "Heart V50Gy",
      value: patientData.dvh.heartV50Gy,
      percentile: percentiles.heartV50Gy,
      requested: 70,
      priority: protocol[4]?.priority || "3",
      unit: "",
    },
    {
      name: "Heart Mean",
      value: patientData.dvh.heartMean,
      percentile: percentiles.heartMean,
      requested: 74,
      priority: protocol[5]?.priority || "3",
      unit: "",
    },
    // Lungs metrics
    {
      name: "Lungs-GTV V20Gy",
      value: patientData.dvh.lungsGTVV20Gy,
      percentile: percentiles.lungsGTVV20Gy,
      requested: 65,
      priority: protocol[6]?.priority || "3",
      unit: "",
    },
    {
      name: "Lungs-GTV V5Gy",
      value: patientData.dvh.lungsGTVV5Gy,
      percentile: percentiles.lungsGTVV5Gy,
      requested: 40,
      priority: protocol[7]?.priority || "3",
      unit: "",
    },
    {
      name: "Lungs-GTV Mean",
      value: patientData.dvh.lungsGTVMean,
      percentile: percentiles.lungsGTVMean,
      requested: 80,
      priority: protocol[8]?.priority || "3",
      unit: "",
    },
    // Esophagus metrics
    {
      name: "Esophagus D0.03cc",
      value: patientData.dvh.esophagusD003cc,
      percentile: percentiles.esophagusD003cc,
      requested: 20,
      priority: protocol[9]?.priority || "3",
      unit: "",
    },
    {
      name: "Esophagus Mean",
      value: patientData.dvh.esophagusMean,
      percentile: percentiles.esophagusMean,
      requested: 66,
      priority: protocol[10]?.priority || "3",
      unit: "",
    },
    {
      name: "Esophagus V60Gy%",
      value: patientData.dvh.esophagusV60GyPercent,
      percentile: percentiles.esophagusV60GyPercent,
      requested: 67,
      priority: protocol[11]?.priority || "3",
      unit: "",
    },
    {
      name: "Esophagus V60Gy",
      value: patientData.dvh.esophagusV60Gy,
      percentile: percentiles.esophagusV60Gy,
      requested: 83,
      priority: protocol[12]?.priority || "3",
      unit: "",
    },
    // Brachial plexus
    {
      name: "BrachialPlex D0.03cc",
      value: patientData.dvh.brachialPlexD003cc,
      percentile: percentiles.brachialPlexD003cc,
      requested: 34,
      priority: protocol[13]?.priority || "3",
      unit: "",
    },
  ]

  // Calculate angle for each metric - start from top-left (315 degrees or -π/4)
  const angleStep = (2 * Math.PI) / metrics.length
  const startAngle = -Math.PI / 4 // Start at top-left (315 degrees)

  // Draw concentric circles for reference
  const circles = [0, 25, 50, 75, 100]
  circles.forEach((circle) => {
    const radius = innerRadius + (outerRadius - innerRadius) * (circle / 100)
    g.append("circle")
      .attr("cx", 0)
      .attr("cy", 0)
      .attr("r", radius)
      .attr("fill", "none")
      .attr("stroke", "#ddd")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "2,2")

    // Add percentile labels
    if (circle > 0) {
      g.append("text")
        .attr("x", 5)
        .attr("y", -radius + 5)
        .attr("dy", "-0.3em")
        .attr("text-anchor", "start")
        .attr("font-size", "12px")
        .attr("fill", "#999")
        .text(`${circle}%`)
    }
  })

  // Function to get color and opacity based on priority
  const getPriorityStyle = (priority: string, baseColor: string) => {
    switch (priority) {
      case "1":
        return { color: baseColor, opacity: 1.0 } // Full opacity for Priority 1
      case "2":
        return { color: baseColor, opacity: 0.7 } // Medium opacity for Priority 2
      case "3":
        return { color: baseColor, opacity: 0.4 } // Light opacity for Priority 3
      case "Report":
        return { color: "white", opacity: 1.0 } // White for Report
      default:
        return { color: baseColor, opacity: 0.4 }
    }
  }

  // Helper function to estimate text width based on font size and content
  const estimateTextWidth = (text: string, fontSize: number) => {
    // Average character width in pixels (approximate)
    const avgCharWidth = fontSize * 0.6
    return text.length * avgCharWidth + 20 // Add padding
  }

  // Draw each metric bar
  metrics.forEach((metric, i) => {
    const angle = startAngle + i * angleStep
    const barHeight = (outerRadius - innerRadius) * (metric.percentile / 100)
    const barWidth = angleStep * 0.7 // Back to 70% width for gaps

    // Calculate difference between achieved and requested percentile
    const diff = metric.percentile - metric.requested

    // Red to green color scheme - equal 20% bins
    let baseColor
    if (metric.percentile < 20)
      baseColor = "#dc2626" // Red 600 - 0-20th percentile
    else if (metric.percentile < 40)
      baseColor = "#f87171" // Red 400 - 20-40th percentile
    else if (metric.percentile < 60)
      baseColor = "#fde047" // Yellow 300 - 40-60th percentile
    else if (metric.percentile < 80)
      baseColor = "#bbf7d0" // Green 200 - 60-80th percentile
    else if (metric.percentile < 90)
      baseColor = "#4ade80" // Green 400 - 80-90th percentile
    else if (metric.percentile < 95)
      baseColor = "#16a34a" // Green 600 - 90-95th percentile
    else baseColor = "#065f46" // Green 800 - 95-100th percentile

    // Get color and opacity based on priority
    const style = getPriorityStyle(metric.priority, baseColor)

    // Check if this sector should be highlighted
    const isSelected = selectedProtocolIndex === i
    const strokeWidth = isSelected ? 3 : 1 // Always have black outline
    const strokeColor = isSelected ? "#2563eb" : "#000000" // Black outline, blue when selected

    // Create arc for the bar - with gaps between sectors
    const arc = d3
      .arc()
      .innerRadius(innerRadius)
      .outerRadius(innerRadius + barHeight)
      .startAngle(angle - barWidth / 2)
      .endAngle(angle + barWidth / 2)

    // Draw the bar with click handler
    const barPath = g
      .append("path")
      .attr("d", arc as any)
      .attr("fill", style.color)
      .attr("opacity", style.opacity)
      .attr("stroke", strokeColor)
      .attr("stroke-width", strokeWidth)
      .style("cursor", "pointer")
      .on("click", (event) => {
        event.stopPropagation()
        const newSelection = selectedProtocolIndex === i ? null : i
        onSectorClick?.(newSelection)
      })
      .on("mouseover", function () {
        if (!isSelected) {
          d3.select(this).attr("stroke", "#666").attr("stroke-width", 2)
        }
      })
      .on("mouseout", function () {
        if (!isSelected) {
          d3.select(this).attr("stroke", "#000000").attr("stroke-width", 1)
        }
      })

    // Add achieved value dot
    const dotRadius = isSelected ? 7 : 5
    const dotAngle = angle
    const dotRadius2 = innerRadius + barHeight
    g.append("circle")
      .attr("cx", dotRadius2 * Math.sin(dotAngle))
      .attr("cy", -dotRadius2 * Math.cos(dotAngle))
      .attr("r", dotRadius)
      .attr("fill", isSelected ? "#1d4ed8" : "black")
      .attr("stroke", isSelected ? "white" : "none")
      .attr("stroke-width", isSelected ? 2 : 0)
      .style("pointer-events", "none") // Don't interfere with sector clicks

    // Add requested value dot (gray)
    const reqHeight = (outerRadius - innerRadius) * (metric.requested / 100)
    const reqRadius = innerRadius + reqHeight
    g.append("circle")
      .attr("cx", reqRadius * Math.sin(dotAngle))
      .attr("cy", -reqRadius * Math.cos(dotAngle))
      .attr("r", dotRadius)
      .attr("fill", isSelected ? "#64748b" : "#888")
      .attr("opacity", 0.7)
      .attr("stroke", isSelected ? "white" : "none")
      .attr("stroke-width", isSelected ? 1 : 0)
      .style("pointer-events", "none") // Don't interfere with sector clicks

    // Add value label
    const labelRadius = outerRadius + 30
    const labelX = labelRadius * Math.sin(angle)
    const labelY = -labelRadius * Math.cos(angle)

    // Create a label group
    const labelGroup = g.append("g").attr("transform", `translate(${labelX}, ${labelY})`)

    // Add text background for better readability
    const label = `${metric.name}`
    const fontSize = isSelected ? 13 : 12
    const labelWidth = estimateTextWidth(label, fontSize)
    const labelHeight = 24

    labelGroup
      .append("rect")
      .attr("x", -labelWidth / 2)
      .attr("y", -labelHeight / 2)
      .attr("width", labelWidth)
      .attr("height", labelHeight)
      .attr("fill", isSelected ? "#dbeafe" : "white")
      .attr("opacity", 0.9)
      .attr("rx", 4)
      .attr("stroke", isSelected ? "#2563eb" : "#ccc")
      .attr("stroke-width", isSelected ? 2 : 1)
      .style("pointer-events", "none") // Don't interfere with sector clicks

    // Add the text label
    labelGroup
      .append("text")
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("font-size", `${fontSize}px`)
      .attr("font-weight", isSelected ? "bold" : "normal")
      .attr("fill", isSelected ? "#1d4ed8" : "black")
      .text(label)
      .style("pointer-events", "none") // Don't interfere with sector clicks

    // Add value tooltip showing both eCDF percentile and DVH value
    const tooltipX = (innerRadius + barHeight) * Math.sin(angle)
    const tooltipY = -(innerRadius + barHeight) * Math.cos(angle)

    // Calculate tooltip width based on content
    const tooltipText = `${metric.percentile}%, ${metric.value}`
    const tooltipWidth = Math.max(90, estimateTextWidth(tooltipText, 10) + 10)

    g.append("rect")
      .attr("x", tooltipX - tooltipWidth / 2)
      .attr("y", tooltipY - 12)
      .attr("width", tooltipWidth)
      .attr("height", 24)
      .attr("fill", isSelected ? "#dbeafe" : "white")
      .attr("stroke", isSelected ? "#2563eb" : "#ccc")
      .attr("stroke-width", isSelected ? 2 : 1)
      .attr("rx", 4)
      .style("pointer-events", "none") // Don't interfere with sector clicks

    g.append("text")
      .attr("x", tooltipX)
      .attr("y", tooltipY)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("font-size", isSelected ? "11px" : "10px")
      .attr("font-weight", isSelected ? "bold" : "normal")
      .attr("fill", isSelected ? "#1d4ed8" : "black")
      .text(tooltipText)
      .style("pointer-events", "none") // Don't interfere with sector clicks

    // Add line from center to label (only for selected)
    if (isSelected) {
      g.append("line")
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", outerRadius * Math.sin(angle))
        .attr("y2", -outerRadius * Math.cos(angle))
        .attr("stroke", "#2563eb")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5,5")
        .attr("opacity", 0.7)
        .style("pointer-events", "none") // Don't interfere with sector clicks
    }
  })
}
