"""
InSight3D - Automated Inspection Report Generator
AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Generates comprehensive Quality Inspection Reports in:
1. CSV Format
2. PDF Format (with headers, metadata, metrics table, and defect breakdown)
"""

import os
import datetime
from typing import Dict, Any, Optional
import pandas as pd


def generate_csv_report(
    analysis_result: Dict[str, Any],
    component_id: str = "COMP-3D-001",
    output_csv_path: str = "results/reports/inspection_report.csv"
) -> str:
    """
    Export defect inspection summary and defect inventory to a CSV report.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Summary row data
    summary_data = {
        "Component ID": [component_id],
        "Inspection Timestamp": [timestamp],
        "Defect Count": [analysis_result.get("defect_count", 0)],
        "Total Defect Area (px)": [analysis_result.get("total_defect_area", 0.0)],
        "Largest Defect Area (px)": [analysis_result.get("largest_defect", 0.0)],
        "Defect Percentage (%)": [analysis_result.get("defect_percentage", 0.0)],
        "Quality Status": [analysis_result.get("status", "UNKNOWN")]
    }

    df_summary = pd.DataFrame(summary_data)

    # Append to existing CSV or write new file
    if os.path.exists(output_csv_path):
        df_existing = pd.read_csv(output_csv_path)
        df_combined = pd.concat([df_existing, df_summary], ignore_index=True)
        df_combined.to_csv(output_csv_path, index=False)
    else:
        df_summary.to_csv(output_csv_path, index=False)

    return output_csv_path


def generate_pdf_report(
    analysis_result: Dict[str, Any],
    component_id: str = "COMP-3D-001",
    output_pdf_path: str = "results/reports/inspection_report.pdf",
    image_paths: Optional[Dict[str, str]] = None
) -> str:
    """
    Generate professional PDF Quality Inspection Report using ReportLab.
    
    Args:
        analysis_result (Dict[str, Any]): Dictionary output from analyze_defects().
        component_id (str): Unique identifier for the inspected component.
        output_pdf_path (str): Target PDF file path.
        image_paths (Dict[str, str], optional): Dict containing paths to original/overlay images.
        
    Returns:
        str: Path to generated PDF file.
    """
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=letter,
            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
        )
        styles = getSampleStyleSheet()
        elements = []

        # Title Header
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#1E293B'),
            spaceAfter=6
        )
        subtitle_style = ParagraphStyle(
            'DocSubTitle',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor('#64748B'),
            spaceAfter=15
        )

        elements.append(Paragraph("InSight3D Quality Inspection Report", title_style))
        elements.append(Paragraph("AI-Based Internal Defect Segmentation in 3D-Printed Components", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0EA5E9'), spaceAfter=15))

        # Metadata Section
        status = analysis_result.get("status", "PASS")
        status_color = "#22C55E" if status == "PASS" else ("#F59E0B" if status == "WARNING" else "#EF4444")

        meta_table_data = [
            [Paragraph("<b>Component ID:</b>", styles['Normal']), Paragraph(component_id, styles['Normal']),
             Paragraph("<b>Inspection Date:</b>", styles['Normal']), Paragraph(timestamp, styles['Normal'])],
            [Paragraph("<b>Model Architecture:</b>", styles['Normal']), Paragraph("U-Net (Keras)", styles['Normal']),
             Paragraph("<b>Quality Status:</b>", styles['Normal']), Paragraph(f"<font color='{status_color}'><b>{status}</b></font>", styles['Normal'])]
        ]

        meta_table = Table(meta_table_data, colWidths=[110, 150, 110, 150])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0'))
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 15))

        # Metrics Summary Section
        elements.append(Paragraph("<b>Defect Analysis Summary Metrics</b>", styles['Heading2']))
        elements.append(Spacer(1, 8))

        metrics_table_data = [
            ["Metric Parameter", "Measured Value", "Unit / Status"],
            ["Total Defect Count", str(analysis_result.get("defect_count", 0)), "Defect regions"],
            ["Total Defect Area", f"{analysis_result.get('total_defect_area', 0.0):.2f}", "Pixels²"],
            ["Largest Defect Region", f"{analysis_result.get('largest_defect', 0.0):.2f}", "Pixels²"],
            ["Defect Percentage", f"{analysis_result.get('defect_percentage', 0.0):.2f}%", "Area %"],
            ["Final Assessment", status, f"Threshold: <1% Pass, 1-3% Warn"]
        ]

        metrics_table = Table(metrics_table_data, colWidths=[200, 160, 160])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0EA5E9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F1F5F9')])
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 15))

        # Defect Inventory Table (If defects present)
        defects = analysis_result.get("defects", [])
        if defects:
            elements.append(Paragraph("<b>Detected Defect Inventory Breakdown</b>", styles['Heading3']))
            elements.append(Spacer(1, 6))
            
            defect_rows = [["Defect #", "Area (px²)", "Centroid (X, Y)", "Bounding Box [X, Y, W, H]"]]
            for d in defects[:10]:  # Limit top 10 for table formatting
                d_id = str(d.get("id"))
                d_area = f"{d.get('area'):.1f}"
                d_cent = f"({d.get('centroid')[0]}, {d.get('centroid')[1]})"
                d_bbox = str(d.get('bbox'))
                defect_rows.append([d_id, d_area, d_cent, d_bbox])

            defect_table = Table(defect_rows, colWidths=[70, 120, 150, 180])
            defect_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#475569')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('PADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0'))
            ]))
            elements.append(defect_table)

        doc.build(elements)
        print(f"[SUCCESS] Generated PDF Inspection Report at: {output_pdf_path}")
        return output_pdf_path

    except ImportError:
        print("[WARNING] 'reportlab' package not installed. Generating simplified text PDF fallback...")
        with open(output_pdf_path.replace(".pdf", ".txt"), "w") as f:
            f.write(f"InSight3D Inspection Report\nComponent ID: {component_id}\nTimestamp: {timestamp}\n")
            f.write(f"Defect Count: {analysis_result.get('defect_count')}\nDefect %: {analysis_result.get('defect_percentage')}%\nStatus: {analysis_result.get('status')}\n")
        return output_pdf_path


def generate_inspection_report(
    analysis_result: Dict[str, Any],
    component_id: str = "COMP-3D-001",
    reports_dir: str = "results/reports"
) -> Dict[str, str]:
    """
    Generate both CSV and PDF inspection reports.
    """
    csv_path = os.path.join(reports_dir, "inspection_report.csv")
    pdf_path = os.path.join(reports_dir, "inspection_report.pdf")

    saved_csv = generate_csv_report(analysis_result, component_id=component_id, output_csv_path=csv_path)
    saved_pdf = generate_pdf_report(analysis_result, component_id=component_id, output_pdf_path=pdf_path)

    return {
        "csv_report": saved_csv,
        "pdf_report": saved_pdf
    }


if __name__ == "__main__":
    print("InSight3D Report Generator Module Test")
    sample_analysis = {
        "defect_count": 3,
        "total_defect_area": 450.5,
        "largest_defect": 231.0,
        "defect_percentage": 2.8,
        "status": "WARNING",
        "defects": [
            {"id": 1, "area": 231.0, "centroid": [128, 120], "bbox": [100, 95, 30, 25]},
            {"id": 2, "area": 120.0, "centroid": [60, 60], "bbox": [50, 50, 20, 20]},
            {"id": 3, "area": 99.5, "centroid": [200, 200], "bbox": [190, 190, 15, 15]}
        ]
    }

    report_paths = generate_inspection_report(sample_analysis, component_id="COMP-TEST-99")
    print(f"Generated Reports: {report_paths}")
