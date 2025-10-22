#!/usr/bin/env python3
"""
Final PDF generation with proper table formatting and figure insertion
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.colors import black, blue, darkblue, grey, HexColor
from pathlib import Path
import re

def clean_text(text):
    """Clean markdown text for reportlab"""
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)      # Italic
    text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)  # Code
    return text

def parse_table(lines, start_idx):
    """Parse markdown table into reportlab table"""
    table_lines = []
    idx = start_idx
    
    # Collect all table lines
    while idx < len(lines) and lines[idx].strip().startswith('|'):
        table_lines.append(lines[idx].strip())
        idx += 1
    
    if len(table_lines) < 2:
        return None, idx
    
    # Parse table data
    table_data = []
    for i, line in enumerate(table_lines):
        # Skip separator line (with dashes)
        if '---' in line:
            continue
        
        # Split by | and clean
        cells = [cell.strip() for cell in line.split('|')]
        # Remove empty first and last cells (from leading/trailing |)
        cells = [cell for cell in cells if cell]
        
        if cells:
            table_data.append(cells)
    
    return table_data, idx

def create_manuscript_pdf():
    """Create PDF manuscript with proper tables and figure insertion"""
    
    print("📄 GENERATING FINAL PDF MANUSCRIPT")
    print("=" * 50)
    
    # Read the markdown file
    markdown_path = Path("MANUSCRIPT_DRAFT.md")
    if not markdown_path.exists():
        print("❌ MANUSCRIPT_DRAFT.md not found")
        return False
    
    print(f"📖 Reading markdown file: {markdown_path}")
    with open(markdown_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Check if figures exist
    figure_paths = {
        'Figure 1': 'results/figure1_algorithm_comparison.png',
        'Figure 2': 'results/figure2_traditional_vs_ml.png', 
        'Figure 3': 'results/figure3_class_balance_impact.png',
        'Figure 4': 'results/figure4_holdout_validation.png'
    }
    
    for name, path in figure_paths.items():
        if not Path(path).exists():
            print(f"❌ Missing: {name} at {path}")
            return False
    
    print("✅ All figures found")
    
    # Create PDF document
    doc = SimpleDocTemplate("Terpene_Synthase_Classification_Manuscript.pdf", 
                          pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=16,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=darkblue
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=12,
        textColor=darkblue
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=8,
        textColor=blue
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    # Parse markdown content
    print("🔄 Parsing markdown content...")
    lines = markdown_content.split('\n')
    story = []
    
    # Track which sections we're in for figure insertion
    in_results = False
    figure1_inserted = False
    figure2_inserted = False
    figure3_inserted = False
    figure4_inserted = False
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        # Track sections
        if line == '## Results':
            in_results = True
        
        # Handle headers
        if line.startswith('# '):
            story.append(Paragraph(clean_text(line[2:]), title_style))
            story.append(Spacer(1, 20))
            i += 1
            continue
            
        elif line.startswith('## '):
            story.append(Paragraph(clean_text(line[3:]), heading1_style))
            story.append(Spacer(1, 6))
            i += 1
            continue
            
        elif line.startswith('### '):
            heading_text = line[4:]
            story.append(Paragraph(clean_text(heading_text), heading2_style))
            story.append(Spacer(1, 4))
            
            # Insert Figure 1 after "Machine Learning Benchmark Results"
            if 'Machine Learning Benchmark Results' in heading_text and not figure1_inserted:
                # Continue reading until we find the figure placeholder or table
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith('**Table 1'):
                        # Found table 1, process paragraphs before it, then insert figure
                        break
                    elif next_line.startswith('##') or next_line.startswith('###'):
                        break
                    elif next_line and not next_line.startswith('|'):
                        story.append(Paragraph(clean_text(next_line), body_style))
                        story.append(Spacer(1, 3))
                    i += 1
                
                # Insert Figure 1
                story.append(Spacer(1, 12))
                img1 = Image(figure_paths['Figure 1'], width=6*inch, height=4*inch)
                story.append(img1)
                story.append(Spacer(1, 6))
                story.append(Paragraph("<b>Figure 1. Machine Learning Algorithm Performance Comparison.</b> F1-scores across seven machine learning algorithms for three target terpene products. Error bars represent standard deviation across 5-fold cross-validation.", body_style))
                story.append(Spacer(1, 12))
                figure1_inserted = True
                continue
            
            # Insert Figure 2 after "Traditional Methods Comparison"
            elif 'Traditional Methods Comparison' in heading_text and in_results and not figure2_inserted:
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith('**Table 2') or next_line.startswith('###'):
                        break
                    elif next_line and not next_line.startswith('|'):
                        story.append(Paragraph(clean_text(next_line), body_style))
                        story.append(Spacer(1, 3))
                    i += 1
                
                # Insert Figure 2
                story.append(Spacer(1, 12))
                img2 = Image(figure_paths['Figure 2'], width=6*inch, height=4*inch)
                story.append(img2)
                story.append(Spacer(1, 6))
                story.append(Paragraph("<b>Figure 2. Traditional Methods vs ESM-2 + ML Performance.</b> Comparative performance for germacrene classification. ESM-2 + SVM-RBF achieves 32% improvement over sequence similarity and 70% improvement over amino acid composition approaches.", body_style))
                story.append(Spacer(1, 12))
                figure2_inserted = True
                continue
            
            # Insert Figure 3 after appropriate section
            elif 'Class Balance' in heading_text and not figure3_inserted:
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith('###') or next_line.startswith('##'):
                        break
                    elif next_line and not next_line.startswith('|'):
                        story.append(Paragraph(clean_text(next_line), body_style))
                        story.append(Spacer(1, 3))
                    i += 1
                
                # Insert Figure 3
                story.append(Spacer(1, 12))
                img3 = Image(figure_paths['Figure 3'], width=6*inch, height=4*inch)
                story.append(img3)
                story.append(Spacer(1, 6))
                story.append(Paragraph("<b>Figure 3. Class Balance Impact on Performance.</b> Relationship between class balance and F1-score performance across three terpene products.", body_style))
                story.append(Spacer(1, 12))
                figure3_inserted = True
                continue
            
            # Insert Figure 4 after "Hold-out Validation"
            elif 'Hold-out Validation' in heading_text and not figure4_inserted:
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith('###') or next_line.startswith('##'):
                        break
                    elif next_line and not next_line.startswith('|'):
                        story.append(Paragraph(clean_text(next_line), body_style))
                        story.append(Spacer(1, 3))
                    i += 1
                
                # Insert Figure 4
                story.append(Spacer(1, 12))
                img4 = Image(figure_paths['Figure 4'], width=6*inch, height=4*inch)
                story.append(img4)
                story.append(Spacer(1, 6))
                story.append(Paragraph("<b>Figure 4. Hold-out Validation Results.</b> Comprehensive evaluation metrics for the XGBoost model on the independent 20% hold-out test set for germacrene classification.", body_style))
                story.append(Spacer(1, 12))
                figure4_inserted = True
                continue
            
            i += 1
            continue
        
        # Handle tables
        elif line.startswith('|'):
            table_data, new_i = parse_table(lines, i)
            if table_data:
                # Create table with styling
                t = Table(table_data, repeatRows=1)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4472C4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#E7E6E6')]),
                ]))
                story.append(Spacer(1, 6))
                story.append(t)
                story.append(Spacer(1, 12))
                i = new_i
                continue
            
        # Skip figure placeholders (we insert them programmatically)
        elif '![Figure' in line or line.startswith('**Figure'):
            i += 1
            continue
        
        # Handle regular paragraphs
        else:
            story.append(Paragraph(clean_text(line), body_style))
            story.append(Spacer(1, 3))
        
        i += 1
    
    print(f"📊 Total story elements: {len(story)}")
    
    # Build PDF
    print("🔄 Building PDF document...")
    doc.build(story)
    
    print("✅ PDF generated successfully!")
    print(f"📁 Output file: Terpene_Synthase_Classification_Manuscript.pdf")
    
    return True

def main():
    """Main function"""
    success = create_manuscript_pdf()
    
    if success:
        print("\n🎉 FINAL PDF GENERATION COMPLETE!")
        print("📄 The PDF includes:")
        print("   • Complete manuscript text with all reviewer feedback")
        print("   • Properly formatted tables with styling")
        print("   • Figures inserted at appropriate locations in text")
        print("   • Professional formatting")
        print("   • Publication-ready layout")
    else:
        print("\n❌ PDF generation failed!")
        print("📋 Check the error messages above")

if __name__ == "__main__":
    main()
