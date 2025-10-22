#!/usr/bin/env python3
"""
PDF generation without duplicate figure legends
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
    """Create PDF manuscript without duplicate figure legends"""
    
    print("📄 GENERATING PDF WITHOUT DUPLICATE LEGENDS")
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
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        # Skip figure descriptions in markdown (we'll add our own)
        if line.startswith('**Figure ') and '.**' in line:
            print(f"   ⏭️  Skipping markdown figure description: {line[:50]}...")
            i += 1
            continue
        
        # Handle markdown image references
        if line.startswith('!['):
            # Extract figure number and path
            match = re.match(r'!\[Figure (\d+)\]\((.*?)\)', line)
            if match:
                fig_num = match.group(1)
                fig_path = match.group(2)
                
                # Add the actual image
                if Path(fig_path).exists():
                    img = Image(fig_path, width=6*inch, height=4*inch)
                    story.append(Spacer(1, 12))
                    story.append(img)
                    story.append(Spacer(1, 6))
                    
                    # Add appropriate caption based on figure number
                    if fig_num == '1':
                        caption = "<b>Figure 1. Machine Learning Algorithm Performance Comparison.</b> F1-scores across seven machine learning algorithms for three target terpene products. Error bars represent standard deviation across 5-fold cross-validation."
                    elif fig_num == '2':
                        caption = "<b>Figure 2. Traditional Methods vs ESM-2 + ML Performance.</b> Comparative performance for germacrene classification. ESM-2 + SVM-RBF achieves 32% improvement over sequence similarity and 70% improvement over amino acid composition approaches."
                    elif fig_num == '3':
                        caption = "<b>Figure 3. Class Balance Impact on Performance.</b> (A) Scatter plot showing the relationship between class balance and best F1-score performance. Germacrene (7.4% class balance) and pinene (6.5%) achieve superior performance compared to myrcene (4.2%). (B) Pie chart showing dataset composition with 1,262 total sequences distributed across target products and other terpene synthases."
                    elif fig_num == '4':
                        caption = "<b>Figure 4. Hold-out Validation Results.</b> Comprehensive evaluation metrics for the XGBoost model on the independent 20% hold-out test set for germacrene classification."
                    
                    story.append(Paragraph(clean_text(caption), body_style))
                    story.append(Spacer(1, 12))
                    print(f"   ✅ Added Figure {fig_num} with caption")
            
            i += 1
            continue
        
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
            story.append(Paragraph(clean_text(line[4:]), heading2_style))
            story.append(Spacer(1, 4))
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
        print("\n🎉 PDF GENERATION COMPLETE - NO DUPLICATE LEGENDS!")
        print("📄 The PDF includes:")
        print("   • Complete manuscript text with all reviewer feedback")
        print("   • Properly formatted tables with styling")
        print("   • Figures with single captions (no duplicates)")
        print("   • Professional formatting")
        print("   • Publication-ready layout")
    else:
        print("\n❌ PDF generation failed!")
        print("📋 Check the error messages above")

if __name__ == "__main__":
    main()
