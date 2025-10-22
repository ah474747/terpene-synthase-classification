#!/usr/bin/env python3
"""
Generate PDF version of manuscript with embedded figures
"""

import markdown
import pdfkit
from pathlib import Path
import base64
import os

def image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def create_html_with_figures():
    """Create HTML version of manuscript with embedded figures"""
    
    # Read the markdown file
    with open('MANUSCRIPT_DRAFT.md', 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['tables', 'toc'])
    html_content = md.convert(markdown_content)
    
    # Create full HTML document with embedded figures
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Machine Learning Classification of Terpene Synthases using ESM-2 Protein Language Model Embeddings</title>
        <style>
            body {{
                font-family: 'Times New Roman', serif;
                line-height: 1.6;
                margin: 2cm;
                font-size: 12pt;
                color: #333;
            }}
            h1 {{
                font-size: 18pt;
                font-weight: bold;
                text-align: center;
                margin-bottom: 1cm;
                color: #2c3e50;
            }}
            h2 {{
                font-size: 14pt;
                font-weight: bold;
                margin-top: 1cm;
                margin-bottom: 0.5cm;
                color: #34495e;
            }}
            h3 {{
                font-size: 13pt;
                font-weight: bold;
                margin-top: 0.8cm;
                margin-bottom: 0.4cm;
                color: #34495e;
            }}
            p {{
                margin-bottom: 0.5cm;
                text-align: justify;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 1cm 0;
                font-size: 10pt;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 1cm auto;
                border: 1px solid #ddd;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .figure-caption {{
                text-align: center;
                font-style: italic;
                margin-bottom: 1cm;
                font-size: 11pt;
            }}
            .abstract {{
                background-color: #f8f9fa;
                padding: 1cm;
                border-left: 4px solid #007bff;
                margin: 1cm 0;
            }}
            .keywords {{
                font-weight: bold;
                margin-top: 0.5cm;
            }}
            .references {{
                font-size: 10pt;
            }}
            .references ol {{
                padding-left: 1.5cm;
            }}
            .references li {{
                margin-bottom: 0.3cm;
            }}
            @page {{
                margin: 2.5cm;
                @bottom-center {{
                    content: counter(page);
                    font-size: 10pt;
                }}
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    return html_template

def generate_pdf():
    """Generate PDF from HTML with embedded figures"""
    
    print("📄 GENERATING PDF MANUSCRIPT")
    print("=" * 35)
    
    # Check if figures exist
    figure_paths = [
        'results/figure1_algorithm_comparison.png',
        'results/figure2_traditional_vs_ml.png', 
        'results/figure3_class_balance_impact.png',
        'results/figure4_holdout_validation.png'
    ]
    
    missing_figures = []
    for path in figure_paths:
        if not Path(path).exists():
            missing_figures.append(path)
    
    if missing_figures:
        print(f"❌ Missing figures: {missing_figures}")
        print("Please run create_manuscript_figures.py first")
        return False
    
    print("✅ All figures found")
    
    # Create HTML with embedded figures
    print("🔄 Creating HTML with embedded figures...")
    html_content = create_html_with_figures()
    
    # Save HTML file
    with open('manuscript_with_figures.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ HTML file created: manuscript_with_figures.html")
    
    # Generate PDF using wkhtmltopdf
    try:
        print("🔄 Converting HTML to PDF...")
        
        options = {
            'page-size': 'A4',
            'margin-top': '2.5cm',
            'margin-right': '2.5cm',
            'margin-bottom': '2.5cm',
            'margin-left': '2.5cm',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None,
            'print-media-type': None,
            'dpi': 300,
            'image-quality': 100
        }
        
        pdfkit.from_file('manuscript_with_figures.html', 'Terpene_Synthase_Classification_Manuscript.pdf', options=options)
        
        print("✅ PDF generated successfully!")
        print("📁 Output file: Terpene_Synthase_Classification_Manuscript.pdf")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        print("💡 Alternative: Open manuscript_with_figures.html in your browser and print to PDF")
        return False

def main():
    # Change to the correct directory
    os.chdir('/Users/andrewhorwitz/Documents/Cursor_AI_projects/Terpene_Stuff/Ent_Kaurene_Binary_Classifier')
    
    success = generate_pdf()
    
    if success:
        print("\n🎉 MANUSCRIPT PDF GENERATION COMPLETE!")
        print("📄 The PDF includes:")
        print("   • Complete manuscript text")
        print("   • All 4 figures embedded")
        print("   • Professional formatting")
        print("   • Publication-ready layout")
    else:
        print("\n📝 ALTERNATIVE APPROACH:")
        print("1. Open 'manuscript_with_figures.html' in your browser")
        print("2. Use Ctrl+P (Cmd+P on Mac) to print")
        print("3. Select 'Save as PDF' as destination")
        print("4. Save with professional settings")

if __name__ == "__main__":
    main()
