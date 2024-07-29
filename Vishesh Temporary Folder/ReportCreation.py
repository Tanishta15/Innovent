import pandas as pd
from datetime import datetime, timedelta
from openai import OpenAI
from fpdf import FPDF

# Load and filter CSV data
data = pd.read_csv('/Users/visheshgoyal/Innovent/Vishesh Temporary Folder/results.csv')
data['Date'] = pd.to_datetime(data['Date'])
end_date = datetime.strptime('14-08-2023', '%d-%m-%Y')
start_date = end_date - timedelta(days=1)
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] < end_date)]
Product_ID = filtered_data['Product_ID'].unique()

part_to_map_image = {
    'A1': '/Users/visheshgoyal/Innovent/Vishesh Temporary Folder/MapA1.png',
    'B1': '/Users/visheshgoyal/Innovent/Vishesh Temporary Folder/MapB1.png',
    'C1': '/Users/visheshgoyal/Innovent/Vishesh Temporary Folder/MapC1.png'
}

part_to_direction = {
    'A1': '/Users/visheshgoyal/Innovent/Vishesh Temporary Folder/directionsA1.txt',
    'B1': '/Users/visheshgoyal/Innovent/Vishesh Temporary Folder/directionsB1.txt',
    'C1': '/Users/visheshgoyal/Innovent/Vishesh Temporary Folder/directionsC1.txt'
}

# Generate summary using OpenAI
def generate_report_summary(filtered_data):
    OpenAI.api_key = 'sk-None-DiN49NbLRnkIQwVgTdyBT3BlbkFJPuIv6N7cbCTw1RpyDwGs'
    response = OpenAI.Completion.create(
        engine="text-davinci-003",
        prompt=f"Generate a summary report from the following inventory data:\n{filtered_data}",
        max_tokens=500
    )
    return response.choices[0].text.strip()

summary = generate_report_summary(filtered_data)

def read_directions_file(file_path):
    with open(file_path, 'r') as file:
        directions = file.read()
    return directions

def create_pdf_report(summary, part_to_map_image, relevant_parts, part_to_direction):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", size=18)
    pdf.cell(200, 10, txt="Inventory Management Report", ln=True, align='C')

    # Summary
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=summary)

    # Add Relevant Map Images and Instructions
    for part in relevant_parts:
        if part in part_to_map_image:
            map_image_path = part_to_map_image[part]
            pdf.add_page()
            pdf.image(map_image_path, x=10, y=20, w=180)
        
        if part in part_to_direction:
            instruction_text_path = part_to_direction[part]
            instructions = part_to_direction(instruction_text_path)
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=instructions)


    pdf.output("inventory_management_report.pdf")

summary = generate_report_summary(filtered_data)
create_pdf_report(summary, part_to_map_image, Product_ID, part_to_direction)