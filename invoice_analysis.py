import os
import re
import pandas as pd
import PyPDF2
from pathlib import Path
from nltk.stem import PorterStemmer
import nltk
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab')

# Initialize stemmer
stemmer = PorterStemmer()

def categorize_task(description):
    """
    Categorize task based on description using stemming and keyword matching
    """
    if not description:
        return "Other"
    
    try:
        # Convert to lowercase and tokenize
        description_lower = description.lower()
        words = nltk.word_tokenize(description_lower)
        stemmed_words = [stemmer.stem(word) for word in words]
        stemmed_text = ' '.join(stemmed_words)
        
        # Task categories with their keywords (stemmed and original)
        task_categories = {
            'motion to compel': [
                'motion to compel', 'compel motion', 'compel', 'opposition to motion to compel'
            ],
            'subpoena': [
                'subpoena', 'subpeona', 'subpoenas', 'subpeonas'
            ],
            'discovery': [
                'discoveri', 'discover', 'product', 'interrogatori', 'product', 
                'product', 'admiss', 'product', 'product', 'rfp', 'rogs', 'srogs',
                'rfas', 'request for product', 'request for admiss', 'special interrogatori'
            ],
            'drafting motion': [
                'draft motion', 'draft brief', 'prepar motion', 'motion draft',
                'brief draft', 'motion prepar', 'draft oppos', 'oppos motion'
            ],
            'CMC statement': [
                'cmc statement', 'case manag confer statement', 'status statement',
                'joint status report', 'status confer statement', 'case manag conference'
            ],
            'CMC attendance': [
                'attend cmc', 'cmc attend', 'case manag confer', 'status confer',
                'particip cmc', 'appear cmc'
            ],
            'meet and confer': [
                'meet confer', 'meet and confer', 'confer with', 'discuss with',
                'meet with', 'M&C'
            ],
            'RFP': [
                'rfp', 'request for product', 'product request', 'product respons'
            ],
            'ROG': [
                'rog', 'interrogatori', 'special interrogatori', 'srog',
                'respons to interrogatori', 'object to interrogatori'
            ],
            'interrogatories': [
                'interrogatori', 'interrog', 'respons to interrogatori',
                'object to interrogatori', 'special interrogatori'
            ],
            'court attendance': [
                'court appear', 'attend hearing', 'attend confer', 'appear before',
                'court hearing', 'status hearing', 'motion hearing'
            ],
            'settlement': [
                'settlement', 'settle', 'settlement discuss', 'settlement confer',
                'settlement propos', 'settlement negoti', 'settlement agreement'
            ],
            'client communication': [
                'client', 'email client', 'call client', 'discuss with client',
                'updat client', 'client updat', 'client meet'
            ],
            'legal research': [
                'research', 'analysi', 'review law', 'legal research',
                'case law', 'statutor', 'regulator', 'legal analysi'
            ],
            'document review': [
                'review', 'draft review', 'document review', 'review document',
                'review draft', 'revis', 'edit'
            ]
        }
        
        # Check each category
        matched_categories = []
        
        for category, keywords in task_categories.items():
            for keyword in keywords:
                # Check both stemmed and original text
                if (keyword in stemmed_text or 
                    keyword in description_lower or
                    any(keyword in stemmed_word for stemmed_word in stemmed_words)):
                    matched_categories.append(category)
                    break  # Only need one match per category
        
        # Return the most specific category, or "Other" if no match
        if matched_categories:
            # Priority for more specific legal tasks
            priority_categories = ['drafting motion', 'court attendance', 'CMC attendance', 
                                  'settlement', 'discovery', 'RFP', 'ROG', 'interrogatories']
            
            for priority_cat in priority_categories:
                if priority_cat in matched_categories:
                    return priority_cat
            
            return matched_categories[0]  # Return first match if no priority category found
        
        return "Other"
    except Exception as e:
        print(f"Error in task categorization: {e}")
        return "Other"

def extract_invoice_data(pdf_path):
    """
    Extract invoice data from PDF file
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            print(f"Reading {pdf_path}")
            print(text)
        return extract_data_from_text(text, pdf_path.name)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None, None

def extract_data_from_text(text, filename):
    """
    Extract fee summary and fee detail data from text
    """
    # Extract header information
    invoice_date = extract_invoice_date(text)
    invoice_no = extract_invoice_no(text)
    matter_no = extract_matter_no(text)
    matter_description = extract_matter_description(text)
    
    # Extract fee summary data
    fee_summary_data = extract_fee_summary(text, filename, invoice_date, invoice_no, matter_no, matter_description)
    
    # Extract fee detail data
    fee_detail_data = extract_fee_detail(text, filename, matter_no)
    
    return fee_summary_data, fee_detail_data

def extract_invoice_date(text):
    """Extract invoice date from text"""
    date_patterns = [
        r'Invoice Date\s+(\d{1,2}-[A-Za-z]{3}-\d{4})',
        r'Invoice Date\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})',
        r'(\d{1,2}-[A-Za-z]{3}-\d{4})\s*Invoice Date',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return "Not Found"

def extract_invoice_no(text):
    """Extract invoice number from text"""
    patterns = [
        r'Invoice No\.\s*([A-Za-z0-9-]+)',
        r'Invoice No\s+([A-Za-z0-9-]+)',
        r'Invoice Number\s+([A-Za-z0-9-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return "Not Found"

def extract_matter_no(text):
    """Extract matter number from text"""
    patterns = [
        r'Our Matter No\.\s*([A-Za-z0-9.-]+)',
        r'Matter No\.\s*([A-Za-z0-9.-]+)',
        r'Our Matter Number\s+([A-Za-z0-9.-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return "Not Found"

def extract_matter_description(text):
    """Extract matter description from text"""
    try:
        # Look for patterns like "REPRESENT TCL AGAINST CADENCE DESIGN SYSTEM, INC. 23-00339"
        # This typically appears after the matter number and invoice number
        patterns = [
            r'Our Matter No\.\s*[A-Za-z0-9.-]+\s*Invoice No\.\s*[A-Za-z0-9-]+\s*(REPRESENT[^\n]+)',
            r'Matter No\.\s*[A-Za-z0-9.-]+\s*Invoice No\.\s*[A-Za-z0-9-]+\s*(REPRESENT[^\n]+)',
            r'REPRESENT[^\n]+(?:\n[^\n]+)*',  # Look for any line starting with REPRESENT
            r'For professional services in connection with[^\n]+',  # Alternative pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                description = match.group(0).strip()
                # Clean up the description - remove extra whitespace
                description = re.sub(r'\s+', ' ', description)
                return description
        
        # If not found with specific patterns, try to find any text between matter/invoice info and FEE SUMMARY
        general_pattern = r'Our Matter No\.\s*[A-Za-z0-9.-]+\s*Invoice No\.\s*[A-Za-z0-9-]+\s*(.*?)(?=FEE SUMMARY|$)'
        match = re.search(general_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            description = match.group(1).strip()
            # Remove any extra whitespace and limit length
            description = re.sub(r'\s+', ' ', description)
            if len(description) > 200:  # If too long, take first part
                description = description[:200] + "..."
            return description
        
        return "Not Found"
    except Exception as e:
        print(f"Error extracting matter description: {e}")
        return "Not Found"

def extract_fee_summary(text, filename, invoice_date, invoice_no, matter_no, matter_description):
    """
    Extract fee summary data from text
    """
    fee_summary_data = []
    
    # Find FEE SUMMARY section
    fee_summary_match = re.search(r'FEE SUMMARY(.*?)(?=FEE DETAIL|Total Hours|\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
    
    if fee_summary_match:
        fee_summary_text = fee_summary_match.group(1)
        
        # Pattern to match timekeeper lines: Name, Hours, Rate
        # Example: "Cloern, B. 10.30 1,150.00"
        #pattern = r'^([A-Za-z]+(?:,\s*[A-Za-z.]+)+)\s+(\d+\.\d+)\s+([0-9,]+\.\d{2})'
        pattern = r'^([A-Za-z]+(?:[-\' ][A-Za-z]+)*(?:,\s*[A-Za-z.]+)+)\s+(\d+\.\d+)\s+([0-9,]+\.\d{2})'
        
        lines = fee_summary_text.split('\n')
        for line in lines:
            line = line.strip()
            match = re.match(pattern, line)
            if match:
                lawyer_name = match.group(1).strip()
                hours = match.group(2)
                rate = match.group(3).replace(',', '')  # Remove commas from rate
                total = float(hours) * float(rate)  
                
                fee_summary_data.append({
                    'pdf_filename': filename,
                    'date_of_invoice': invoice_date,
                    'invoice_no': invoice_no,
                    'matter_no': matter_no,
                    'matter_description': matter_description,
                    'lawyer_name': lawyer_name,
                    'hours': float(hours),
                    'rate': float(rate),
                    'total': total
                })
    
    return fee_summary_data

def extract_fee_detail(text, filename, matter_no=0):
    """
    Extract fee detail data from text
    """
    fee_detail_data = []
    
    # Find FEE DETAIL section
    fee_detail_match = re.search(r'FEE DETAIL(.*?)(?=Total\s+\d+\.\d{2}|REPRESENT|\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
    
    if fee_detail_match:
        fee_detail_text = fee_detail_match.group(1)
        
        # Pattern to match fee detail lines: Date, Timekeeper, Hours, Description
        # Example: "11-Jul-24 Rorwin, J.K. 0.50 Review draft email..."
        #pattern = r'^(\d{1,2}-[A-Za-z]{3}-\d{2,4})\s+([A-Za-z]+(?:,\s*[A-Za-z.]+)+)\s+(\d+\.\d+)\s+(.*)'
        # fix hyphenated name
        pattern = r'^(\d{1,2}-[A-Za-z]{3}-\d{2,4})\s+([A-Za-z]+(?:[\s-]*[A-Za-z]*(?:,\s*[A-Za-z.]+)*))\s+(\d+\.\d+)\s+(.*)'
        
        lines = text.split('\n')
        current_entry = None
        
        for line in lines:
            line = line.strip()
            print(f"   {line}")
            
            # Check if this is a new entry
            match = re.match(pattern, line)
            if match:
                # Save previous entry if exists
                if current_entry:
                    # Categorize task before saving
                    current_entry['task_category'] = categorize_task(current_entry['description'])
                    fee_detail_data.append(current_entry)
                
                # Start new entry
                #matter_str = matter_no
                date_str = match.group(1)
                timekeeper = match.group(2).strip()
                hours = match.group(3)
                description = match.group(4).strip()

                # Clean up timekeeper name - remove extra spaces around hyphens
                timekeeper = re.sub(r'\s*-\s*', '-', timekeeper)
                
                # Convert date format from "dd-mmm-yy" to "dd-mm-yyyy"
                try:
                    # Parse the original date
                    date_obj = datetime.strptime(date_str, '%d-%b-%y')
                    # Format to dd-mm-yyyy
                    formatted_date = date_obj.strftime('%d-%m-%Y')
                except ValueError:
                    # If parsing fails, try with 4-digit year
                    try:
                        date_obj = datetime.strptime(date_str, '%d-%b-%Y')
                        formatted_date = date_obj.strftime('%d-%m-%Y')
                    except ValueError:
                        # If both fail, keep original date
                        formatted_date = date_str
                        print(f"Warning: Could not parse date: {date_str}")
                
                current_entry = {
                    'pdf_filename': filename,
                    #'matter_no': matter_no,
                    'date': date_str,
                    'date_': formatted_date,  # Use the converted date
                    'timekeeper': timekeeper,
                    'hours': float(hours),
                    'description': description
                }
            else:
                # This is a continuation line, append to current description
                if current_entry and line:
                    current_entry['description'] += ' ' + line.strip()
        
        # Don't forget the last entry
        if current_entry:
            # Categorize task before saving
            current_entry['task_category'] = categorize_task(current_entry['description'])
            fee_detail_data.append(current_entry)
    
    return fee_detail_data
'''
def extract_fee_detail(text, filename):
    """
    Extract fee detail data from text
    """
    fee_detail_data = []
    
    # Find FEE DETAIL section
    fee_detail_match = re.search(r'FEE DETAIL(.*?)(?=Total\s+\d+\.\d{2}|REPRESENT|\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
    
    if fee_detail_match:
        fee_detail_text = fee_detail_match.group(1)
        
        # Pattern to match fee detail lines: Date, Timekeeper, Hours, Description
        # Example: "11-Jul-24 Crowther, R.C. 0.50 Review draft email..."
        pattern = r'^(\d{1,2}-[A-Za-z]{3}-\d{2,4})\s+([A-Za-z]+(?:,\s*[A-Za-z.]+)+)\s+(\d+\.\d+)\s+(.*)'
        
        lines = fee_detail_text.split('\n')
        current_entry = None
        
        for line in lines:
            line = line.strip()
            
            # Check if this is a new entry
            match = re.match(pattern, line)
            if match:
                # Save previous entry if exists
                if current_entry:
                    # Categorize task before saving
                    current_entry['task_category'] = categorize_task(current_entry['description'])
                    fee_detail_data.append(current_entry)
                
                # Start new entry
                date = match.group(1)
                timekeeper = match.group(2).strip()
                hours = match.group(3)
                description = match.group(4).strip()
                
                current_entry = {
                    'pdf_filename': filename,
                    'date': date,
                    'timekeeper': timekeeper,
                    'hours': float(hours),
                    'description': description
                }
            else:
                # This is a continuation line, append to current description
                if current_entry and line:
                    current_entry['description'] += ' ' + line.strip()
        
        # Don't forget the last entry
        if current_entry:
            # Categorize task before saving
            current_entry['task_category'] = categorize_task(current_entry['description'])
            fee_detail_data.append(current_entry)
    
    return fee_detail_data
'''
def add_rate_to_fee_detail(fee_detail_df, fee_summary_df):
    """
    Add rate to Fee Detail by matching with Fee Summary data
    """
    if fee_detail_df.empty or fee_summary_df.empty:
        return fee_detail_df
    
    # Create a mapping from (pdf_filename, lawyer_name) to rate
    rate_mapping = {}
    for _, row in fee_summary_df.iterrows():
        key = (row['pdf_filename'], row['lawyer_name'])
        rate_mapping[key] = row['rate']
    
    # Add rate column to fee_detail_df
    fee_detail_df['rate'] = fee_detail_df.apply(
        lambda row: rate_mapping.get((row['pdf_filename'], row['timekeeper']), 0), 
        axis=1
    )
    
    # Calculate amount (rate × hours)
    fee_detail_df['amount'] = fee_detail_df['rate'] * fee_detail_df['hours']
    
    return fee_detail_df

def create_monthly_task_pivot(fee_detail_df):
    """
    Create a pivot table showing sum of hours by task category and month,
    with timekeepers as comma-separated list and rate×hours breakdown
    """
    if fee_detail_df.empty or 'task_category' not in fee_detail_df.columns:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original
    df = fee_detail_df.copy()
    
    # Convert date to datetime and extract year-month
    try:
        # Handle different date formats (e.g., "11-Jul-24", "11-Jul-2024")
        df['date_dt'] = pd.to_datetime(df['date'], format='%d-%b-%y', errors='coerce')
        # If the above fails, try with 4-digit year
        if df['date_dt'].isna().any():
            df['date_dt'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')
        
        # Extract year and month for grouping
        df['year_month'] = df['date_dt'].dt.to_period('M')
        df['month_name'] = df['date_dt'].dt.strftime('%Y-%m')
    except Exception as e:
        print(f"Error processing dates: {e}")
        return pd.DataFrame()
    
    # Group by task category and month, then aggregate
    pivot_data = []
    
    for (task_category, month), group in df.groupby(['task_category', 'year_month']):
        total_hours = group['hours'].sum()
        total_amount = group['amount'].sum()
        timekeepers = ', '.join(sorted(group['timekeeper'].unique()))
        month_str = str(month)
        
        # Create timekeeper breakdown with rate×hours
        timekeeper_breakdown = []
        for timekeeper in sorted(group['timekeeper'].unique()):
            tk_data = group[group['timekeeper'] == timekeeper]
            tk_hours = tk_data['hours'].sum()
            tk_amount = tk_data['amount'].sum()
            # Get the rate (should be consistent for same timekeeper in same period)
            tk_rate = tk_data['rate'].iloc[0] if not tk_data.empty else 0
            timekeeper_breakdown.append(f"{timekeeper}: {tk_hours:.2f}h × ${tk_rate:.2f} = ${tk_amount:.2f}")
        
        timekeeper_details = '; '.join(timekeeper_breakdown)
        
        pivot_data.append({
            'Task Category': task_category,
            'Month': month_str,
            'Total Hours': total_hours,
            'Total Amount': total_amount,
            'Timekeepers': timekeepers,
            'Timekeeper Breakdown': timekeeper_details
        })
    
    pivot_df = pd.DataFrame(pivot_data)
    
    # Sort by month and task category
    if not pivot_df.empty:
        pivot_df = pivot_df.sort_values(['Month', 'Task Category'])
    
    return pivot_df

def create_monthly_timekeeper_pivot(fee_detail_df, fee_summary_df):
    """
    Create a pivot table for each month and each matter number showing:
    - Total fees (hours × rate) and hours for each timekeeper
    - Timekeepers listed in columns
    - Summary columns for total hours and total amount
    """
    if fee_detail_df.empty or fee_summary_df.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original
    df = fee_detail_df.copy()
    
    # Add matter_no and matter_description to fee_detail_df by merging with fee_summary_df
    matter_mapping = fee_summary_df[['pdf_filename', 'matter_no', 'matter_description']].drop_duplicates()
    df = df.merge(matter_mapping, on='pdf_filename', how='left')
    
    # Convert date to datetime and extract year-month
    try:
        # Handle different date formats
        df['date_dt'] = pd.to_datetime(df['date'], format='%d-%b-%y', errors='coerce')
        if df['date_dt'].isna().any():
            df['date_dt'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')
        
        # Extract year and month for grouping
        df['year_month'] = df['date_dt'].dt.to_period('M')
        df['month_name'] = df['date_dt'].dt.strftime('%Y-%m')
    except Exception as e:
        print(f"Error processing dates in monthly timekeeper pivot: {e}")
        return pd.DataFrame()
    
    # Create pivot table for hours
    hours_pivot = df.pivot_table(
        index=['matter_no', 'month_name'],
        columns='timekeeper',
        values='hours',
        aggfunc='sum',
        fill_value=0
    )
    
    # Create pivot table for amounts
    amount_pivot = df.pivot_table(
        index=['matter_no', 'month_name'],
        columns='timekeeper',
        values='amount',
        aggfunc='sum',
        fill_value=0
    )
    
    # Combine both pivot tables
    combined_data = []
    
    for (matter_no, month_name), group in df.groupby(['month_name', 'matter_no']):
        # Get matter description (should be the same for all entries with same matter_no)
        matter_desc = group['matter_description'].iloc[0] if not group['matter_description'].isna().all() else "Not Found"
        
        # Calculate total hours and total amount for this matter and month
        total_hours_all = group['hours'].sum()
        total_amount_all = group['amount'].sum()
        
        row_data = {
            'Month': month_name,
            'Matter No': matter_no,
            'Matter Description': matter_desc,
            'Total Hours': total_hours_all,  # Sum of all hours
            'Total Amount': total_amount_all,  # Sum of all amount
        }
        
        # Add hours for each timekeeper
        for timekeeper in group['timekeeper'].unique():
            tk_data = group[group['timekeeper'] == timekeeper]
            total_hours = tk_data['hours'].sum()
            total_amount = tk_data['amount'].sum()
            
            row_data[f'{timekeeper} - Hours'] = total_hours
            row_data[f'{timekeeper} - Amount'] = total_amount
        
        combined_data.append(row_data)
    
    combined_df = pd.DataFrame(combined_data)
    
    # Fill NaN values with 0
    combined_df = combined_df.fillna(0)
    
    # Sort by matter no and month
    if not combined_df.empty:
        combined_df = combined_df.sort_values(['Matter No', 'Month'])
    
    # Reorder columns to put summary columns after description
    column_order = ['Month', 'Matter No', 'Matter Description', 'Total Hours', 'Total Amount']
    
    # Add timekeeper columns (both hours and amount) after the summary columns
    timekeeper_columns = [col for col in combined_df.columns if col not in column_order]
    
    # Separate hours and amount columns for better ordering
    hours_columns = [col for col in timekeeper_columns if ' - Hours' in col]
    amount_columns = [col for col in timekeeper_columns if ' - Amount' in col]
    
    # Sort timekeeper columns alphabetically
    hours_columns.sort()
    amount_columns.sort()
    
    # Create final column order: fixed columns, then hours columns, then amount columns
    final_column_order = column_order + hours_columns + amount_columns
    
    # Reorder the DataFrame columns
    combined_df = combined_df[final_column_order]
    
    return combined_df
'''
def create_monthly_timekeeper_pivot(fee_detail_df, fee_summary_df):
    """
    Create a pivot table for each month and each matter number showing:
    - Total fees (hours × rate) and hours for each timekeeper
    - Timekeepers listed in columns
    """
    if fee_detail_df.empty or fee_summary_df.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original
    df = fee_detail_df.copy()
    
    # Add matter_no and matter_description to fee_detail_df by merging with fee_summary_df
    matter_mapping = fee_summary_df[['pdf_filename', 'matter_no', 'matter_description']].drop_duplicates()
    df = df.merge(matter_mapping, on='pdf_filename', how='left')
    
    # Convert date to datetime and extract year-month
    try:
        # Handle different date formats
        df['date_dt'] = pd.to_datetime(df['date'], format='%d-%b-%y', errors='coerce')
        if df['date_dt'].isna().any():
            df['date_dt'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')
        
        # Extract year and month for grouping
        df['year_month'] = df['date_dt'].dt.to_period('M')
        df['month_name'] = df['date_dt'].dt.strftime('%Y-%m')
    except Exception as e:
        print(f"Error processing dates in monthly timekeeper pivot: {e}")
        return pd.DataFrame()
    
    # Create pivot table for hours
    hours_pivot = df.pivot_table(
        index=['matter_no', 'month_name'],
        columns='timekeeper',
        values='hours',
        aggfunc='sum',
        fill_value=0
    )
    
    # Create pivot table for amounts
    amount_pivot = df.pivot_table(
        index=['matter_no', 'month_name'],
        columns='timekeeper',
        values='amount',
        aggfunc='sum',
        fill_value=0
    )
    
    # Combine both pivot tables
    combined_data = []
    
    for (matter_no, month_name), group in df.groupby(['month_name', 'matter_no']):
        # Get matter description (should be the same for all entries with same matter_no)
        matter_desc = group['matter_description'].iloc[0] if not group['matter_description'].isna().all() else "Not Found"
        
        row_data = {
            'Month': month_name,
            'Matter No': matter_no,
            'Matter Description': matter_desc,
        }
        
        # Add hours for each timekeeper
        for timekeeper in group['timekeeper'].unique():
            tk_data = group[group['timekeeper'] == timekeeper]
            total_hours = tk_data['hours'].sum()
            total_amount = tk_data['amount'].sum()
            
            row_data[f'{timekeeper} - Hours'] = total_hours
            row_data[f'{timekeeper} - Amount'] = total_amount
        
        combined_data.append(row_data)
    
    combined_df = pd.DataFrame(combined_data)
    
    # Fill NaN values with 0
    combined_df = combined_df.fillna(0)
    
    # Sort by matter no and month
    if not combined_df.empty:
        combined_df = combined_df.sort_values(['Matter No', 'Month'])
    
    return combined_df
'''
def create_monthly_summary(fee_detail_df):
    """
    Create a summary per month with total hours and total amount across all matters and timekeepers
    """
    if fee_detail_df.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original
    df = fee_detail_df.copy()
    
    # Convert date to datetime and extract year-month
    try:
        # Handle different date formats
        df['date_dt'] = pd.to_datetime(df['date'], format='%d-%b-%y', errors='coerce')
        if df['date_dt'].isna().any():
            df['date_dt'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')
        
        # Extract year and month for grouping
        df['year_month'] = df['date_dt'].dt.to_period('M')
        df['month_name'] = df['date_dt'].dt.strftime('%Y-%m')
        df['month_year'] = df['date_dt'].dt.strftime('%B %Y')  # e.g., "August 2024"
    except Exception as e:
        print(f"Error processing dates in monthly summary: {e}")
        return pd.DataFrame()
    
    # Group by month and calculate totals
    monthly_data = []
    
    for month_name, group in df.groupby('month_name'):
        month_year = group['month_year'].iloc[0] if not group['month_year'].isna().all() else month_name
        
        total_hours = group['hours'].sum()
        total_amount = group['amount'].sum()
        
        # Count unique matters and timekeepers
        unique_matters = group['matter_no'].nunique() if 'matter_no' in group.columns else 0
        unique_timekeepers = group['timekeeper'].nunique()
        
        monthly_data.append({
            'Month': month_name,
            'Month Year': month_year,
            'Total Hours': total_hours,
            'Total Amount': total_amount,
            'Number of Matters': unique_matters,
            'Number of Timekeepers': unique_timekeepers
        })
    
    monthly_df = pd.DataFrame(monthly_data)
    
    # Sort by month
    if not monthly_df.empty:
        monthly_df = monthly_df.sort_values('Month')
    
    # Calculate cumulative totals
    if not monthly_df.empty:
        monthly_df['Cumulative Hours'] = monthly_df['Total Hours'].cumsum()
        monthly_df['Cumulative Amount'] = monthly_df['Total Amount'].cumsum()
    
    return monthly_df


def process_pdf_directory(directory_path):
    """
    Process all PDF files in the directory and return dataframes
    """
    all_fee_summary = []
    all_fee_detail = []
    
    pdf_files = list(Path(directory_path).glob('*.pdf'))
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        fee_summary, fee_detail = extract_invoice_data(pdf_file)
        
        if fee_summary:
            all_fee_summary.extend(fee_summary)
        
        if fee_detail:
            all_fee_detail.extend(fee_detail)
    
    # Create dataframes
    fee_summary_df = pd.DataFrame(all_fee_summary) if all_fee_summary else pd.DataFrame()
    fee_detail_df = pd.DataFrame(all_fee_detail) if all_fee_detail else pd.DataFrame()
    
    # Add rate and amount to fee detail
    if not fee_detail_df.empty and not fee_summary_df.empty:
        fee_detail_df = add_rate_to_fee_detail(fee_detail_df, fee_summary_df)
    
    return fee_summary_df, fee_detail_df

def save_to_excel(fee_summary_df, fee_detail_df, output_path):
    """
    Save data to Excel file with multiple sheets
    """
    # Check if we have any data to save
    if fee_summary_df.empty and fee_detail_df.empty:
        print("No data to save. Creating empty Excel file with message.")
        # Create a minimal DataFrame with a message
        message_df = pd.DataFrame({'Message': ['No invoice data was extracted from the PDF files.']})
        message_df.to_excel(output_path, sheet_name='No Data', index=False)
        return
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Save Fee Summary sheet
        if not fee_summary_df.empty:
            fee_summary_df.to_excel(writer, sheet_name='Fee Summary', index=False)
            print(f"Saved {len(fee_summary_df)} fee summary entries")
        else:
            # Create empty sheet with message
            empty_summary_df = pd.DataFrame({'Message': ['No fee summary data found.']})
            empty_summary_df.to_excel(writer, sheet_name='Fee Summary', index=False)
        
        # Save Fee Detail sheet
        if not fee_detail_df.empty:
            fee_detail_df.to_excel(writer, sheet_name='Fee Detail', index=False)
            print(f"Saved {len(fee_detail_df)} fee detail entries")
        else:
            # Create empty sheet with message
            empty_detail_df = pd.DataFrame({'Message': ['No fee detail data found.']})
            empty_detail_df.to_excel(writer, sheet_name='Fee Detail', index=False)
        
        # Create and save Task by Month pivot table
        if not fee_detail_df.empty and 'task_category' in fee_detail_df.columns:
            pivot_df = create_monthly_task_pivot(fee_detail_df)
            if not pivot_df.empty:
                pivot_df.to_excel(writer, sheet_name='Task by Month', index=False)
                print(f"Saved Task by Month pivot table with {len(pivot_df)} entries")
            else:
                empty_pivot_df = pd.DataFrame({'Message': ['Could not create pivot table from fee detail data.']})
                empty_pivot_df.to_excel(writer, sheet_name='Task by Month', index=False)
        else:
            empty_pivot_df = pd.DataFrame({'Message': ['No fee detail data available for pivot table.']})
            empty_pivot_df.to_excel(writer, sheet_name='Task by Month', index=False)
        
        # Create and save Monthly Summary sheet
        if not fee_detail_df.empty:
            monthly_summary_df = create_monthly_summary(fee_detail_df)
            if not monthly_summary_df.empty:
                monthly_summary_df.to_excel(writer, sheet_name='Monthly Summary', index=False)
                print(f"Saved monthly summary with {len(monthly_summary_df)} entries")
            else:
                empty_monthly_df = pd.DataFrame({'Message': ['Could not create monthly summary.']})
                empty_monthly_df.to_excel(writer, sheet_name='Monthly Summary', index=False) 

        # Create and save Monthly Timekeeper pivot table
        if not fee_detail_df.empty and not fee_summary_df.empty:
            timekeeper_pivot_df = create_monthly_timekeeper_pivot(fee_detail_df, fee_summary_df)
            if not timekeeper_pivot_df.empty:
                timekeeper_pivot_df.to_excel(writer, sheet_name='Monthly Timekeeper', index=False)
                print(f"Saved Monthly Timekeeper pivot table with {len(timekeeper_pivot_df)} entries")
            else:
                empty_timekeeper_pivot = pd.DataFrame({'Message': ['Could not create monthly timekeeper pivot table.']})
                empty_timekeeper_pivot.to_excel(writer, sheet_name='Monthly Timekeeper', index=False)
        else:
            empty_timekeeper_pivot = pd.DataFrame({'Message': ['No data available for monthly timekeeper pivot table.']})
            empty_timekeeper_pivot.to_excel(writer, sheet_name='Monthly Timekeeper', index=False)
    
    print(f"Data saved to: {output_path}")

def analyze_task_categories(fee_detail_df):
    """
    Analyze and print summary of task categories
    """
    if not fee_detail_df.empty and 'task_category' in fee_detail_df.columns:
        print("\nTask Category Summary:")
        category_summary = fee_detail_df.groupby('task_category').agg({
            'hours': 'sum',
            'amount': 'sum'
        }).round(2)
        
        category_summary = category_summary.sort_values('hours', ascending=False)
        print(category_summary)
        
        print(f"\nTotal hours categorized: {category_summary['hours'].sum():.2f}")
        print(f"Total amount: ${category_summary['amount'].sum():.2f}")

def main():
    # Directory containing PDF files
    #pdf_directory = r"D:\OneDrive - 紫藤知识产权集团\Documents\Cadence\Invoices\125592-Invoices-08-14-2025\Outstanding"
    pdf_directory = r"D:\OneDrive - 紫藤知识产权集团\Documents\Cadence\Invoices\Cadence_latest invoices_1015"
    #pdf_directory = r"D:\OneDrive - 紫藤知识产权集团\Documents\Cadence\Invoices\125592-Invoices-08-14-2025\Paid"
    
    # Output Excel file
    output_excel = "invoice_data_analysis.xlsx"
    
    try:
        # Process all PDF files
        fee_summary_df, fee_detail_df = process_pdf_directory(pdf_directory)
        
        # Analyze task categories
        if not fee_detail_df.empty:
            analyze_task_categories(fee_detail_df)
        
        # Save to Excel
        save_to_excel(fee_summary_df, fee_detail_df, output_excel)
        
        # Print summary
        print(f"\nProcessing complete!")
        print(f"Fee Summary entries: {len(fee_summary_df)}")
        print(f"Fee Detail entries: {len(fee_detail_df)}")
        
        if not fee_summary_df.empty:
            print(f"\nFee Summary columns: {list(fee_summary_df.columns)}")
            # Show sample of matter descriptions
            if 'matter_description' in fee_summary_df.columns:
                unique_descriptions = fee_summary_df['matter_description'].unique()
                print(f"\nSample matter descriptions found:")
                for desc in unique_descriptions[:3]:  # Show first 3
                    print(f"  - {desc}")
        
        if not fee_detail_df.empty:
            print(f"Fee Detail columns: {list(fee_detail_df.columns)}")
            
    except Exception as e:
        print(f"Error in main process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()