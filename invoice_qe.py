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

def extract_quinn_invoice_date(text):
    """Extract invoice date from Quinn Emanuel text - first date in dd Month yyyy format"""
    # Look for the first date in the format "15 October 2025"
    date_pattern = r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})'
    matches = re.findall(date_pattern, text)
    
    if matches:
        return matches[0]  # Return the first date found
    return "Not Found"

def extract_quinn_invoice_no(text):
    """Extract invoice number from Quinn Emanuel text"""
    patterns = [
        r'Invoice No:\s*([A-Za-z0-9\-]+)',
        r'Invoice Number:\s*([A-Za-z0-9\-]+)',
        r'Invoice\s*#?\s*([A-Za-z0-9\-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return "Not Found"

def extract_quinn_matter_no(text):
    """Extract matter number from Quinn Emanuel text"""
    patterns = [
        r'Matter No:\s*([A-Za-z0-9.\-]+)',
        r'Our Matter No\.?\s*([A-Za-z0-9.\-]+)',
        r'Matter Number:\s*([A-Za-z0-9.\-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return "Not Found"

def extract_quinn_client_matter(text):
    """Extract client/matter description from Quinn Emanuel text"""
    patterns = [
        r'Matter No:.*?\n.*?\n(.*?)(?:\n|$)',
        r'Maxeon/Maoxing v Aiko',
        r'Responsible Attorney.*?\n(.*?)(?:\n|$)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "Not Found"

def parse_european_number(number_str):
    """Parse European number format (1.225,00 -> 1225.00)"""
    try:
        # Remove dots (thousand separators) and replace comma (decimal) with dot
        cleaned = number_str.replace('.', '').replace(',', '.')
        return float(cleaned)
    except:
        return 0.0

def parse_european_hours(hours_str):
    """Parse European hours format with comma as decimal separator (0,50 -> 0.50)"""
    try:
        # For hours, we just need to replace comma with dot
        # Hours don't typically have thousand separators
        return float(hours_str.replace(',', '.'))
    except:
        print(f"Warning: Could not parse hours value: {hours_str}")
        return 0.0

def extract_quinn_fee_summary(text, filename, invoice_date, invoice_no, matter_no, client_matter):
    """
    Extract fee summary data from Quinn Emanuel text with format:
    Attorneys Init. Title Hours Rate Amount
    Johannes Bukow JB6 Partner 40,20 1.225,00 49.245,00
    """
    fee_summary_data = []
    
    # Find the Fee Summary section
    fee_summary_match = re.search(r'Fee Summary(.*?)(?=Total Hours|Statement Detail|$)', text, re.IGNORECASE | re.DOTALL)
    
    if not fee_summary_match:
        # Try to find the table by header pattern
        fee_summary_match = re.search(r'Attorneys\s+Init\.?\s+Title\s+Hours\s+Rate\s+Amount(.*?)(?=Total Hours|Statement Detail|$)', text, re.IGNORECASE | re.DOTALL)
    
    if fee_summary_match:
        fee_summary_text = fee_summary_match.group(1)
        print(f"Found Fee Summary section: {fee_summary_text[:500]}...")  # Debug
        
        # More flexible pattern for Quinn Emanuel fee summary lines
        # "Johannes Bukow JB6 Partner 40,20 1.225,00 49.245,00"
        patterns = [
            r'^([A-Za-z]+(?:\s+[A-Za-z]+)+)\s+([A-Z0-9]+)\s+([A-Za-z]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)$',
            r'^([A-Za-z]+(?:\s+[A-Za-z]+)+)\s+([A-Z0-9]+)\s+([A-Za-z\s]+?)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)$',
        ]
        
        lines = fee_summary_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            print(f"Processing fee summary line: '{line}'")  # Debug
            
            parsed = False
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    lawyer_name = match.group(1).strip()
                    initials = match.group(2).strip()
                    title = match.group(3).strip()
                    hours_str = match.group(4).strip()
                    rate_str = match.group(5).strip()
                    amount_str = match.group(6).strip()
                    
                    # Parse European number formats
                    hours = parse_european_hours(hours_str)
                    rate = parse_european_number(rate_str)
                    amount = parse_european_number(amount_str)
                    
                    fee_summary_data.append({
                        'pdf_filename': filename,
                        'date_of_invoice': invoice_date,
                        'invoice_no': invoice_no,
                        'matter_no': matter_no,
                        'client_matter': client_matter,
                        'lawyer_name': lawyer_name,
                        'initials': initials,
                        'title': title,
                        'hours': hours,
                        'rate': rate,
                        'amount': amount
                    })
                    print(f"Successfully parsed: {lawyer_name} - {hours} hours at rate {rate}")  # Debug
                    parsed = True
                    break
            
            if not parsed:
                print(f"Warning: Could not parse fee summary line: '{line}'")
    
    else:
        print("Warning: Could not find Fee Summary section")
    
    return fee_summary_data

def extract_quinn_fee_detail(text, filename, fee_summary_data):
    """
    Extract fee detail data from Quinn Emanuel text - Statement Detail section
    """
    fee_detail_data = []
    
    # Create a mapping
    timekeeper_rate_map = {}
    for summary in fee_summary_data:
        timekeeper_rate_map[summary['initials']] = {
            'rate': summary['rate'],
            'lawyer_name': summary['lawyer_name'],
            'title': summary['title']
        }
    print(f"Created timekeeper rate map: {timekeeper_rate_map}")  # Debug

    # Find Statement Detail section
    #fee_detail_match = re.search(r'Statement Detail(.*?)(?=Total Hours\s+\d+[,.]\d{2}|Fee Summary|\Z)', text, re.IGNORECASE | re.DOTALL)
    
    #if not fee_detail_match:
        # Try alternative pattern
    fee_detail_match = re.search(r'Date\s+Timekeeper\s+Description\s+Hours(.*?)(?=Total Hours\s+\d+[,.]\d{2}|Fee Summary|\Z)', text, re.IGNORECASE | re.DOTALL)
    
    if fee_detail_match:
        fee_detail_text = fee_detail_match.group(1)
        print(f"Found Fee Detail section, length: {len(fee_detail_text)}")  # Debug
        
        #fee_detail_text = re.sub(r'\d{1,2}\s+[A-Za-z]+\s+\d{4}', '', fee_detail_text)

        #fee_detail_text = re.sub(r'quinn\s+emanuel\s+\|\s+germany.*Date Timekeeper Description Hours', '', fee_detail_text)
        fee_detail_text = re.sub(r'\squinn\s+emanuel\s*[|]\s*germany.*?Date\s*Timekeeper\s*Description\s*Hours', '', fee_detail_text, flags=re.IGNORECASE | re.DOTALL)
        print(f"Text: {(fee_detail_text)}")

        lines = fee_detail_text.split('\n')
        current_entry = None
        current_description = []
        in_multi_line_entry = False  # Flag to track multi-line entry state
        
        for i, line in enumerate(lines):
            line = re.sub(r'quinn\s+emanuel\s+\|\s+germany', '', line)
            line = line.strip()
            
            if not line or ' | ' in line or 'Invoice No:' in line or 'Matter No:' in line or \
                'Quinn Emanuel Urquhart' in line or '201996011004, with headquarters located' in line or 'laws of the State of California.' in line or \
                'employee or consultant with an equivalent status' in line or 'referred to as partners while not belonging to the partnership' in line or \
                'Date Timekeeper Description Hours' in line:
                continue
            #print(f">>>{line}")    
            # Skip header lines
            #if any(header in line.lower() for header in ['date', 'timekeeper', 'description', 'hours']):
            #    continue
            
            # Check if this line starts with a date pattern (DD/MM/YY)
            #line ='11/07/25 JB6 Review patent EP 627; update call with German team; summarizing questions and correspondence with client about'
            date_match = re.match(r'^(\d{2}/\d{2}/\d{2})[\s]+([A-Z0-9]{2,})\s+(.*)', line)
            #date_match = re.match(r'^\d{2}/\d{2}/\d{2}', date)
            print(date_match)
            if date_match:
                
                # Save previous entry if exists
                if current_entry and current_description:
                    # Join the accumulated description lines and look for hours at the end
                    full_description = ' '.join(current_description)
                    print(f"DEBUG: Processing accumulated description: {full_description}")
                    #hours_match = re.search(r'(\d+[,.]\d+$)', full_description)
                    hours_match = re.search(r'(\d+[,.]\d+)', full_description)

                    if hours_match:
                        hours_str = hours_match.group(1)
                        description = full_description[:hours_match.start()].strip()
                        current_entry['hours'] = parse_european_hours(hours_str)
                        current_entry['description'] = description

                        # Look up rate from fee summary based on timekeeper
                        timekeeper = current_entry['timekeeper']
                        rate_info = timekeeper_rate_map.get(timekeeper, {})
                        rate = rate_info.get('rate', 0.0)
                        lawyer_name = rate_info.get('lawyer_name', 'Unknown')
                        title = rate_info.get('title', 'Unknown')
                        
                        # Calculate amount
                        amount = hours * rate
                        
                        current_entry.update({
                            'hours': current_entry['hours'],
                            'description': description,
                            'rate': rate,
                            'lawyer_name': lawyer_name,
                            'title': title,
                            'amount': amount
                        })

                        fee_detail_data.append(current_entry)
                        print(f"Completed multi-line entry: {current_entry['date']} {current_entry['timekeeper']} {current_entry['hours']}h at rate {rate} = {amount}")
                    else:
                        hours_match = re.search(r'(\d+[,.]\d+)$', full_description) #try again to catch full line
                        if hours_match:
                            hours_str = hours_match.group(1)
                            description = full_description[:hours_match.start()].strip()
                            current_entry['hours'] = parse_european_hours(hours_str)
                            current_entry['description'] = description

                            # Look up rate from fee summary based on timekeeper
                            timekeeper = current_entry['timekeeper']
                            rate_info = timekeeper_rate_map.get(timekeeper, {})
                            rate = rate_info.get('rate', 0.0)
                            lawyer_name = rate_info.get('lawyer_name', 'Unknown')
                            title = rate_info.get('title', 'Unknown')
                            
                            # Calculate amount
                            amount = hours * rate
                            
                            current_entry.update({
                                'hours': current_entry['hours'],
                                'description': current_entry['description'],
                                'rate': rate,
                                'lawyer_name': lawyer_name,
                                'title': title,
                                'amount': amount
                            })

                            fee_detail_data.append(current_entry)
                            print(f"Last line entry: {current_entry['date']} {current_entry['timekeeper']} {current_entry['hours']}h at rate {rate} = {amount}")
                        else:
                            print(f"WARNING: No hours found for entry: {current_entry['date']} {current_entry['timekeeper']} {full_description}")

                
                # Start new entry
                date = date_match.group(1)
                timekeeper = date_match.group(2).strip()
                rest_of_line = date_match.group(3).strip()
                #rest_of_line = line[len(date) + len(timekeeper) + 2:]

                # Look up rate from fee summary based on timekeeper
                rate_info = timekeeper_rate_map.get(timekeeper, {})
                rate = rate_info.get('rate', 0.0)
                lawyer_name = rate_info.get('lawyer_name', 'Unknown')
                title = rate_info.get('title', 'Unknown')

                # Check if this line already contains hours
                hours_match = re.search(r'(\d+[,.]\d+)$', rest_of_line)
                if hours_match:
                    # Single line entry with hours
                    hours_str = hours_match.group(1)
                    description = rest_of_line[:hours_match.start()].strip()
                    hours = parse_european_hours(hours_str)

                    # Calculate amount
                    amount = hours * rate

                    entry = {
                        'pdf_filename': filename,
                        'date': date,
                        'timekeeper': timekeeper,
                        'hours': hours,
                        'description': description,
                        'rate': rate,
                        'lawyer_name': lawyer_name,
                        'title': title,
                        'amount': amount
                    }
                    fee_detail_data.append(entry)
                    current_entry = None
                    current_description = []
                    in_multi_line_entry = False  # Reset flag
                    print(f"Parsed single-line entry: {date} {timekeeper} {hours}h at rate {rate} = {amount} - {description[:50]}...")
                else:
                    # Multi-line entry - start collecting
                    current_entry = {
                        'pdf_filename': filename,
                        'date': date,
                        'timekeeper': timekeeper,
                        'hours': None,
                        'description': '',
                        'rate': rate,
                        'lawyer_name': lawyer_name,
                        'title': title,
                        'amount': 0.0
                    }
                    current_description = [rest_of_line]
                    in_multi_line_entry = True  # Set flag
                    print(f"Started multi-line entry: {date} {timekeeper} at rate {rate} - {rest_of_line[:20]}...{rest_of_line[-10:]}")
            
            # If we're in the middle of a multi-line entry, add to description
            elif in_multi_line_entry:
                # Skip date lines in format "dd B yyyy" even in continuation lines

                # Check if this line contains hours (end of entry)
                hours_match = re.search(r'(\d+[,.]\d+)$', line)
                if hours_match:
                    # This line contains hours - complete the entry immediately
                    hours_str = hours_match.group(1)
                    description_part = line[:hours_match.start()].strip()
                    
                    if description_part:
                        current_description.append(description_part)
                    
                    full_description = ' '.join(current_description)
                    current_entry['hours'] = parse_european_hours(hours_str)
                    current_entry['description'] = full_description
                    # Calculate amount using the rate we already stored
                    amount = hours * current_entry['rate']
                    
                    current_entry.update({
                        'hours': current_entry['hours'],
                        'description': current_entry['description'],
                        'amount': amount
                    })                    
                    fee_detail_data.append(current_entry)
                    print(f"Completed multi-line entry with hours detected: {current_entry['date']} {current_entry['timekeeper']} {current_entry['hours']}h at rate {current_entry['rate']} = {amount}")
                    
                    current_entry = None
                    current_description = []
                    in_multi_line_entry = False  # Reset flag
                else:
                    # This is a continuation line without hours
                    current_description.append(line)
                    print(f"Added continuation: {line[:250]}...")
        
        # Handle the last entry if we're still collecting
        if current_entry and current_description:
            full_description = ' '.join(current_description)
            hours_match = re.search(r'(\d+[,.]\d+)', full_description)
            
            if hours_match:
                hours_str = hours_match.group(1)
                description = full_description[:hours_match.start()].strip()
                current_entry['hours'] = parse_european_hours(hours_str)
                current_entry['description'] = description
                # Calculate amount using the rate we already stored
                amount = hours * current_entry['rate']
                
                current_entry.update({
                    'hours': current_entry['hours'],
                    'description': current_entry['description'],
                    'amount': amount
                })                
                fee_detail_data.append(current_entry)
                print(f"Completed final entry: {current_entry['date']} {current_entry['timekeeper']} {current_entry['hours']}h at rate {current_entry['rate']} = {amount}")
            else:
                print(f"Warning: Incomplete final entry: {current_entry['date']} {current_entry['timekeeper']}")
    
    else:
        print("Warning: Could not find Fee Detail section")
    
    return fee_detail_data

'''
def extract_quinn_fee_detail(text, filename):
    """
    Extract fee detail data from Quinn Emanuel text - Statement Detail section
    """
    fee_detail_data = []
    
    # Find Statement Detail section
    fee_detail_match = re.search(r'Statement Detail(.*?)(?=Total Hours\s+\d+[,.]\d{2}|Fee Summary|\Z)', text, re.IGNORECASE | re.DOTALL)
    
    if not fee_detail_match:
        # Try alternative pattern
        fee_detail_match = re.search(r'Date\s+Timekeeper\s+Description\s+Hours(.*?)(?=Total Hours\s+\d+[,.]\d{2}|Fee Summary|\Z)', text, re.IGNORECASE | re.DOTALL)
    
    if fee_detail_match:
        fee_detail_text = fee_detail_match.group(1)
        print(f"Found Fee Detail section, length: {len(fee_detail_text)}")  # Debug
        
        fee_detail_text = re.sub(r'quinn\s+emanuel\s+\|\s+germany$', '', fee_detail_text)
        #fee_detail_text = re.sub(r'\d{1,2}\s+[A-Za-z]+\s+\d{4}', '', fee_detail_text)

        print(f"Text: {(fee_detail_text)}")
        lines = fee_detail_text.split('\n')
        current_entry = None
        current_description = []
        
        for i, line in enumerate(lines):
            line = re.sub(r'quinn\s+emanuel\s+\|\s+germany', '', line)
            line = line.strip()
            
            if not line or ' | ' in line or 'Invoice No:' in line or 'Matter No:' in line or \
                'Quinn Emanuel Urquhart' in line or '201996011004, with headquarters located' in line or 'laws of the State of California.' in line or \
                'employee or consultant with an equivalent status' in line or 'referred to as partners while not belonging to the partnership' in line or \
                'Date Timekeeper Description Hours' in line:
                continue
            print(f">>>{line}")    
            # Skip header lines
            if any(header in line.lower() for header in ['date', 'timekeeper', 'description', 'hours']):
                continue
            
            # Check if this line starts with a date pattern (DD/MM/YY)
            date_match = re.match(r'^(\d{2}/\d{2}/\d{2})\s+([A-Z0-9]{2,})\s+(.*)', line)
            
            if date_match:
                # Save previous entry if exists
                if current_entry and current_description:
                    # Join the accumulated description lines and look for hours at the end
                    full_description = ' '.join(current_description)
                    print(f"DEBUG: Processing accumulated description: {full_description}")
                    hours_match = re.search(r'(\d+[,.]\d+)$', full_description)
                    if not hours_match:
                        # Try with spaces around the hours
                        hours_match = re.search(r'\s+(\d+[,.]\d+)$', full_description)

                    if hours_match:
                        hours_str = hours_match.group(1)
                        description = full_description[:hours_match.start()].strip()
                        current_entry['hours'] = parse_european_hours(hours_str)
                        current_entry['description'] = description
                        fee_detail_data.append(current_entry)
                        print(f"Completed multi-line entry: {current_entry['date']} {current_entry['timekeeper']} {current_entry['hours']}h")
                    else:
                        hours_match = re.search(r'(\d+[,.]\d+)$', full_description) #try again to catch full line
                        if hours_match:
                            hours_str = hours_match.group(1)
                            description = full_description[:hours_match.start()].strip()
                            current_entry['hours'] = parse_european_hours(hours_str)
                            current_entry['description'] = description
                            fee_detail_data.append(current_entry)
                            print(f"Last line entry: {current_entry['date']} {current_entry['timekeeper']} {current_entry['hours']}h")
                        else:
                            print(f"WARNING: No hours found for entry: {current_entry['date']} {current_entry['timekeeper']} {full_description}")

                
                # Start new entry
                date = date_match.group(1)
                timekeeper = date_match.group(2).strip()
                rest_of_line = date_match.group(3).strip()
                
                # Check if this line already contains hours
                hours_match = re.search(r'(\d+[,.]\d+)$', rest_of_line)
                if hours_match:
                    # Single line entry with hours
                    hours_str = hours_match.group(1)
                    description = rest_of_line[:hours_match.start()].strip()
                    hours = parse_european_hours(hours_str)
                    
                    entry = {
                        'pdf_filename': filename,
                        'date': date,
                        'timekeeper': timekeeper,
                        'hours': hours,
                        'description': description
                    }
                    fee_detail_data.append(entry)
                    current_entry = None
                    current_description = []
                    print(f"Parsed single-line entry: {date} {timekeeper} {hours}h - {description[:50]}...")
                else:
                    # Multi-line entry - start collecting
                    current_entry = {
                        'pdf_filename': filename,
                        'date': date,
                        'timekeeper': timekeeper,
                        'hours': None,
                        'description': ''
                    }
                    current_description = [rest_of_line]
                    print(f"Started multi-line entry: {date} {timekeeper} - {rest_of_line[:20]}...{rest_of_line[:-10]}")
            
            # If we're in the middle of a multi-line entry, add to description
            elif current_entry is not None:
                # Skip date lines in format "dd B yyyy" even in continuation lines

                # Check if this line contains hours (end of entry)
                #hours_match = re.search(r'(\d+[,.]\d+)$', line)
                hours_match = re.search(r'(\d+[,.]\d+)', line)
                if hours_match:
                    # This line ends with hours - complete the entry
                    hours_str = hours_match.group(1)
                    description_part = line[:hours_match.start()].strip()
                    
                    if description_part:
                        current_description.append(description_part)
                    
                    full_description = ' '.join(current_description)
                    current_entry['hours'] = parse_european_hours(hours_str)
                    current_entry['description'] = full_description
                    fee_detail_data.append(current_entry)
                    print(f"Completed multi-line entry: {current_entry['date']} {current_entry['timekeeper']} {current_entry['hours']}h")
                    
                    current_entry = None
                    current_description = []
                else:
                    # This is a continuation line without hours
                    current_description.append(line)
                    print(f"Added continuation: {line[:250]}...")
        
        # Handle the last entry if we're still collecting
        if current_entry and current_description:
            full_description = ' '.join(current_description)
            #hours_match = re.search(r'(\d+[,.]\d+)$', full_description)
            hours_match = re.search(r'(\d+[,.]\d+)', full_description)
            
            if hours_match:
                hours_str = hours_match.group(1)
                description = full_description[:hours_match.start()].strip()
                current_entry['hours'] = parse_european_hours(hours_str)
                current_entry['description'] = description
                fee_detail_data.append(current_entry)
                print(f"Completed final entry: {current_entry['date']} {current_entry['timekeeper']} {current_entry['hours']}h")
            else:
                print(f"Warning: Incomplete final entry: {current_entry['date']} {current_entry['timekeeper']}")
    
    else:
        print("Warning: Could not find Fee Detail section")
    
    return fee_detail_data
'''
def parse_european_hours(hours_str):
    """Parse European number format with comma as decimal separator"""
    try:
        # Replace comma with dot for float conversion
        return float(hours_str.replace(',', '.'))
    except ValueError:
        print(f"Warning: Could not parse hours value: {hours_str}")
        return 0.0

'''
def extract_quinn_fee_detail(text, filename):
    """
    Extract fee detail data from Quinn Emanuel text - Statement Detail section
    """
    fee_detail_data = []
    
    # Find Statement Detail section
    fee_detail_match = re.search(r'Statement Detail(.*?)(?=Total Hours\s+\d+[,.]\d{2}|Fee Summary|\Z)', text, re.IGNORECASE | re.DOTALL)
    
    if not fee_detail_match:
        # Try alternative pattern
        fee_detail_match = re.search(r'Date\s+Timekeeper\s+Description\s+Hours(.*?)(?=Total Hours\s+\d+[,.]\d{2}|Fee Summary|\Z)', text, re.IGNORECASE | re.DOTALL)
    
    if fee_detail_match:
        fee_detail_text = fee_detail_match.group(1)
        print(f"Found Fee Detail section, length: {len(fee_detail_text)}")  # Debug
        
        # Improved pattern to handle multi-line descriptions
        lines = fee_detail_text.split('\n')
        current_entry = None
        collecting_description = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Skip header lines
            if any(header in line.lower() for header in ['date', 'timekeeper', 'description', 'hours']):
                continue
            
            # Check if this line starts with a date pattern (DD/MM/YY)
            date_match = re.match(r'^(\d{2}/\d{2}/\d{2})\s+([A-Z]{2})\s+(.*)', line)
            
            if date_match:
                # Save previous entry if exists
                if current_entry:
                    fee_detail_data.append(current_entry)
                
                # Start new entry
                date = date_match.group(1)
                timekeeper = date_match.group(2).strip()
                description_start = date_match.group(3).strip()
                
                # Check if this line already contains hours at the end
                hours_match = re.search(r'(\d+[,.]\d+)$', description_start)
                if hours_match:
                    # Extract hours from description
                    hours_str = hours_match.group(1)
                    description = description_start[:hours_match.start()].strip()
                    hours = parse_european_hours(hours_str)
                    
                    current_entry = {
                        'pdf_filename': filename,
                        'date': date,
                        'timekeeper': timekeeper,
                        'hours': hours,
                        'description': description
                    }
                    print(f"Parsed fee detail with hours: {date} {timekeeper} {hours}h - {description[:50]}...")
                    current_entry = None  # Reset since we have complete entry
                else:
                    # Start collecting multi-line description
                    current_entry = {
                        'pdf_filename': filename,
                        'date': date,
                        'timekeeper': timekeeper,
                        'hours': None,
                        'description': description_start
                    }
                    collecting_description = True
                    print(f"Started fee detail: {date} {timekeeper} - {description_start[:50]}...")
            
            # Check if we're collecting description and this line contains hours
            elif collecting_description and current_entry:
                hours_match = re.search(r'(\d+[,.]\d+)$', line)
                if hours_match:
                    # Found hours at the end of the line
                    hours_str = hours_match.group(1)
                    description_continuation = line[:hours_match.start()].strip()
                    
                    # Append the final part of description
                    if description_continuation:
                        current_entry['description'] += ' ' + description_continuation
                    
                    current_entry['hours'] = parse_european_hours(hours_str)
                    fee_detail_data.append(current_entry)
                    print(f"Completed fee detail: {current_entry['date']} {current_entry['timekeeper']} {current_entry['hours']}h")
                    current_entry = None
                    collecting_description = False
                else:
                    # This is a continuation line of description
                    current_entry['description'] += ' ' + line
                    print(f"Added continuation to description: {line[:50]}...")
        
        # Don't forget the last entry if we're still collecting
        if current_entry and current_entry['hours'] is not None:
            fee_detail_data.append(current_entry)
        elif current_entry:
            print(f"Warning: Incomplete entry at end: {current_entry['date']} {current_entry['timekeeper']}")
    
    else:
        print("Warning: Could not find Fee Detail section")
    
    return fee_detail_data




def extract_quinn_fee_detail(text, filename):
    """
    Extract fee detail data from Quinn Emanuel text - Statement Detail section
    """
    fee_detail_data = []
    
    # Find Statement Detail section
    fee_detail_match = re.search(r'Statement Detail(.*?)(?=Total Hours\s+\d+[,.]\d{2}|Fee Summary|\Z)', text, re.IGNORECASE | re.DOTALL)
    
    if not fee_detail_match:
        # Try alternative pattern
        fee_detail_match = re.search(r'Date\s+Timekeeper\s+Description\s+Hours(.*?)(?=Total Hours\s+\d+[,.]\d{2}|Fee Summary|\Z)', text, re.IGNORECASE | re.DOTALL)
    
    if fee_detail_match:
        fee_detail_text = fee_detail_match.group(1)
        print(f"Found Fee Detail section, length: {len(fee_detail_text)}")  # Debug
        
        # Quinn Emanuel pattern: "01/09/25 JB6 Follow-up with team... 0,50"
        # More flexible patterns to handle variations
        patterns = [
            r'^(\d{2}/\d{2}/\d{2})\s+([A-Z0-9]+)\s+(.*?)\s+(\d+[,.]\d+)$',
            r'^(\d{1,2}/\d{1,2}/\d{2,4})\s+([A-Z0-9]+)\s+(.*?)\s+(\d+[,.]\d+)$',
        ]
        
        lines = fee_detail_text.split('\n')
        current_entry = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Skip header lines
            if any(header in line.lower() for header in ['date', 'timekeeper', 'description', 'hours']):
                continue
                
            # Check if this line starts with a date (DD/MM/YY or similar)
            parsed = False
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous entry if exists
                    if current_entry:
                        fee_detail_data.append(current_entry)
                    
                    # Start new entry
                    date = match.group(1)
                    timekeeper = match.group(2).strip()
                    description = match.group(3).strip()
                    hours_str = match.group(4)
                    
                    hours = parse_european_hours(hours_str)
                    
                    current_entry = {
                        'pdf_filename': filename,
                        'date': date,
                        'timekeeper': timekeeper,
                        'hours': hours,
                        'description': description
                    }
                    print(f"Parsed fee detail: {date} {timekeeper} {hours}h - {description[:50]}...")  # Debug
                    parsed = True
                    break
            
            if not parsed:
                # This might be a continuation line, append to current description
                if current_entry and line and not re.match(r'^\d+[,.]\d+$', line.strip()):
                    current_entry['description'] += ' ' + line.strip()
                    print(f"Added continuation to description: {line[:50]}...")  # Debug
                else:
                    print(f"Warning: Could not parse fee detail line {i}: '{line}'")
        
        # Don't forget the last entry
        if current_entry:
            fee_detail_data.append(current_entry)
    
    else:
        print("Warning: Could not find Fee Detail section")
    
    return fee_detail_data
    '''
def categorize_task_quinn(description):
    """
    Categorize task based on description for Quinn Emanuel - patent litigation focused
    """
    if not description:
        return "Other"
    
    try:
        description_lower = description.lower()
        
        # Patent litigation specific task categories
        if any(word in description_lower for word in ['test', 'lab', 'experiment', 'iv test', 'testing', 'zsw', 'csem', 'laboratori']):
            return "testing"
        elif any(word in description_lower for word in ['claim', 'infringement analysi', 'chart', 'element mapping']):
            return "claim charting"
        elif any(word in description_lower for word in ['draft complaint', 'complaint', 'brief', 'motion', 'plead']):
            return "complaint drafting"
        elif any(word in description_lower for word in ['evidence', 'proof', 'document', 'expert report', 'present evidence']):
            return "evidence preparation"
        elif any(word in description_lower for word in ['client', 'email client', 'call client', 'discuss with client', 'purplevine']):
            return "client communication"
        elif any(word in description_lower for word in ['research', 'legal research', 'case law', 'statutor', 'regulator']):
            return "legal research"
        elif any(word in description_lower for word in ['review', 'analyze', 'analysis', 'examin']):
            return "document review"
        elif any(word in description_lower for word in ['strateg', 'plan', 'approach', 'next step']):
            return "strategy development"
        elif any(word in description_lower for word in ['expert', 'witness', 'testifi', 'csem']):
            return "expert coordination"
        elif any(word in description_lower for word in ['confer', 'meet', 'discuss', 'call with']):
            return "meet and confer"
        else:
            return "Other"
    except Exception as e:
        print(f"Error in task categorization: {e}")
        return "Other"

def extract_quinn_invoice_data(pdf_path):
    """
    Extract invoice data from Quinn Emanuel PDF file
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        return extract_quinn_data_from_text(text, pdf_path.name)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None, None

def extract_quinn_data_from_text(text, filename):
    """
    Extract fee summary and fee detail data from Quinn Emanuel invoice text
    """
    print(f"\nProcessing file: {filename}")  # Debug
    
    # Extract header information
    invoice_date = extract_quinn_invoice_date(text)
    invoice_no = extract_quinn_invoice_no(text)
    matter_no = extract_quinn_matter_no(text)
    client_matter = extract_quinn_client_matter(text)
    
    print(f"Extracted - Date: {invoice_date}, Invoice: {invoice_no}, Matter: {matter_no}")  # Debug
    
    # Extract fee summary data
    fee_summary_data = extract_quinn_fee_summary(text, filename, invoice_date, invoice_no, matter_no, client_matter)
    
    # Extract fee detail data
    fee_detail_data = extract_quinn_fee_detail(text, filename, fee_summary_data)
    
    # Categorize tasks in fee detail
    for entry in fee_detail_data:
        entry['task_category'] = categorize_task_quinn(entry['description'])
    
    return fee_summary_data, fee_detail_data

def process_quinn_pdf_directory(directory_path):
    """
    Process all Quinn Emanuel PDF files in the directory and return dataframes
    """
    all_fee_summary = []
    all_fee_detail = []
    
    pdf_files = list(Path(directory_path).glob('*.pdf'))
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        fee_summary, fee_detail = extract_quinn_invoice_data(pdf_file)
        
        if fee_summary:
            all_fee_summary.extend(fee_summary)
            print(f"Found {len(fee_summary)} fee summary entries")
        
        if fee_detail:
            all_fee_detail.extend(fee_detail)
            print(f"Found {len(fee_detail)} fee detail entries")
    
    # Create dataframes
    fee_summary_df = pd.DataFrame(all_fee_summary) if all_fee_summary else pd.DataFrame()
    fee_detail_df = pd.DataFrame(all_fee_detail) if all_fee_detail else pd.DataFrame()
    
    return fee_summary_df, fee_detail_df

def create_monthly_timekeeper_pivot(fee_summary_df, fee_detail_df):
    """
    Create monthly pivot tables for timekeeper hours and amounts
    Returns two dataframes: monthly_hours_pivot, monthly_amount_pivot
    """
    if fee_summary_df.empty or fee_detail_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Create a combined dataframe for analysis
    combined_data = []
    
    for _, fee_detail_row in fee_detail_df.iterrows():
        # Find matching fee summary entry to get rate and full name
        matching_summary = fee_summary_df[
            (fee_summary_df['pdf_filename'] == fee_detail_row['pdf_filename']) & 
            (fee_summary_df['initials'] == fee_detail_row['timekeeper'])
        ]
        
        if not matching_summary.empty:
            rate = matching_summary.iloc[0]['rate']
            lawyer_name = matching_summary.iloc[0]['lawyer_name']
            title = matching_summary.iloc[0]['title']
            matter_no = matching_summary.iloc[0]['matter_no']
            invoice_date = matching_summary.iloc[0]['date_of_invoice']
            
            # Parse the work date (DD/MM/YY format)
            try:
                work_date = datetime.strptime(fee_detail_row['date'], '%d/%m/%y')
                year_month = work_date.strftime('%Y-%m')
                month_name = work_date.strftime('%B %Y')
            except:
                year_month = "Unknown"
                month_name = "Unknown"
            
            amount = fee_detail_row['hours'] * rate
            
            combined_data.append({
                'pdf_filename': fee_detail_row['pdf_filename'],
                'matter_no': matter_no,
                'invoice_date': invoice_date,
                'work_date': fee_detail_row['date'],
                'year_month': year_month,
                'month_name': month_name,
                'timekeeper': fee_detail_row['timekeeper'],
                'lawyer_name': lawyer_name,
                'title': title,
                'hours': fee_detail_row['hours'],
                'rate': rate,
                'amount': amount,
                'description': fee_detail_row['description'],
                'task_category': fee_detail_row['task_category']
            })
    
    if not combined_data:
        return pd.DataFrame(), pd.DataFrame()
    
    combined_df = pd.DataFrame(combined_data)
    
    # Create monthly hours pivot
    monthly_hours_pivot = combined_df.pivot_table(
        index=['year_month', 'month_name'],
        columns='lawyer_name',
        values='hours',
        aggfunc='sum',
        fill_value=0
    ).round(2)
    
    # Add total row and column
    monthly_hours_pivot['Total Hours'] = monthly_hours_pivot.sum(axis=1)
    monthly_hours_pivot.loc['Total'] = monthly_hours_pivot.sum()
    
    # Create monthly amount pivot
    monthly_amount_pivot = combined_df.pivot_table(
        index=['year_month', 'month_name'],
        columns='lawyer_name',
        values='amount',
        aggfunc='sum',
        fill_value=0
    ).round(2)
    
    # Add total row and column
    monthly_amount_pivot['Total Amount'] = monthly_amount_pivot.sum(axis=1)
    monthly_amount_pivot.loc['Total'] = monthly_amount_pivot.sum()
    
    return monthly_hours_pivot, monthly_amount_pivot

def create_detailed_monthly_summary(fee_summary_df, fee_detail_df):
    """
    Create a detailed monthly summary with timekeeper breakdown
    """
    if fee_summary_df.empty or fee_detail_df.empty:
        return pd.DataFrame()
    
    detailed_data = []
    
    for _, fee_detail_row in fee_detail_df.iterrows():
        # Find matching fee summary entry
        matching_summary = fee_summary_df[
            (fee_summary_df['pdf_filename'] == fee_detail_row['pdf_filename']) & 
            (fee_summary_df['initials'] == fee_detail_row['timekeeper'])
        ]
        
        if not matching_summary.empty:
            rate = matching_summary.iloc[0]['rate']
            lawyer_name = matching_summary.iloc[0]['lawyer_name']
            title = matching_summary.iloc[0]['title']
            matter_no = matching_summary.iloc[0]['matter_no']
            
            # Parse work date
            try:
                work_date = datetime.strptime(fee_detail_row['date'], '%d/%m/%y')
                year_month = work_date.strftime('%Y-%m')
                month_name = work_date.strftime('%B %Y')
            except:
                year_month = "Unknown"
                month_name = "Unknown"
            
            amount = fee_detail_row['hours'] * rate
            
            detailed_data.append({
                'Month': month_name,
                'Year-Month': year_month,
                'Matter No': matter_no,
                'Timekeeper': lawyer_name,
                'Title': title,
                'Hours': fee_detail_row['hours'],
                'Rate': rate,
                'Amount': amount,
                'Task Category': fee_detail_row['task_category'],
                'Description': fee_detail_row['description'][:100] + '...' if len(fee_detail_row['description']) > 100 else fee_detail_row['description']
            })
    
    if not detailed_data:
        return pd.DataFrame()
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # Create summary by month and timekeeper
    summary_df = detailed_df.groupby(['Year-Month', 'Month', 'Timekeeper', 'Title']).agg({
        'Hours': 'sum',
        'Amount': 'sum'
    }).reset_index()
    
    # Sort by year-month and timekeeper
    summary_df = summary_df.sort_values(['Year-Month', 'Timekeeper'])
    
    return summary_df

def main_quinn():
    """
    Main function for processing Quinn Emanuel invoices
    """
    # Directory containing Quinn Emanuel PDF files
    pdf_directory = r"D:\OneDrive - 紫藤知识产权集团\Documents\Maxeon\invoices"
    
    # Output Excel file
    output_excel = "quinn_emanuel_invoice_analysis.xlsx"
    
    try:
        # Process all Quinn Emanuel PDF files
        fee_summary_df, fee_detail_df = process_quinn_pdf_directory(pdf_directory)
        
        # Create monthly pivot tables
        monthly_hours_pivot, monthly_amount_pivot = create_monthly_timekeeper_pivot(fee_summary_df, fee_detail_df)
        
        # Create detailed monthly summary
        detailed_monthly_summary = create_detailed_monthly_summary(fee_summary_df, fee_detail_df)
        
        # Save to Excel
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            if not fee_summary_df.empty:
                fee_summary_df.to_excel(writer, sheet_name='Fee Summary', index=False)
                print(f"\nSaved {len(fee_summary_df)} fee summary entries")
                
                # Display sample of extracted data
                print("\nSample Fee Summary Data:")
                print(fee_summary_df[['lawyer_name', 'hours', 'rate', 'amount']].head())
            
            if not fee_detail_df.empty:
                fee_detail_df.to_excel(writer, sheet_name='Fee Detail', index=False)
                print(f"Saved {len(fee_detail_df)} fee detail entries")
            
            # Save monthly pivot tables
            if not monthly_hours_pivot.empty:
                monthly_hours_pivot.to_excel(writer, sheet_name='Monthly Hours Pivot')
                print(f"Saved Monthly Hours Pivot table")
            
            if not monthly_amount_pivot.empty:
                monthly_amount_pivot.to_excel(writer, sheet_name='Monthly Amount Pivot')
                print(f"Saved Monthly Amount Pivot table")
            
            # Save detailed monthly summary
            if not detailed_monthly_summary.empty:
                detailed_monthly_summary.to_excel(writer, sheet_name='Detailed Monthly Summary', index=False)
                print(f"Saved Detailed Monthly Summary with {len(detailed_monthly_summary)} entries")
        
        print(f"\nQuinn Emanuel Processing complete!")
        print(f"Output saved to: {output_excel}")
        
        # Print summary statistics
        if not monthly_hours_pivot.empty:
            print(f"\nMonthly Hours Summary:")
            print(monthly_hours_pivot[['Total Hours']])
            
        if not monthly_amount_pivot.empty:
            print(f"\nMonthly Amount Summary:")
            print(monthly_amount_pivot[['Total Amount']])
        
    except Exception as e:
        print(f"Error in Quinn Emanuel main process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_quinn()