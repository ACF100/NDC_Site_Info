import streamlit as st
import pandas as pd
import os
import requests
import json
import time
import pickle
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import re
import warnings
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

@dataclass
class ProductInfo:
    ndc: str
    product_name: str
    labeler_name: str
    spl_id: Optional[str] = None
    fei_numbers: List[str] = None
    establishments: List[Dict] = None

@dataclass
class FEIMatch:
    fei_number: str
    xml_location: str
    match_type: str  # 'FEI_NUMBER' or 'DUNS_NUMBER'
    establishment_name: str = None
    xml_context: str = None  # Surrounding XML context

class NDCToLocationMapper:
    def __init__(self):
        self.base_openfda_url = "https://api.fda.gov"
        self.dailymed_base_url = "https://dailymed.nlm.nih.gov/dailymed"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FDA-Research-Tool/1.0 (research@fda.gov)'
        })

        # Initialize empty databases
        self.fei_database = {}
        self.duns_database = {}
        self.inspection_database = {}
        self.database_loaded = False
        self.database_date = None
        
        # Load optimized databases (fast!)
        self.load_optimized_databases()

    def load_optimized_databases(self):
        """Load pre-processed databases (2-5 seconds vs 30-60 seconds)"""
        
        # Try to load optimized databases in order of preference
        optimized_files = [
            'fda_databases_optimized.pkl.gz',  # Compressed (fastest)
            'data/fda_databases_optimized.pkl.gz',
            'fda_databases_optimized.pkl',     # Uncompressed
            'data/fda_databases_optimized.pkl'
        ]
        
        for file_path in optimized_files:
            if os.path.exists(file_path):
                try:
                    if file_path.endswith('.gz'):
                        import gzip
                        with gzip.open(file_path, 'rb') as f:
                            data = pickle.load(f)
                    else:
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                    
                    # Load the processed data
                    self.fei_database = data['fei_database']
                    self.duns_database = data['duns_database']
                    self.inspection_database = data.get('inspection_database', {})
                    self.database_date = data.get('database_date', 'Unknown')
                    self.database_loaded = True
                    
                    return
                    
                except Exception as e:
                    continue
        
        # If no optimized files found, fallback to CSV loading
        st.warning("⚠️ Optimized database not found. Loading from CSV files (this will take longer)...")
        self.load_database_automatically()

    def load_database_automatically(self):
        """Automatically load both establishment and inspection databases"""
        try:
            # Load establishment database
            establishment_files = [
                "drls_reg.csv",
                "data/drls_reg.csv", 
                "./drls_reg.csv",
                "../drls_reg.csv",
                "drls_reg.xlsx",
                "data/drls_reg.xlsx", 
                "./drls_reg.xlsx",
                "../drls_reg.xlsx"
            ]
            
            # Load inspection database  
            inspection_files = [
                "inspection_outcomes_reg.csv.gz",
                "data/inspection_outcomes_reg.csv.gz",
                "./inspection_outcomes_reg.csv.gz", 
                "../inspection_outcomes_reg.csv.gz",
                "inspection_outcomes_reg.csv",
                "data/inspection_outcomes_reg.csv",
                "./inspection_outcomes_reg.csv", 
                "../inspection_outcomes_reg.csv",
                "inspection_outcomes.xlsx",
                "data/inspection_outcomes.xlsx",
                "./inspection_outcomes.xlsx", 
                "../inspection_outcomes.xlsx"
            ]
            
            # Try to load establishment database
            for file_path in establishment_files:
                if os.path.exists(file_path):
                    self.load_fei_database_from_spreadsheet(file_path)
                    if self.fei_database or self.duns_database:
                        self.database_loaded = True
                        try:
                            mod_time = os.path.getmtime(file_path)
                            self.database_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d")
                        except:
                            self.database_date = "Unknown"
                        break
            
            # Try to load inspection database
            for file_path in inspection_files:
                if os.path.exists(file_path):
                    if self.load_inspection_database_from_spreadsheet(file_path):
                        st.success(f"✅ Loaded inspection outcomes database: {len(self.inspection_database):,} FEI records")
                        break
            
            if not hasattr(self, 'inspection_database'):
                self.inspection_database = {}
                
            # Show error if no establishment database found
            if not self.database_loaded:
                st.error("❌ Could not load establishment database from any source")
                
        except Exception as e:
            st.error(f"❌ Error during database loading: {str(e)}")

    def load_fei_database_from_spreadsheet(self, file_path: str):
        """Load FEI and DUNS database from a spreadsheet with cross-linked identifiers"""
        try:
            # Try to read the file with different engines
            try:
                # Force all columns to be read as strings to preserve leading zeros
                df = pd.read_excel(file_path, dtype=str)
            except:
                try:
                    df = pd.read_csv(file_path, dtype=str)
                except Exception as e:
                    return

            # Look for FEI_NUMBER, DUNS_NUMBER, ADDRESS, and FIRM_NAME columns (case insensitive, flexible matching)
            fei_col = None
            duns_col = None
            address_col = None
            firm_name_col = None

            for col in df.columns:
                col_lower = col.lower().strip().replace('_', '').replace(' ', '')
                col_original = col.strip()
                
                # More flexible FEI column matching
                if ('fei' in col_lower and 'number' in col_lower) or col_lower == 'feinumber':
                    fei_col = col_original
                # More flexible DUNS column matching - EXCLUDE registrant DUNS explicitly
                elif ('duns' in col_lower and 'number' in col_lower) or col_lower == 'dunsnumber':
                    # CRITICAL: Exclude any column with registrant/owner/parent in the name
                    if not any(word in col_lower for word in ['registrant', 'owner', 'parent', 'company']):
                        duns_col = col_original
                        print(f"DEBUG: Selected DUNS column: {col_original}")
                    else:
                        print(f"DEBUG: Skipping registrant/owner DUNS column: {col_original}")
                # More flexible ADDRESS column matching
                elif 'address' in col_lower:
                    address_col = col_original
                # More flexible FIRM_NAME column matching
                elif ('firm' in col_lower and 'name' in col_lower) or col_lower == 'firmname':
                    firm_name_col = col_original

            print(f"DEBUG: Final columns - FEI: {fei_col}, DUNS: {duns_col}, Address: {address_col}, Firm: {firm_name_col}")

            if not fei_col and not duns_col:
                return

            if not address_col:
                return

            # Process each row
            fei_count = 0
            duns_count = 0
            
            for idx, row in df.iterrows():
                try:
                    address = str(row[address_col]).strip()
                    
                    # Skip empty address rows
                    if pd.isna(row[address_col]) or address == 'nan' or address == '':
                        continue

                    # Parse address components
                    address_parts = self.parse_address(address)
                    
                    # Get firm name
                    firm_name = 'Unknown'
                    if firm_name_col and not pd.isna(row[firm_name_col]):
                        firm_name = str(row[firm_name_col]).strip()
                        if firm_name == 'nan' or firm_name == '':
                            firm_name = 'Unknown'

                    # Get BOTH FEI and FACILITY DUNS from the same row (NOT registrant DUNS)
                    fei_number = None
                    facility_duns_number = None
                    
                    if fei_col and not pd.isna(row[fei_col]):
                        fei_raw = str(row[fei_col]).strip()
                        if fei_raw != 'nan' and fei_raw != '' and fei_raw != '0000000' and fei_raw != '0000000000':
                            fei_clean = re.sub(r'[^\d]', '', fei_raw)
                            if len(fei_clean) >= 7:
                                fei_number = fei_raw

                    # CRITICAL FIX: Only use facility DUNS, not registrant DUNS
                    if duns_col and not pd.isna(row[duns_col]):
                        duns_raw = str(row[duns_col]).strip()
                        if duns_raw != 'nan' and duns_raw != '':
                            duns_clean = re.sub(r'[^\d]', '', duns_raw)
                            if len(duns_clean) >= 8:
                                facility_duns_number = duns_raw

                    # Only process if we have at least one identifier
                    if not fei_number and not facility_duns_number:
                        continue

                    # Create establishment data with BOTH identifiers
                    establishment_data = {
                        'establishment_name': address_parts.get('establishment_name', 'Unknown'),
                        'firm_name': firm_name,
                        'address_line_1': address_parts.get('address_line_1', address),
                        'city': address_parts.get('city', 'Unknown'),
                        'state_province': address_parts.get('state_province', 'Unknown'),
                        'country': address_parts.get('country', 'Unknown'),
                        'postal_code': address_parts.get('postal_code', ''),
                        'latitude': address_parts.get('latitude'),
                        'longitude': address_parts.get('longitude'),
                        'search_method': 'spreadsheet_database',
                        'original_fei': fei_number,                    # Store FEI
                        'original_duns': facility_duns_number          # Store FACILITY DUNS only
                    }

                    # Store under FEI variants if FEI exists
                    if fei_number:
                        fei_variants = self._generate_all_id_variants(fei_number)
                        for key in fei_variants:
                            if key:
                                self.fei_database[key] = establishment_data.copy()
                        fei_count += 1

                    # Store under FACILITY DUNS variants if FACILITY DUNS exists (NOT registrant DUNS)
                    if facility_duns_number:
                        duns_variants = self._generate_all_id_variants(facility_duns_number)
                        for key in duns_variants:
                            if key:
                                self.duns_database[key] = establishment_data.copy()
                        duns_count += 1

                    # Store under DUNS variants if DUNS exists
                    if duns_number:
                        duns_variants = self._generate_all_id_variants(duns_number)
                        for key in duns_variants:
                            if key:
                                self.duns_database[key] = establishment_data.copy()
                        duns_count += 1

                except Exception as e:
                    continue

        except Exception as e:
            pass

    def load_inspection_database_from_spreadsheet(self, file_path: str):
        """Load inspection outcomes database from spreadsheet (supports .gz compression)"""
        try:
            # Handle compressed files
            if file_path.endswith('.gz'):
                import gzip
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    df = pd.read_csv(f, dtype=str)
            else:
                # Try to read the file with different engines
                try:
                    df = pd.read_csv(file_path, dtype=str)
                except:
                    df = pd.read_excel(file_path, dtype=str)
            
            # Create inspection database
            self.inspection_database = {}
            
            for idx, row in df.iterrows():
                try:
                    fei_number = str(row.get('FEI Number', '')).strip()
                    if not fei_number or fei_number == 'nan':
                        continue
                    
                    # Clean FEI number
                    fei_clean = re.sub(r'[^\d]', '', fei_number)
                    if len(fei_clean) < 7:
                        continue
                    
                    inspection_record = {
                        'fei_number': fei_number,
                        'legal_name': str(row.get('Legal Name', '')).strip(),
                        'city': str(row.get('City', '')).strip(),
                        'state': str(row.get('State', '')).strip(),
                        'zip': str(row.get('Zip', '')).strip(),
                        'country': str(row.get('Country/Area', '')).strip(),
                        'fiscal_year': str(row.get('Fiscal Year', '')).strip(),
                        'inspection_id': str(row.get('Inspection ID', '')).strip(),
                        'posted_citations': str(row.get('Posted Citations', '')).strip(),
                        'inspection_end_date': str(row.get('Inspection End Date', '')).strip(),
                        'classification': str(row.get('Classification', '')).strip(),
                        'project_area': str(row.get('Project Area', '')).strip(),
                        'product_type': str(row.get('Product Type', '')).strip(),
                        'additional_details': str(row.get('Additional Details', '')).strip(),
                        'fmd_145_date': str(row.get('FMD-145 Date', '')).strip()
                    }
                    
                    # Store under all FEI variants
                    fei_variants = self._generate_all_id_variants(fei_number)
                    for variant in fei_variants:
                        if variant not in self.inspection_database:
                            self.inspection_database[variant] = []
                        self.inspection_database[variant].append(inspection_record)
                        
                except Exception as e:
                    continue
                    
            return len(self.inspection_database) > 0
            
        except Exception as e:
            return False

    def get_inspection_summary(self, inspections: List[Dict]) -> Dict:
        """Generate simplified summary showing most recent inspection and any OAI dates"""
        if not inspections:
            return {
                'total_records': 0,
                'most_recent_date': None,
                'most_recent_outcome': None,
                'oai_dates': [],
                'status': 'No inspection records found'
            }
        
        # Sort by date to get most recent (handle different date formats)
        def parse_date(date_str):
            if not date_str or date_str == 'Date unknown':
                return ''
            # Remove timestamp if present
            date_only = str(date_str).split(' ')[0]
            
            # Convert MM/DD/YYYY to YYYY-MM-DD for proper sorting
            if '/' in date_only:
                try:
                    parts = date_only.split('/')
                    if len(parts) == 3:
                        # Convert MM/DD/YYYY to YYYY-MM-DD
                        return f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                except:
                    pass
            return date_only
        
        sorted_inspections = sorted(inspections, 
                                  key=lambda x: parse_date(x.get('inspection_date', '')), 
                                  reverse=True)
        most_recent = sorted_inspections[0]
        
        # Get the most recent date and outcome (remove timestamp)
        most_recent_date = parse_date(most_recent.get('inspection_date', 'Date unknown'))
        most_recent_classification = most_recent.get('classification', 'Outcome unknown')
        
        # Categorize classification names - FIXED to match your exact data format
        classification_map = {
            'No Action Indicated (NAI)': 'Compliant (No Action Indicated)',
            'Voluntary Action Indicated (VAI)': 'Compliant (Voluntary Action Indicated)', 
            'Official Action Indicated (OAI)': 'Unacceptable Compliance (Official Action Indicated)',
            'No Action Indicated': 'Compliant (No Action Indicated)',
            'Voluntary Action Indicated': 'Compliant (Voluntary Action Indicated)',
            'Official Action Indicated': 'Unacceptable Compliance (Official Action Indicated)'
        }
        
        categorized_classification = classification_map.get(most_recent_classification, most_recent_classification)
        
        # Find all OAI inspection dates - FIXED to match your exact format
        oai_dates = []
        for inspection in sorted_inspections:
            classification = inspection.get('classification', '').strip()
            # Check for OAI in your exact data format
            if classification == 'Official Action Indicated (OAI)':
                oai_date = parse_date(inspection.get('inspection_date', ''))
                if oai_date and oai_date not in oai_dates:
                    oai_dates.append(oai_date)
        
        # Build status string
        if most_recent_date != 'Date unknown' and most_recent_date:
            status = f"{categorized_classification} {most_recent_date}"
        else:
            status = categorized_classification
        
        # Add OAI history if any exist
        if oai_dates:
            if len(oai_dates) == 1:
                status += f" | History of Unacceptable Compliance: Official Action Indicated {oai_dates[0]}"
            else:
                status += f" | History of Unacceptable Compliance: Official Action Indicated {', '.join(oai_dates)}"
        
        return {
            'total_records': len(inspections),
            'most_recent_date': most_recent_date,
            'most_recent_outcome': categorized_classification,
            'oai_dates': oai_dates,
            'status': status
        }

    def _generate_all_id_variants(self, id_number: str) -> List[str]:
        """Generate all possible variants of an ID number for matching"""
        clean_id = re.sub(r'[^\d]', '', str(id_number))
        variants = []
        
        # Add original formats
        variants.extend([
            str(id_number).strip(),
            clean_id,
            clean_id.lstrip('0')
        ])
        
        # Add numeric conversion variants
        try:
            id_as_int = int(clean_id)
            # Add padded versions for different lengths
            for padding in [8, 9, 10, 11, 12, 13, 14, 15]:
                padded = f"{id_as_int:0{padding}d}"
                variants.append(padded)
            
            # Add string of int
            variants.append(str(id_as_int))
            
        except ValueError:
            pass
        
        # Special handling for numbers that might have been stored with/without leading zeros
        if clean_id.startswith('00'):
            # For numbers starting with 00, try removing different amounts of leading zeros
            variants.append(clean_id[1:])  # Remove one zero
            variants.append(clean_id[2:])  # Remove two zeros
        elif clean_id.startswith('0'):
            # For numbers starting with 0, try removing the leading zero
            variants.append(clean_id[1:])
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys([v for v in variants if v]))

    def parse_address(self, address: str) -> Dict:
        """Parse address string into components"""
        try:
            # Basic address parsing - you can enhance this based on your data format
            parts = {
                'establishment_name': 'Unknown',
                'address_line_1': address,
                'city': 'Unknown',
                'state_province': 'Unknown',
                'country': 'Unknown',
                'postal_code': '',
                'latitude': None,
                'longitude': None
            }

            # Try to extract establishment name (first line before comma or newline)
            lines = address.replace('\\n', ',').split(',')
            if len(lines) > 0:
                parts['establishment_name'] = lines[0].strip()

            # Try to extract city, state, country from last parts
            if len(lines) >= 2:
                parts['address_line_1'] = lines[1].strip() if len(lines) > 1 else lines[0].strip()

            if len(lines) >= 3:
                # Look for city in second to last part
                city_part = lines[-2].strip()
                parts['city'] = city_part

            if len(lines) >= 4:
                # Look for state/country in last part
                last_part = lines[-1].strip()
                parts['state_province'] = last_part

                # Common country patterns
                if any(country in last_part.upper() for country in ['USA', 'US', 'UNITED STATES']):
                    parts['country'] = 'USA'
                elif any(country in last_part.upper() for country in ['GERMANY', 'DEUTSCHLAND']):
                    parts['country'] = 'Germany'
                elif any(country in last_part.upper() for country in ['SWITZERLAND', 'SCHWEIZ']):
                    parts['country'] = 'Switzerland'
                elif any(country in last_part.upper() for country in ['SINGAPORE']):
                    parts['country'] = 'Singapore'
                else:
                    parts['country'] = last_part

            # Extract postal code (look for patterns like 12345 or 12345-6789)
            postal_match = re.search(r'\b(\d{5}(?:-\d{4})?|\d{4,6})\b', address)
            if postal_match:
                parts['postal_code'] = postal_match.group(1)

            return parts

        except Exception as e:
            return {
                'establishment_name': 'Unknown',
                'address_line_1': address,
                'city': 'Unknown',
                'state_province': 'Unknown',
                'country': 'Unknown',
                'postal_code': '',
                'latitude': None,
                'longitude': None
            }

    def validate_ndc_format(self, ndc: str) -> bool:
        """Validate NDC format - more flexible to accept various formats"""
        ndc = str(ndc).strip()
        
        # Remove any non-digit, non-dash characters
        clean_ndc = re.sub(r'[^\d\-]', '', ndc)
        
        # Check if it's a valid NDC format
        patterns = [
            r'^\d{4,5}-\d{3,4}-\d{1,2}$',  # Standard format with dashes
            r'^\d{10,11}$',                 # All digits, 10 or 11 digits
            r'^\d{8,9}$'                    # Sometimes shorter formats exist
        ]
        
        # Also accept if it becomes valid after normalization
        if any(re.match(pattern, clean_ndc) for pattern in patterns):
            return True
        
        # Try to normalize and see if it becomes valid
        try:
            normalized = self.normalize_ndc(clean_ndc)
            return any(re.match(pattern, normalized) for pattern in patterns[:2])
        except:
            pass
        
        # Accept any string of 8-11 digits
        digits_only = re.sub(r'[^\d]', '', ndc)
        return len(digits_only) >= 8 and len(digits_only) <= 11

    def normalize_ndc(self, ndc: str) -> str:
        """Normalize NDC to standard format - FIXED for all input formats"""
        # Remove any non-digit, non-dash characters
        clean_ndc = re.sub(r'[^\d\-]', '', str(ndc))
        
        # If it already has dashes and is valid, return as-is
        if '-' in clean_ndc:
            # Check if it's already in valid format
            if re.match(r'^\d{4,5}-\d{3,4}-\d{1,2}$', clean_ndc):
                return clean_ndc
            # If dashes are in wrong places, remove them and reformat
            clean_ndc = clean_ndc.replace('-', '')
        
        # Work with digits only - this handles the case where NDC is entered without hyphens
        digits_only = clean_ndc
        
        # FIXED: Handle all possible NDC lengths correctly
        if len(digits_only) == 11:
            # 11-digit format: 5-4-2
            return f"{digits_only[:5]}-{digits_only[5:9]}-{digits_only[9:]}"
        elif len(digits_only) == 10:
            # 10-digit format: could be 5-3-2 or 4-4-2
            # Most common is to pad to 11 digits and format as 5-4-2
            padded = '0' + digits_only
            return f"{padded[:5]}-{padded[5:9]}-{padded[9:]}"
        elif len(digits_only) == 9:
            # 9-digit format: pad to 11 and format
            padded = '00' + digits_only
            return f"{padded[:5]}-{padded[5:9]}-{padded[9:]}"
        elif len(digits_only) == 8:
            # 8-digit format: pad to 11 and format
            padded = '000' + digits_only
            return f"{padded[:5]}-{padded[5:9]}-{padded[9:]}"
        else:
            # Return original if we can't format it properly
            return clean_ndc

    def normalize_ndc_for_matching(self, ndc: str) -> List[str]:
        """Generate multiple NDC formats for matching - COMPLETELY FIXED"""
        clean_ndc = re.sub(r'[^\d\-]', '', str(ndc))
        variants = set()  # Use set to avoid duplicates
        
        # Remove dashes to get base digits
        digits_only = clean_ndc.replace('-', '')
        
        # Add the original digits
        variants.add(digits_only)
        
        # FIXED: Handle NDC input without hyphens more comprehensively
        # For a string like "0069005801", we need to try different interpretations:
        
        if len(digits_only) == 10:
            # 10-digit NDC could be interpreted as:
            # 1. 5-3-2 format: first 5 digits are labeler, next 3 are product, last 2 are package
            # 2. 4-4-2 format: first 4 digits are labeler, next 4 are product, last 2 are package
            
            # Interpretation 1: 5-3-2 (pad product to 4 digits)
            labeler_5 = digits_only[:5]
            product_3 = digits_only[5:8] 
            package_2 = digits_only[8:]
            
            # Create 5-4-2 by padding product
            product_4_padded = '0' + product_3
            variant_5_4_2 = labeler_5 + product_4_padded + package_2
            variants.add(variant_5_4_2)
            variants.add(f"{labeler_5}-{product_4_padded}-{package_2}")
            variants.add(f"{labeler_5}-{product_3}-{package_2}")  # Original 5-3-2
            
            # Interpretation 2: 4-4-2 (pad labeler to 5 digits)
            labeler_4 = digits_only[:4]
            product_4 = digits_only[4:8]
            package_2 = digits_only[8:]
            
            # Create 5-4-2 by padding labeler
            labeler_5_padded = '0' + labeler_4
            variant_5_4_2_alt = labeler_5_padded + product_4 + package_2
            variants.add(variant_5_4_2_alt)
            variants.add(f"{labeler_5_padded}-{product_4}-{package_2}")
            variants.add(f"{labeler_4}-{product_4}-{package_2}")  # Original 4-4-2
            
            # If product starts with 0, also try without the leading 0
            if product_4.startswith('0'):
                product_3_unpadded = product_4[1:]
                variants.add(f"{labeler_5_padded}-{product_3_unpadded}-{package_2}")
                variants.add(labeler_5_padded + product_3_unpadded + package_2)

        elif len(digits_only) == 11:
            # 11-digit format - try different segment interpretations
            # Standard 5-4-2
            labeler_5 = digits_only[:5]
            product_4 = digits_only[5:9]
            package_2 = digits_only[9:]
            
            variants.add(f"{labeler_5}-{product_4}-{package_2}")
            
            # Try 5-3-2 by removing leading zero from product if it starts with 0
            if product_4.startswith('0'):
                product_3 = product_4[1:]
                variant_5_3_2 = labeler_5 + product_3 + package_2
                variants.add(variant_5_3_2)
                variants.add(f"{labeler_5}-{product_3}-{package_2}")
            
            # Try 4-4-2 by removing leading zero from labeler if it starts with 0
            if labeler_5.startswith('0'):
                labeler_4 = labeler_5[1:]
                variant_4_4_2 = labeler_4 + product_4 + package_2
                variants.add(variant_4_4_2)
                variants.add(f"{labeler_4}-{product_4}-{package_2}")

        # Handle other lengths
        if len(digits_only) == 8:
            # Pad to different lengths
            variants.add('000' + digits_only)  # 11 digits
            variants.add('00' + digits_only)   # 10 digits  
            variants.add('0' + digits_only)    # 9 digits
        elif len(digits_only) == 9:
            variants.add('00' + digits_only)   # 11 digits
            variants.add('0' + digits_only)    # 10 digits
            if digits_only.startswith('0'):
                variants.add(digits_only[1:])  # 8 digits (remove leading zero)

        # Generate ALL possible formatted versions
        formatted_variants = set()
        for variant in list(variants):
            variant_clean = variant.replace('-', '')
            
            if len(variant_clean) == 11:
                formatted_variants.add(f"{variant_clean[:5]}-{variant_clean[5:9]}-{variant_clean[9:]}")
            elif len(variant_clean) == 10:
                # Try both 5-3-2 and 4-4-2 interpretations
                formatted_variants.add(f"{variant_clean[:5]}-{variant_clean[5:8]}-{variant_clean[8:]}")  # 5-3-2
                formatted_variants.add(f"{variant_clean[:4]}-{variant_clean[4:8]}-{variant_clean[8:]}")  # 4-4-2
            elif len(variant_clean) == 9:
                formatted_variants.add(f"{variant_clean[:4]}-{variant_clean[4:7]}-{variant_clean[7:]}")
            elif len(variant_clean) == 8:
                formatted_variants.add(f"{variant_clean[:4]}-{variant_clean[4:6]}-{variant_clean[6:]}")
        
        # Combine all variants
        all_variants = variants.union(formatted_variants)
        
        # Add base NDC variants (labeler-product without package)
        base_variants = set()
        for variant in formatted_variants:
            if '-' in variant and variant.count('-') == 2:
                parts = variant.split('-')
                if len(parts) == 3:  # Standard NDC format
                    base_ndc = f"{parts[0]}-{parts[1]}"  # Remove package part
                    base_variants.add(base_ndc)
                    
                    # Also add base NDC without dashes
                    base_ndc_no_dash = f"{parts[0]}{parts[1]}"
                    base_variants.add(base_ndc_no_dash)
        
        all_variants = all_variants.union(base_variants)
        
        # Convert to list and remove empty strings
        return [v for v in all_variants if v and len(v) >= 6]

    def extract_labeler_from_product_name(self, product_name: str) -> str:
        """Extract labeler name from product name - enhanced extraction"""
        try:
            # Method 1: Look for text in brackets at the end
            bracket_match = re.search(r'\[([^\]]+)\]\s*$', product_name)
            if bracket_match:
                labeler = bracket_match.group(1).strip()
                if labeler and labeler.lower() not in ['unknown', 'n/a', 'none']:
                    return labeler
            
            # Method 2: Look for any brackets in the product name
            all_brackets = re.findall(r'\[([^\]]+)\]', product_name)
            if all_brackets:
                # Take the last bracketed text (usually the manufacturer)
                labeler = all_brackets[-1].strip()
                if labeler and labeler.lower() not in ['unknown', 'n/a', 'none']:
                    return labeler
            
            # Method 3: Look for text after "by" or "from"
            by_match = re.search(r'\b(?:by|from)\s+([^,\[\]]+)', product_name, re.IGNORECASE)
            if by_match:
                labeler = by_match.group(1).strip()
                if labeler and labeler.lower() not in ['unknown', 'n/a', 'none']:
                    return labeler
            
            return 'Not specified'
            
        except Exception as e:
            return 'Not specified'

    def get_ndc_info_comprehensive(self, ndc: str) -> Optional[ProductInfo]:
        """Get NDC info from multiple sources"""
        # Try DailyMed first
        dailymed_info = self.get_ndc_info_from_dailymed(ndc)
        if dailymed_info:
            return dailymed_info

        # Try openFDA as fallback
        openfda_info = self.get_ndc_info_from_openfda(ndc)
        if openfda_info:
            return openfda_info

        return None

    def get_ndc_info_from_dailymed(self, ndc: str) -> Optional[ProductInfo]:
        """Get NDC info from DailyMed with improved labeler extraction"""
        try:
            # Generate comprehensive list of NDC variants
            ndc_variants = self.normalize_ndc_for_matching(ndc)
            
            # Also try the original and basic normalizations
            additional_variants = [
                ndc.replace('-', ''),
                ndc,
                self.normalize_ndc(ndc),
                self.normalize_ndc_11digit(ndc),
                self.normalize_ndc_10digit(ndc)
            ]
            
            # Combine and deduplicate
            all_variants = list(set(ndc_variants + additional_variants))
            
            # Try each variant
            for ndc_variant in all_variants:
                if not ndc_variant or len(ndc_variant) < 6:
                    continue
                    
                try:
                    search_url = f"{self.dailymed_base_url}/services/v2/spls.json"
                    params = {'ndc': ndc_variant, 'page_size': 1}
                    response = self.session.get(search_url, params=params)

                    if response.status_code == 200:
                        data = response.json()
                        if data.get('data'):
                            spl_data = data['data'][0]
                            product_name = spl_data.get('title', 'Unknown')
                            
                            # Try multiple methods to get labeler
                            labeler_name = None
                            
                            # Method 1: From API labeler field
                            api_labeler = spl_data.get('labeler', '').strip()
                            if api_labeler and api_labeler not in ['Unknown', '', 'None']:
                                labeler_name = api_labeler
                            
                            # Method 2: Extract from product name
                            if not labeler_name:
                                labeler_name = self.extract_labeler_from_product_name(product_name)
                            
                            # Method 3: Try to get from SPL XML directly
                            if labeler_name in ['Not specified', 'Unknown', ''] and spl_data.get('setid'):
                                spl_labeler, _ = self.extract_labeler_from_spl(spl_data.get('setid'))
                                if spl_labeler and spl_labeler != 'Unknown':
                                    labeler_name = spl_labeler
                            
                            # Final fallback
                            if not labeler_name or labeler_name in ['Unknown', 'Not specified', '']:
                                labeler_name = 'Labeler name not available'
                            
                            return ProductInfo(
                                ndc=ndc,  # Return original NDC as entered
                                product_name=product_name,
                                labeler_name=labeler_name,
                                spl_id=spl_data.get('setid')
                            )
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass

        return None

    def get_ndc_info_from_openfda(self, ndc: str) -> Optional[ProductInfo]:
        """Get NDC info from openFDA - try more variants"""
        try:
            # Generate comprehensive list of NDC variants
            ndc_variants = self.normalize_ndc_for_matching(ndc)
            
            # Also try the original and basic normalizations
            additional_variants = [
                ndc.replace('-', ''),
                ndc,
                self.normalize_ndc(ndc),
                self.normalize_ndc_11digit(ndc),
                self.normalize_ndc_10digit(ndc)
            ]
            
            # Combine and deduplicate
            all_variants = list(set(ndc_variants + additional_variants))

            for ndc_variant in all_variants:
                if not ndc_variant or len(ndc_variant) < 6:
                    continue
                    
                try:
                    url = f"{self.base_openfda_url}/drug/label.json"
                    params = {'search': f'openfda.product_ndc:"{ndc_variant}"', 'limit': 1}
                    response = self.session.get(url, params=params)

                    if response.status_code == 200:
                        data = response.json()
                        if data.get('results'):
                            result = data['results'][0]
                            openfda = result.get('openfda', {})

                            brand_names = openfda.get('brand_name', [])
                            generic_names = openfda.get('generic_name', [])
                            manufacturer_names = openfda.get('manufacturer_name', [])

                            product_name = (brand_names[0] if brand_names else
                                          generic_names[0] if generic_names else 'Unknown')
                            labeler_name = manufacturer_names[0] if manufacturer_names else 'Unknown'

                            return ProductInfo(ndc=ndc, product_name=product_name, labeler_name=labeler_name)
                except Exception as e:
                    continue
        except Exception as e:
            pass

        return None

    def normalize_ndc_11digit(self, ndc: str) -> str:
        """Convert NDC to 11-digit format"""
        clean_ndc = ndc.replace('-', '')
        return '0' + clean_ndc if len(clean_ndc) == 10 else clean_ndc

    def normalize_ndc_10digit(self, ndc: str) -> str:
        """Convert NDC to 10-digit format"""
        clean_ndc = ndc.replace('-', '')
        return clean_ndc[1:] if len(clean_ndc) == 11 and clean_ndc.startswith('0') else clean_ndc

    def lookup_fei_establishment(self, fei_number: str) -> Optional[Dict]:
        """Look up establishment information using FEI number from spreadsheet database"""
        try:
            # Try EXPANDED formats for FEI lookup
            fei_variants = self._generate_all_id_variants(fei_number)

            for fei_variant in fei_variants:
                if fei_variant in self.fei_database:
                    establishment_info = self.fei_database[fei_variant].copy()
                    establishment_info['fei_number'] = fei_variant
                    return establishment_info
                    
            return None
        except Exception as e:
            return None

    def lookup_duns_establishment(self, duns_number: str) -> Optional[Dict]:
        """Look up establishment information using DUNS number from spreadsheet database"""
        try:
            
            duns_variants = self._generate_all_id_variants(duns_number)
            

            for duns_variant in duns_variants:
                if duns_variant in self.duns_database:
                    establishment_info = self.duns_database[duns_variant].copy()
                    
                    establishment_info['duns_number'] = duns_variant
                    return establishment_info
                    
            
            return None
        except Exception as e:
            
            return None

    def find_fei_duns_matches_in_spl(self, spl_id: str) -> List[FEIMatch]:
        """Find FEI and DUNS numbers - COMPREHENSIVE approach to find ALL establishments"""
        matches = []
        
        try:
            spl_url = f"{self.dailymed_base_url}/services/v2/spls/{spl_id}.xml"
            response = self.session.get(spl_url)

            if response.status_code != 200:
                return matches

            content = response.text
            processed_ids = set()
            
            # STRATEGY 1: Find ALL ID elements with 7+ digit extensions
            all_id_pattern = r'<id[^>]*extension="(\d{7,15})"[^>]*>'
            all_id_matches = re.finditer(all_id_pattern, content, re.IGNORECASE)
            
            st.write("🔍 **All ID elements found in SPL:**")
            
            for id_match in all_id_matches:
                extension = id_match.group(1)
                clean_extension = re.sub(r'[^\d]', '', extension)
                
                # Skip if already processed
                if clean_extension in processed_ids:
                    continue
                
                # Get context around this ID
                context_start = max(0, id_match.start() - 300)
                context_end = min(len(content), id_match.end() + 300)
                context = content[context_start:context_end]
                
                # Extract establishment name from context
                establishment_name = self._extract_name_from_context_regex(context)
                
                st.write(f"- ID: {extension} | Name: {establishment_name}")
                
                # Check if this ID exists in our databases
                fei_match = self._check_database_match(extension, 'FEI_NUMBER')
                duns_match = self._check_database_match(extension, 'DUNS_NUMBER')
                
                if fei_match or duns_match:
                    match_type = 'FEI_NUMBER' if fei_match else 'DUNS_NUMBER'
                    
                    # Calculate line number for location
                    line_num = content[:id_match.start()].count('\n') + 1
                    xml_location = f"Line {line_num}"
                    
                    match = FEIMatch(
                        fei_number=clean_extension,
                        xml_location=xml_location,
                        match_type=match_type,
                        establishment_name=establishment_name,
                        xml_context=context[:200] + "..." if len(context) > 200 else context
                    )
                    
                    matches.append(match)
                    processed_ids.add(clean_extension)
                    
                    st.write(f"  ✅ **FOUND IN DATABASE** as {match_type}")
                else:
                    st.write(f"  ❌ Not found in database")
            
            st.write(f"\n**Total establishments found in database: {len(matches)}**")
            
            return matches
            
        except Exception as e:
            st.write(f"Error in find_fei_duns_matches_in_spl: {e}")
            return matches

    def _is_within_author_section(self, element) -> bool:
        """Check if element is within author section (labeler context)"""
        current = element
        while current is not None:
            if current.tag.endswith('author'):
                return True
            current = current.getparent() if hasattr(current, 'getparent') else None
        return False

    def _get_element_xpath(self, element, root) -> str:
        """Generate XPath-like location for an element"""
        try:
            path_parts = []
            current = element
            
            # Build path by walking up the tree
            while current is not None and current != root:
                tag = current.tag.split('}')[-1] if '}' in current.tag else current.tag
                
                # Count siblings with same tag to get position
                parent = current.getparent() if hasattr(current, 'getparent') else None
                if parent is not None:
                    siblings = [sibling for sibling in parent if sibling.tag == current.tag]
                    if len(siblings) > 1:
                        index = siblings.index(current) + 1
                        path_parts.insert(0, f"{tag}[{index}]")
                    else:
                        path_parts.insert(0, tag)
                else:
                    path_parts.insert(0, tag)
                    
                current = parent
                
            return "/" + "/".join(path_parts) if path_parts else "unknown_xpath"
        except Exception as e:
            return "xpath_error"

    def _get_element_context(self, element, root) -> str:
        """Get surrounding context for an element"""
        try:
            context_parts = []
            
            # Get parent element information
            parent = element.getparent() if hasattr(element, 'getparent') else None
            if parent is not None:
                parent_tag = parent.tag.split('}')[-1] if '}' in parent.tag else parent.tag
                context_parts.append(f"Parent: {parent_tag}")
                
                # Look for name elements in parent
                for child in parent:
                    child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if 'name' in child_tag.lower() and child.text:
                        context_parts.append(f"Name: {child.text.strip()}")
                        break
            
            # Get element attributes
            attrs = []
            for key, value in element.attrib.items():
                key_clean = key.split('}')[-1] if '}' in key else key
                attrs.append(f"{key_clean}='{value}'")
            
            if attrs:
                context_parts.append(f"Attributes: {', '.join(attrs)}")
                
            return " | ".join(context_parts)
        except Exception as e:
            return "context_unavailable"

    def _find_matches_flexible_xml(self, root, processed_ids: set) -> List[FEIMatch]:
        """Flexible XML parsing that works with any SPL structure"""
        matches = []
        
        # Find ALL id elements in the document
        for id_elem in root.iter():
            if not id_elem.tag.endswith('id') or not id_elem.get('extension'):
                continue
                
            extension = id_elem.get('extension')
            clean_extension = re.sub(r'[^\d]', '', extension)
            
            # Skip if already processed or too short
            if clean_extension in processed_ids or len(clean_extension) < 7:
                continue
            
            # Determine context - is this an establishment or labeler?
            context_type = self._determine_context_type(id_elem)
            
            # Skip main labelers, keep establishments
            if context_type == 'MAIN_LABELER':
                continue
            elif context_type in ['ESTABLISHMENT', 'UNKNOWN']:
                # Check if this ID exists in our databases
                match = self._create_match_from_xml_element(id_elem, extension, clean_extension, root)
                if match:
                    matches.append(match)
                    processed_ids.add(clean_extension)
        
        return matches

    def _find_matches_flexible_regex(self, content: str, processed_ids: set) -> List[FEIMatch]:
        """Flexible regex-based matching for any SPL structure"""
        matches = []
        
        # Find all ID elements that might be establishments
        id_pattern = r'<id\s+([^>]*extension="(\d{7,15})"[^>]*)>'
        id_matches = re.finditer(id_pattern, content, re.IGNORECASE)
        
        for id_match in id_matches:
            extension = id_match.group(2)
            clean_extension = re.sub(r'[^\d]', '', extension)
            
            # Skip if already processed
            if clean_extension in processed_ids:
                continue
            
            # Get context around this ID to determine if it's an establishment
            context_start = max(0, id_match.start() - 500)
            context_end = min(len(content), id_match.end() + 500)
            context = content[context_start:context_end]
            
            # Skip if this looks like a main labeler
            if self._is_main_labeler_context(context):
                continue
            
            # Try to find establishment name
            establishment_name = self._extract_name_from_context_regex(context)
            
            # Check if this ID exists in our databases
            match = self._create_match_from_regex_flexible(extension, clean_extension, id_match, establishment_name, context)
            if match:
                matches.append(match)
                processed_ids.add(clean_extension)
        
        return matches

    def _determine_context_type(self, id_elem) -> str:
        """Determine if an ID element represents a labeler or establishment"""
        # Walk up the XML tree to understand context
        current = id_elem
        path = []
        
        for _ in range(10):  # Limit traversal depth
            current = current.getparent() if hasattr(current, 'getparent') else None
            if current is None:
                break
            
            tag = current.tag.split('}')[-1] if '}' in current.tag else current.tag
            path.append(tag)
            
            # Main labeler patterns
            if tag == 'representedOrganization' and 'author' in path:
                return 'MAIN_LABELER'
            
            # Establishment patterns
            if tag == 'assignedOrganization':
                # Check if this is nested under author (could be establishment)
                if 'author' in path:
                    return 'ESTABLISHMENT'
                # Or if it's in other contexts
                return 'ESTABLISHMENT'
        
        return 'UNKNOWN'

    def _is_main_labeler_context(self, context: str) -> bool:
        """Check if context suggests this is a main labeler rather than establishment"""
        # Look for representedOrganization pattern (main labeler)
        if '<representedOrganization' in context and '<author' in context:
            # Check if there's no assignedOrganization between them
            repr_pos = context.find('<representedOrganization')
            author_pos = context.find('<author')
            if author_pos < repr_pos:
                between_text = context[author_pos:repr_pos]
                if '<assignedOrganization' not in between_text:
                    return True
        return False

    def _create_match_from_xml_element(self, id_elem, extension: str, clean_extension: str, root) -> Optional[FEIMatch]:
        """Create match with database lookup"""
        # Try FEI database first
        fei_variants = self._generate_all_id_variants(extension)
        for fei_key in fei_variants:
            if fei_key in self.fei_database:
                return FEIMatch(
                    fei_number=clean_extension,
                    xml_location=self._get_element_xpath(id_elem, root),
                    match_type='FEI_NUMBER',
                    establishment_name=self._extract_establishment_name_from_context(id_elem),
                    xml_context=self._get_element_context(id_elem, root)
                )
        
        # Try DUNS database
        duns_variants = self._generate_all_id_variants(extension)
        for duns_key in duns_variants:
            if duns_key in self.duns_database:
                return FEIMatch(
                    fei_number=clean_extension,
                    xml_location=self._get_element_xpath(id_elem, root),
                    match_type='DUNS_NUMBER',
                    establishment_name=self._extract_establishment_name_from_context(id_elem),
                    xml_context=self._get_element_context(id_elem, root)
                )
        
        return None

    def _create_match_from_regex_flexible(self, extension: str, clean_extension: str, id_match, establishment_name: str, context: str) -> Optional[FEIMatch]:
        """Create match from regex parsing with database lookup"""
        # Calculate line number for location
        line_num = id_match.string[:id_match.start()].count('\n') + 1
        xml_location = f"Line {line_num} (regex)"
        
        # Try FEI database first
        fei_variants = self._generate_all_id_variants(extension)
        for fei_key in fei_variants:
            if fei_key in self.fei_database:
                return FEIMatch(
                    fei_number=clean_extension,
                    xml_location=xml_location,
                    match_type='FEI_NUMBER',
                    establishment_name=establishment_name,
                    xml_context=context[:200] + "..." if len(context) > 200 else context
                )
        
        # Try DUNS database
        duns_variants = self._generate_all_id_variants(extension)
        for duns_key in duns_variants:
            if duns_key in self.duns_database:
                return FEIMatch(
                    fei_number=clean_extension,
                    xml_location=xml_location,
                    match_type='DUNS_NUMBER',
                    establishment_name=establishment_name,
                    xml_context=context[:200] + "..." if len(context) > 200 else context
                )
        
        return None

    def _extract_establishments_ndc_specific(self, content: str, target_ndc: str) -> List[Dict]:
        """NDC-SPECIFIC approach using XML hierarchy - handles both representedOrganization and assignedOrganization"""
        establishments = {}
        
        # Generate all possible NDC variants for matching
        ndc_variants = self.normalize_ndc_for_matching(target_ndc)
        
        # Complete operation mapping
        operation_codes = {
            'C25391': 'Analysis', 'C25394': 'API Manufacture', 
            'C43360': 'Manufacture', 'C82401': 'Manufacture', 'C43359': 'Manufacture',
            'C84731': 'Pack', 'C84732': 'Label', 'C48482': 'Repack',
            'C73606': 'Relabel', 'C25392': 'Sterilize'
        }
        
        # FIXED: Look for BOTH assignedOrganization AND representedOrganization
        # Pattern 1: assignedOrganization (manufacturing establishments)
        assigned_pattern = r'<[^:]*:?assignedOrganization[^>]*>.*?</[^:]*:?assignedOrganization>'
        assigned_matches = re.finditer(assigned_pattern, content, re.DOTALL | re.IGNORECASE)
        
        # Pattern 2: representedOrganization (main labelers who also manufacture)
        represented_pattern = r'<[^:]*:?representedOrganization[^>]*>.*?</[^:]*:?representedOrganization>'
        represented_matches = re.finditer(represented_pattern, content, re.DOTALL | re.IGNORECASE)
        
        # Process both types of organizations
        all_org_matches = list(assigned_matches) + list(represented_matches)
        
        for org_match in all_org_matches:
            org_section = org_match.group(0)
            
            # Extract establishment ID from this organization
            id_match = re.search(r'<[^:]*:?id[^>]*extension="(\d{7,15})"', org_section)
            if not id_match:
                continue
                
            establishment_id = id_match.group(1)
            
            # Skip if already processed (avoid duplicates)
            if establishment_id in establishments:
                continue
            
            # Extract establishment name
            name_match = re.search(r'<[^:]*:?name[^>]*>([^<]+)</[^:]*:?name>', org_section)
            establishment_name = name_match.group(1).strip() if name_match else "Unknown"
            
            # Find ALL performance elements within THIS organization section
            perf_pattern = r'<[^:]*:?performance[^>]*>.*?</[^:]*:?performance>'
            perf_elements = re.findall(perf_pattern, org_section, re.DOTALL | re.IGNORECASE)
            
            operations = []
            for perf_elem in perf_elements:
                # Check if this performance element mentions our target NDC
                ndc_pattern = r'<[^:]*:?code[^>]*code="([^"]*)"[^>]*codeSystem="2\.16\.840\.1\.113883\.6\.69"'
                ndc_matches = re.findall(ndc_pattern, perf_elem, re.IGNORECASE)
                
                ndc_found = False
                for ndc_code in ndc_matches:
                    ndc_variants_found = self.normalize_ndc_for_matching(ndc_code.strip())
                    if any(v in ndc_variants for v in ndc_variants_found):
                        ndc_found = True
                        break
                
                if ndc_found:
                    # Extract operation from this performance element
                    op_pattern = r'<[^:]*:?code[^>]*code="(C\d+)"[^>]*displayName="([^"]*)"'
                    op_match = re.search(op_pattern, perf_elem)
                    if op_match:
                        op_code = op_match.group(1)
                        display_name = op_match.group(2).lower()
                        
                        # Determine operation name
                        operation_name = None
                        if op_code == 'C25394' or ('api' in display_name and 'manufacture' in display_name):
                            operation_name = 'API Manufacture'
                        elif op_code in operation_codes:
                            operation_name = operation_codes[op_code]
                        
                        if operation_name and operation_name not in operations:
                            operations.append(operation_name)
            
            # Add establishment if it has operations and is in database
            if operations:
                if self._check_database_match(establishment_id, 'FEI_NUMBER') or self._check_database_match(establishment_id, 'DUNS_NUMBER'):
                    establishments[establishment_id] = {
                        'id': establishment_id,
                        'name': establishment_name,
                        'operations': operations
                    }
        
        return list(establishments.values())


    def _find_matches_with_regex_filtered(self, content: str, spl_id: str) -> List[FEIMatch]:
        """Fallback regex-based matching that excludes author sections"""
        matches = []
        
        try:
            # Find assignedOrganization sections that are NOT in author sections
            # Look for assignedOrganization that contains ID elements
            org_pattern = r'<assignedOrganization[^>]*>.*?</assignedOrganization>'
            org_matches = re.finditer(org_pattern, content, re.DOTALL | re.IGNORECASE)
            
            for org_match in org_matches:
                org_content = org_match.group(0)
                
                # Skip if this organization is within an author section
                # Check 500 characters before this match for <author>
                context_start = max(0, org_match.start() - 500)
                context_before = content[context_start:org_match.start()]
                
                # Count author tags vs closing author tags to see if we're inside one
                author_opens = context_before.count('<author')
                author_closes = context_before.count('</author>')
                
                if author_opens > author_closes:
                    continue  # We're inside an author section, skip
                
                # Find IDs within this organization
                id_pattern = r'<id\\s+([^>]*extension="(\\d{7,15})"[^>]*)>'
                id_matches = re.finditer(id_pattern, org_content, re.IGNORECASE)
                
                for id_match in id_matches:
                    extension = id_match.group(2)
                    clean_extension = re.sub(r'[^\\d]', '', extension)
                    
                    # Calculate line number for location
                    line_num = content[:org_match.start() + id_match.start()].count('\\n') + 1
                    xml_location = f"Line {line_num} (regex-based)"
                    
                    # Get surrounding context
                    start_context = max(0, id_match.start() - 100)
                    end_context = min(len(org_content), id_match.end() + 100)
                    xml_context = org_content[start_context:end_context].replace('\\n', ' ').strip()
                    
                    # Check databases
                    fei_variants = self._generate_all_id_variants(extension)
                    duns_variants = self._generate_all_id_variants(extension)
                    
                    # Try FEI first
                    for fei_key in fei_variants:
                        if fei_key in self.fei_database:
                            fei_match = FEIMatch(
                                fei_number=clean_extension,
                                xml_location=xml_location,
                                match_type='FEI_NUMBER',
                                establishment_name=self._extract_name_from_context_regex(xml_context),
                                xml_context=xml_context[:200] + "..." if len(xml_context) > 200 else xml_context
                            )
                            matches.append(fei_match)
                            break
                    else:
                        # Try DUNS if no FEI match
                        for duns_key in duns_variants:
                            if duns_key in self.duns_database:
                                duns_match = FEIMatch(
                                    fei_number=clean_extension,
                                    xml_location=xml_location,
                                    match_type='DUNS_NUMBER',
                                    establishment_name=self._extract_name_from_context_regex(xml_context),
                                    xml_context=xml_context[:200] + "..." if len(xml_context) > 200 else xml_context
                                )
                                matches.append(duns_match)
                                break
                        
        except Exception as e:
            pass
            
        return matches

    def _extract_name_from_context_regex(self, context: str) -> str:
        """Extract establishment name from context using regex"""
        try:
            # Look for name tags
            name_match = re.search(r'<name[^>]*>([^<]+)</name>', context, re.IGNORECASE)
            if name_match:
                return name_match.group(1).strip()
            return "Unknown"
        except Exception as e:
            return "Unknown"

    # ADD THE NEW METHOD HERE:
    def _check_database_match(self, extension: str, match_type: str) -> bool:
        """Check if an extension exists in FEI or DUNS database"""
        variants = self._generate_all_id_variants(extension)
        
        if match_type == 'FEI_NUMBER':
            return any(variant in self.fei_database for variant in variants)
        elif match_type == 'DUNS_NUMBER':
            return any(variant in self.duns_database for variant in variants)
        
        return False

    def _find_matches_flexible_xml(self, root, processed_ids: set) -> List[FEIMatch]:
        """Flexible XML parsing that works with any SPL structure"""
        matches = []
        
        # Find ALL id elements in the document
        for id_elem in root.iter():
            if not id_elem.tag.endswith('id') or not id_elem.get('extension'):
                continue
                
            extension = id_elem.get('extension')
            clean_extension = re.sub(r'[^\d]', '', extension)
            
            # Skip if already processed or too short
            if clean_extension in processed_ids or len(clean_extension) < 7:
                continue
            
            # Determine context - is this an establishment or labeler?
            context_type = self._determine_context_type(id_elem)
            
            # Skip main labelers, keep establishments
            if context_type == 'MAIN_LABELER':
                continue
            elif context_type in ['ESTABLISHMENT', 'UNKNOWN']:
                # Check if this ID exists in our databases
                match = self._create_match_from_xml_element(id_elem, extension, clean_extension, root)
                if match:
                    matches.append(match)
                    processed_ids.add(clean_extension)
        
        return matches

    def _find_matches_flexible_regex(self, content: str, processed_ids: set) -> List[FEIMatch]:
        """Flexible regex-based matching for any SPL structure"""
        matches = []
        
        # Find all ID elements that might be establishments
        id_pattern = r'<id\s+([^>]*extension="(\d{7,15})"[^>]*)>'
        id_matches = re.finditer(id_pattern, content, re.IGNORECASE)
        
        for id_match in id_matches:
            extension = id_match.group(2)
            clean_extension = re.sub(r'[^\d]', '', extension)
            
            # Skip if already processed
            if clean_extension in processed_ids:
                continue
            
            # Get context around this ID to determine if it's an establishment
            context_start = max(0, id_match.start() - 500)
            context_end = min(len(content), id_match.end() + 500)
            context = content[context_start:context_end]
            
            # Skip if this looks like a main labeler
            if self._is_main_labeler_context(context):
                continue
            
            # Try to find establishment name
            establishment_name = self._extract_name_from_context_regex(context)
            
            # Check if this ID exists in our databases
            match = self._create_match_from_regex_flexible(extension, clean_extension, id_match, establishment_name, context)
            if match:
                matches.append(match)
                processed_ids.add(clean_extension)
        
        return matches

    def _determine_context_type(self, id_elem) -> str:
        """Determine if an ID element represents a labeler or establishment"""
        # Walk up the XML tree to understand context
        current = id_elem
        path = []
        
        for _ in range(10):  # Limit traversal depth
            current = current.getparent() if hasattr(current, 'getparent') else None
            if current is None:
                break
            
            tag = current.tag.split('}')[-1] if '}' in current.tag else current.tag
            path.append(tag)
            
            # Main labeler patterns
            if tag == 'representedOrganization' and 'author' in path:
                return 'MAIN_LABELER'
            
            # Establishment patterns
            if tag == 'assignedOrganization':
                # Check if this is nested under author (could be establishment)
                if 'author' in path:
                    return 'ESTABLISHMENT'
                # Or if it's in other contexts
                return 'ESTABLISHMENT'
        
        return 'UNKNOWN'

    def _is_main_labeler_context(self, context: str) -> bool:
        """Check if context suggests this is a main labeler rather than establishment"""
        # Look for representedOrganization pattern (main labeler)
        if '<representedOrganization' in context and '<author' in context:
            # Check if there's no assignedOrganization between them
            repr_pos = context.find('<representedOrganization')
            author_pos = context.find('<author')
            if author_pos < repr_pos:
                between_text = context[author_pos:repr_pos]
                if '<assignedOrganization' not in between_text:
                    return True
        return False

    def _create_match_from_xml_element(self, id_elem, extension: str, clean_extension: str, root) -> Optional[FEIMatch]:
        """Create match with database lookup"""
        # Try FEI database first
        fei_variants = self._generate_all_id_variants(extension)
        for fei_key in fei_variants:
            if fei_key in self.fei_database:
                return FEIMatch(
                    fei_number=clean_extension,
                    xml_location=self._get_element_xpath(id_elem, root),
                    match_type='FEI_NUMBER',
                    establishment_name=self._extract_establishment_name_from_context(id_elem),
                    xml_context=self._get_element_context(id_elem, root)
                )
        
        # Try DUNS database
        duns_variants = self._generate_all_id_variants(extension)
        for duns_key in duns_variants:
            if duns_key in self.duns_database:
                return FEIMatch(
                    fei_number=clean_extension,
                    xml_location=self._get_element_xpath(id_elem, root),
                    match_type='DUNS_NUMBER',
                    establishment_name=self._extract_establishment_name_from_context(id_elem),
                    xml_context=self._get_element_context(id_elem, root)
                )
        
        return None

    def _create_match_from_regex_flexible(self, extension: str, clean_extension: str, id_match, establishment_name: str, context: str) -> Optional[FEIMatch]:
        """Create match from regex parsing with database lookup"""
        # Calculate line number for location
        line_num = id_match.string[:id_match.start()].count('\n') + 1
        xml_location = f"Line {line_num} (regex)"
        
        # Try FEI database first
        fei_variants = self._generate_all_id_variants(extension)
        for fei_key in fei_variants:
            if fei_key in self.fei_database:
                return FEIMatch(
                    fei_number=clean_extension,
                    xml_location=xml_location,
                    match_type='FEI_NUMBER',
                    establishment_name=establishment_name,
                    xml_context=context[:200] + "..." if len(context) > 200 else context
                )
        
        # Try DUNS database
        duns_variants = self._generate_all_id_variants(extension)
        for duns_key in duns_variants:
            if duns_key in self.duns_database:
                return FEIMatch(
                    fei_number=clean_extension,
                    xml_location=xml_location,
                    match_type='DUNS_NUMBER',
                    establishment_name=establishment_name,
                    xml_context=context[:200] + "..." if len(context) > 200 else context
                )
        
        return None

    def _extract_operations_flexible(self, content: str, establishment_id: str, target_ndc: str, establishment_info: Dict) -> Tuple[List[str], List[str]]:
        """Flexible operation extraction that works with various XML structures"""
        operations = []
        quotes = []
        
        # Generate NDC variants for matching
        ndc_variants = self.normalize_ndc_for_matching(target_ndc)
        
        # Operation code mappings
        operation_codes = {
            'C43360': 'Manufacture', 'C82401': 'Manufacture', 'C25391': 'Analysis',
            'C84731': 'Pack', 'C84732': 'Label', 'C48482': 'Repack',
            'C73606': 'Relabel', 'C25392': 'Sterilize', 'C25394': 'API Manufacture',
            'C43359': 'Manufacture'
        }
        
        # Strategy 1: Find operations near this establishment ID
        # Look for performance/businessOperation elements that are "close" to our establishment ID
        
        # Split content around our establishment ID
        id_pattern = rf'<id[^>]*extension="[^"]*{re.escape(establishment_id)}"[^>]*>'
        id_matches = list(re.finditer(id_pattern, content, re.IGNORECASE))
        
        for id_match in id_matches:
            # Look for operations within reasonable distance (e.g., next 5000 characters)
            search_start = id_match.end()
            search_end = min(len(content), search_start + 5000)
            search_section = content[search_start:search_end]
            
            # Find performance elements
            perf_elements = re.findall(r'<performance[^>]*>.*?</performance>', search_section, re.DOTALL | re.IGNORECASE)
            perf_elements.extend(re.findall(r'<businessOperation[^>]*>.*?</businessOperation>', search_section, re.DOTALL | re.IGNORECASE))
            
            for perf_elem in perf_elements:
                # Extract operation
                operation_found = None
                code_match = re.search(r'<code[^>]*code="([^"]*)"[^>]*(?:displayName="([^"]*)")?', perf_elem, re.IGNORECASE)
                
                if code_match:
                    operation_code = code_match.group(1)
                    display_name = code_match.group(2) or ""
                    
                    if operation_code == 'C25394' or 'api' in display_name.lower():
                        operation_found = 'API Manufacture'
                    elif operation_code in operation_codes:
                        operation_found = operation_codes[operation_code]
                
                if operation_found:
                    # Check for NDC match
                    ndc_matches = re.findall(r'<code[^>]*code="([^"]*)"[^>]*codeSystem="2\.16\.840\.1\.113883\.6\.69"', perf_elem, re.IGNORECASE)
                    
                    ndc_found = False
                    for ndc_code in ndc_matches:
                        ndc_variants_found = self.normalize_ndc_for_matching(ndc_code.strip())
                        if any(v in ndc_variants for v in ndc_variants_found):
                            ndc_found = True
                            break
                    
                    if ndc_found and operation_found not in operations:
                        operations.append(operation_found)
                        quotes.append(f'Found {operation_found} operation for National Drug Code {target_ndc}')
        
        # Strategy 2: If no NDC-specific operations found, look for general operations
        if not operations:
            # Look for any operations associated with this establishment (without NDC requirement)
            for id_match in id_matches:
                search_start = id_match.end()
                search_end = min(len(content), search_start + 3000)
                search_section = content[search_start:search_end]
                
                # Look for any operation codes
                all_operations = re.findall(r'<code[^>]*code="([^"]*)"[^>]*displayName="([^"]*)"', search_section, re.IGNORECASE)
                
                for op_code, display_name in all_operations:
                    if op_code in operation_codes and operation_codes[op_code] not in operations:
                        operations.append(operation_codes[op_code])
                        quotes.append(f'Found {operation_codes[op_code]} operation (general) at this establishment')
        
        return operations, quotes

    def extract_ndc_specific_operations(self, section: str, target_ndc: str, establishment_name: str) -> Tuple[List[str], List[str]]:
        """Extract operations that are specific to the target NDC from an establishment section"""
        operations = []
        quotes = []
        api_manufacture_codes = set()  # Track which codes gave us API Manufacture
        manufacture_codes = set()      # Track which codes gave us generic Manufacture

        # Generate all possible NDC variants for matching
        ndc_variants = self.normalize_ndc_for_matching(target_ndc)

        # Updated operation mappings
        operation_codes = {
            'C43360': 'Manufacture',
            'C82401': 'Manufacture', 
            'C25391': 'Analysis',
            'C84731': 'Pack',
            'C84732': 'Label',
            'C48482': 'Repack',
            'C73606': 'Relabel',
            'C25392': 'Sterilize',
            'C25394': 'API Manufacture',
            'C43359': 'Manufacture'
        }

        # Look for performance elements with actDefinition (this is the correct structure for SPL)
        # FIXED - Only get direct performance elements, not nested ones:
        performance_elements = []
        # Split the section to find performance elements that belong directly to this establishment
        # Look for performance elements that come after the establishment ID but before any nested assignedEntity
        parts = re.split(r'<assignedEntity>', section)
        if len(parts) > 1:
            # Take the first part after the main establishment (before any nested establishments)
            main_establishment_section = parts[1].split('</assignedEntity>')[0] if '</assignedEntity>' in parts[1] else parts[1]
            performance_elements = re.findall(r'<performance[^>]*>.*?</performance>', main_establishment_section, re.DOTALL | re.IGNORECASE)
        else:
            # Fallback to original method if structure is different
            performance_elements = re.findall(r'<performance[^>]*>.*?</performance>', section, re.DOTALL | re.IGNORECASE)

        # ADD THIS NEW DEBUG CODE HERE:
        if "080129000" in establishment_name or "Genentech" in establishment_name:
            st.write(f"\n=== XML SECTION BEING PROCESSED ===")
            st.write(f"Establishment: {establishment_name}")
            st.write(f"Section content (first 1000 chars): {section[:1000]}")
            st.write(f"=== END XML SECTION ===")

        # ADD DEBUG CODE HERE:
        if "080129000" in establishment_name or "Genentech" in establishment_name:
            st.write(f"\n=== DEBUG: {establishment_name} ===")
            st.write(f"Target NDC: {target_ndc}")
            st.write(f"Performance elements found: {len(performance_elements)}")

        for perf_elem in performance_elements:
            # Extract operation code and displayName from actDefinition
            operation_found = None
            operation_code_match = re.search(r'<code[^>]*code="([^"]*)"[^>]*displayName="([^"]*)"', perf_elem, re.IGNORECASE)
            
            if operation_code_match:
                operation_code = operation_code_match.group(1)
                display_name = operation_code_match.group(2).lower()
                
                # ADD MORE DEBUG CODE HERE:
                if "080129000" in establishment_name or "Genentech" in establishment_name:
                    st.write(f"Operation: {operation_code} = {display_name}")
                
                # Check for API Manufacture first (more specific)
                if operation_code == 'C25394' or 'api' in display_name:
                    operation_found = 'API Manufacture'
                    api_manufacture_codes.add(operation_code)
                elif operation_code in operation_codes:
                    operation_found = operation_codes[operation_code]
                    if operation_found == 'Manufacture':
                        manufacture_codes.add(operation_code)

            if operation_found:
                # Look for NDC codes in manufacturedMaterialKind
                ndc_code_pattern = r'<code[^>]*code="([^"]*)"[^>]*codeSystem="2\.16\.840\.1\.113883\.6\.69"'
                ndc_matches = re.findall(ndc_code_pattern, perf_elem, re.IGNORECASE)
                
                # ADD DEBUG FOR NDC MATCHES:
                if "080129000" in establishment_name or "Genentech" in establishment_name:
                    st.write(f"NDCs found in this operation: {ndc_matches}")
                
                ndc_found_in_operation = False
                for ndc_code in ndc_matches:
                    # Clean up the NDC code
                    clean_ndc = ndc_code.strip()
                    
                    # Generate variants for this NDC code
                    potential_variants = self.normalize_ndc_for_matching(clean_ndc)

                    # Check if any variant matches our target NDC
                    matching_variants = [v for v in potential_variants if v in ndc_variants]
                    if matching_variants:
                        ndc_found_in_operation = True
                        break

                # If our target NDC was found in this operation, add it
                if ndc_found_in_operation and operation_found not in operations:
                    # ADD DEBUG FOR ADDING OPERATIONS:
                    if "080129000" in establishment_name or "Genentech" in establishment_name:
                        st.write(f"ADDING OPERATION: {operation_found}")
                    operations.append(operation_found)
                    quotes.append(f'"Found {operation_found} operation for National Drug Code {target_ndc} in {establishment_name}"')

        # SMART FILTERING: Only remove "Manufacture" if we found API Manufacture 
        # AND the Manufacture came from a code that could be misinterpreted
        if 'API Manufacture' in operations and 'Manufacture' in operations:
            # Only remove if they seem to be the same operation detected differently
            # Keep both if they're truly separate operations
            if len(api_manufacture_codes) > 0 and len(manufacture_codes) > 0:
                # Check if any manufacture codes overlap with what should be API codes
                overlapping = api_manufacture_codes.intersection(manufacture_codes)
                if overlapping or len(operations) == 2:  # If only these two operations, likely duplicate
                    operations.remove('Manufacture')
                    quotes = [q for q in quotes if 'Manufacture operation' not in q or 'API Manufacture operation' in q]

        return operations, quotes

    def extract_general_operations(self, section: str, establishment_name: str) -> Tuple[List[str], List[str]]:
        """Extract general operations from an establishment section (not NDC-specific)"""
        operations = []
        quotes = []

        # Updated operation mappings
        operation_codes = {
            'C43360': 'Manufacture',
            'C82401': 'Manufacture', 
            'C25391': 'Analysis',
            'C84731': 'Pack',
            'C25392': 'Sterilize',
            'C48482': 'Repack',
            'C73606': 'Relabel',
            'C84732': 'Label',
            'C25394': 'API Manufacture',
            'C43359': 'Manufacture'
        }

        operation_names = {
            'manufacture': 'Manufacture',
            'api manufacture': 'API Manufacture',
            'analysis': 'Analysis',
            'label': 'Label',
            'pack': 'Pack',
            'repack': 'Repack',
            'relabel': 'Relabel',
            'sterilize': 'Sterilize'
        }

        # Look for business operations
        business_operations = re.findall(r'<businessOperation[^>]*>.*?</businessOperation>', section, re.DOTALL | re.IGNORECASE)

        for bus_op in business_operations:
            operation_found = None

            # Check for displayName attributes
            display_name_match = re.search(r'displayName="([^"]*)"', bus_op, re.IGNORECASE)
            if display_name_match:
                display_name = display_name_match.group(1).lower()
                if 'api' in display_name and 'manufacture' in display_name:
                    operation_found = 'API Manufacture'
                else:
                    for name, operation in operation_names.items():
                        if name in display_name and operation != 'API Manufacture':
                            operation_found = operation
                            break

            # Check for operation codes
            if not operation_found:
                for code, operation in operation_codes.items():
                    if code in bus_op:
                        operation_found = operation
                        break

            if operation_found and operation_found not in operations:
                operations.append(operation_found)
                quotes.append(f'Found {operation_found} operation in {establishment_name}')

        # SMART FILTERING: Same logic as above
        if 'API Manufacture' in operations and 'Manufacture' in operations:
            # Only remove if they seem to be the same operation, not separate ones
            if len([op for op in operations if 'API' in op]) == 1 and len([op for op in operations if op == 'Manufacture']) == 1:
                operations.remove('Manufacture')
                quotes = [q for q in quotes if 'Manufacture operation' not in q or 'API Manufacture operation' in q]

        return operations, quotes

    def get_establishment_info(self, product_info: ProductInfo) -> List[Dict]:
        """Get establishment information for a product with inspection data"""
        establishments = []

        if product_info.spl_id:
            _, _, establishments_info = self.extract_establishments_with_fei(product_info.spl_id, product_info.ndc)
            
            if establishments_info:
                for establishment in establishments_info:
                    # Get FEI from the establishment data
                    fei_number = establishment.get('fei_number') or establishment.get('original_fei')
                    
                    # Clean up FEI number if it exists
                    if fei_number and str(fei_number).strip() not in ['nan', '', 'None', '0000000', '0000000000']:
                        fei_clean = str(fei_number).strip()
                        
                        # Look up inspections using the FEI number
                        inspections = self.get_facility_inspections(fei_clean)
                        inspection_summary = self.get_inspection_summary(inspections)
                        
                        establishment['inspections'] = inspections[:10]
                        establishment['inspection_summary'] = inspection_summary
                        establishment['fei_number'] = fei_clean
                    else:
                        establishment['inspections'] = []
                        establishment['inspection_summary'] = {
                            'total_records': 0,
                            'status': 'No FEI number available for inspection lookup'
                        }
                    
                    establishments.append(establishment)
        
        return establishments

    def extract_establishments_with_fei(self, spl_id: str, target_ndc: str) -> Tuple[List[str], List[str], List[Dict]]:
        """Extract operations, quotes, and detailed establishment info - NDC-SPECIFIC VERSION"""
        try:
            spl_url = f"{self.dailymed_base_url}/services/v2/spls/{spl_id}.xml"
            response = self.session.get(spl_url)

            if response.status_code != 200:
                return [], [], []

            content = response.text
            establishments_info = []
            
            # Use the NDC-specific approach (method you already have)
            ndc_establishments = self._extract_establishments_ndc_specific(content, target_ndc)
            
            st.write(f"**Found {len(ndc_establishments)} establishments with operations for NDC {target_ndc}**")
            
            # Convert to the expected format
            for est in ndc_establishments:
                st.write(f"- {est['name']} ({est['id']}): {', '.join(est['operations'])}")
                
                # Look up full establishment info from database
                if self._check_database_match(est['id'], 'FEI_NUMBER'):
                    establishment_info = self.lookup_fei_establishment(est['id'])
                    match_type = 'FEI_NUMBER'
                else:
                    establishment_info = self.lookup_duns_establishment(est['id'])
                    match_type = 'DUNS_NUMBER'
                
                if establishment_info:
                    establishment_info['operations'] = est['operations']
                    establishment_info['quotes'] = [f'Found {op} operation for National Drug Code {target_ndc}' for op in est['operations']]
                    establishment_info['match_type'] = match_type
                    establishment_info['xml_location'] = f"ID: {est['id']}"
                    establishment_info['xml_context'] = f"Establishment: {est['name']}"
                    
                    establishments_info.append(establishment_info)

            return [], [], establishments_info

        except Exception as e:
            st.write(f"Error in extract_establishments_with_fei: {e}")
            return [], [], []

    def get_facility_inspections(self, fei_number: str) -> List[Dict]:
        """Get inspection history - prioritize local database, fallback to API"""
        inspections = []
        
        try:
            # First try local inspection database
            local_inspections = self.get_facility_inspections_from_database(fei_number)
            if local_inspections:
                inspections.extend(local_inspections)
            
            # Also get enforcement records from API as supplementary data
            enforcement_inspections = []
            try:
                enforcement_sources = [
                    self.get_drug_inspections(fei_number),
                    self.get_device_inspections(fei_number),
                    self.get_food_inspections(fei_number),
                    self.get_warning_letters(fei_number)
                ]
                
                for source_inspections in enforcement_sources:
                    if source_inspections:
                        enforcement_inspections.extend(source_inspections)
            except:
                pass
            
            # Add enforcement records
            inspections.extend(enforcement_inspections)
            
            # Remove duplicates and sort
            unique_inspections = self.deduplicate_inspections(inspections)
            return sorted(unique_inspections, key=lambda x: x.get('inspection_date', ''), reverse=True)
            
        except Exception as e:
            return []

    def extract_company_names(self, product_info: ProductInfo) -> List[str]:
        """Extract company names from product information"""
        company_names = []

        # Extract from product name (text in brackets)
        bracket_matches = re.findall(r'\[([^\]]+)\]', product_info.product_name)
        for match in bracket_matches:
            clean_match = re.sub(r'\s+(INC|LLC|CORP|LTD|CO\.?|COMPANY)\.?$', '', match, flags=re.IGNORECASE)
            if len(clean_match) > 3:
                company_names.append(clean_match.strip())

        # Add labeler name
        if product_info.labeler_name and product_info.labeler_name != 'Unknown':
            company_names.append(product_info.labeler_name)

        return company_names

    def create_establishments_from_spl(self, company_names: List[str], product_info: ProductInfo) -> List[Dict]:
        """Create multiple establishments based on SPL data with NDC-specific operations"""
        establishments = []

        if not product_info or not product_info.spl_id:
            return establishments

        # Get operations and establishment info from SPL for the specific NDC
        _, _, establishments_info = self.extract_establishments_with_fei(product_info.spl_id, product_info.ndc)

        if establishments_info:
            # Use the establishments found in SPL with their specific operations
            establishments = establishments_info

        return establishments

    def extract_labeler_from_spl(self, spl_id: str) -> Tuple[str, str]:
        """Extract labeler name and DUNS from SPL"""
        try:
            spl_url = f"{self.dailymed_base_url}/services/v2/spls/{spl_id}.xml"
            response = self.session.get(spl_url)

            if response.status_code != 200:
                return "Unknown", None

            content = response.text
            
            # Parse XML to find labeler information
            try:
                root = ET.fromstring(content)
                
                # Look for author section which typically contains labeler information
                for elem in root.iter():
                    if 'author' in elem.tag.lower():
                        labeler_name = None
                        labeler_duns = None
                        
                        # Look for representedOrganization within author
                        for child in elem.iter():
                            if 'representedOrganization' in child.tag.lower() or 'organization' in child.tag.lower():
                                # Look for name
                                for name_elem in child.iter():
                                    if name_elem.tag.endswith('name') and name_elem.text:
                                        labeler_name = name_elem.text.strip()
                                        break
                                
                                # Look for ID (DUNS)
                                for id_elem in child.iter():
                                    if id_elem.tag.endswith('id') and id_elem.get('extension'):
                                        extension = id_elem.get('extension')
                                        clean_extension = re.sub(r'[^\d]', '', extension)
                                        if len(clean_extension) >= 8:  # Looks like DUNS
                                            labeler_duns = clean_extension
                                            break
                                
                                if labeler_name:
                                    return labeler_name, labeler_duns
                
                # Fallback: look for any organization name in the document
                org_name_pattern = r'<name[^>]*>([^<]+(?:Inc|LLC|Corp|Company|Ltd)[^<]*)</name>'
                name_matches = re.findall(org_name_pattern, content, re.IGNORECASE)
                if name_matches:
                    return name_matches[0].strip(), None
                    
            except ET.XMLSyntaxError:
                # Fallback to regex-based approach
                # Look for labeler name in author sections
                author_pattern = r'<author[^>]*>.*?<representedOrganization[^>]*>.*?<name[^>]*>([^<]+)</name>.*?</representedOrganization>.*?</author>'
                author_matches = re.findall(author_pattern, content, re.DOTALL | re.IGNORECASE)
                if author_matches:
                    return author_matches[0].strip(), None
                
                # Look for any organization name
                org_pattern = r'<name[^>]*>([^<]+(?:Inc|LLC|Corp|Company|Ltd)[^<]*)</name>'
                org_matches = re.findall(org_pattern, content, re.IGNORECASE)
                if org_matches:
                    return org_matches[0].strip(), None
            
            return "Unknown", None
                
        except Exception as e:
            return "Unknown", None

    def find_labeler_info_in_spl(self, spl_id: str, labeler_name: str) -> Optional[Dict]:
        """Find labeler information from SPL with enhanced fallback"""
        try:
            # First, extract the actual labeler name and DUNS from SPL
            actual_labeler_name, labeler_duns = self.extract_labeler_from_spl(spl_id)
            
            # Use the extracted name if available, otherwise use the provided name
            if actual_labeler_name != "Unknown":
                labeler_name = actual_labeler_name
            
            # Try to find DUNS information if we have it
            if labeler_duns:
                duns_info = self.lookup_duns_establishment(labeler_duns)
                if duns_info:
                    return {
                        'establishment_name': duns_info.get('establishment_name', labeler_name),
                        'firm_name': duns_info.get('firm_name', labeler_name),
                        'address_line_1': duns_info.get('address_line_1', 'Unknown'),
                        'city': duns_info.get('city', 'Unknown'),
                        'state_province': duns_info.get('state_province', 'Unknown'),
                        'country': duns_info.get('country', 'Unknown'),
                        'postal_code': duns_info.get('postal_code', ''),
                        'latitude': duns_info.get('latitude'),
                        'longitude': duns_info.get('longitude'),
                        'search_method': 'labeler_duns_database',
                        'duns_number': labeler_duns,
                        'match_type': 'LABELER'
                    }
            
            # REMOVED: Don't return labeler-only info as an establishment
            return None
                
        except Exception as e:
            return None

    def process_single_ndc(self, ndc: str) -> pd.DataFrame:
        """Process a single NDC number"""
        if not self.validate_ndc_format(ndc):
            return pd.DataFrame()

        normalized_ndc = self.normalize_ndc(ndc)

        product_info = self.get_ndc_info_comprehensive(normalized_ndc)
        if not product_info:
            return pd.DataFrame()

        establishments = self.get_establishment_info(product_info)

        results = []
        if establishments:
            for establishment in establishments:
                results.append({
                    'ndc': ndc,
                    'product_name': product_info.product_name,
                    'labeler_name': product_info.labeler_name,
                    'spl_id': product_info.spl_id,
                    'fei_number': establishment.get('fei_number'),
                    'duns_number': establishment.get('duns_number'),
                    'establishment_name': establishment.get('establishment_name'),
                    'firm_name': establishment.get('firm_name'),
                    'address_line_1': establishment.get('address_line_1'),
                    'city': establishment.get('city'),
                    'state': establishment.get('state_province'),
                    'country': establishment.get('country'),
                    'postal_code': establishment.get('postal_code', ''),
                    'latitude': establishment.get('latitude'),
                    'longitude': establishment.get('longitude'),
                    'spl_operations': ', '.join(establishment.get('operations', [])) if establishment.get('operations') else 'None found for this National Drug Code',
                    'spl_quotes': ' | '.join(establishment.get('quotes', [])),
                    'search_method': establishment.get('search_method'),
                    'xml_location': establishment.get('xml_location', 'Unknown'),
                    'match_type': establishment.get('match_type', 'Unknown'),
                    'xml_context': establishment.get('xml_context', '')
                })
        else:
            # FIXED: Show that no manufacturing establishments were identified
            results.append({
                'ndc': ndc,
                'product_name': product_info.product_name,
                'labeler_name': product_info.labeler_name,
                'spl_id': product_info.spl_id,
                'fei_number': None,
                'duns_number': None,
                'establishment_name': None,
                'firm_name': None,
                'address_line_1': None,
                'city': None,
                'state': None,
                'country': None,
                'postal_code': '',
                'latitude': None,
                'longitude': None,
                'spl_operations': None,
                'spl_quotes': None,
                'search_method': 'no_establishments_found',
                'xml_location': None,
                'match_type': None,
                'xml_context': ''
            })

        return pd.DataFrame(results)

    def get_facility_inspections_from_database(self, fei_number: str) -> List[Dict]:
        """Get inspection outcomes from local database"""
        inspections = []
        
        try:
            # Generate FEI variants for lookup
            fei_variants = self._generate_all_id_variants(fei_number)
            
            for variant in fei_variants:
                if variant in self.inspection_database:
                    for record in self.inspection_database[variant]:
                        inspection = {
                            'inspection_type': 'FDA_INSPECTION',
                            'inspection_date': record.get('inspection_end_date', ''),  # FIX: Map to inspection_date
                            'inspection_id': record.get('inspection_id', ''),
                            'classification': record.get('classification', 'Unknown'),
                            'status': record.get('posted_citations', 'Unknown'),
                            'fiscal_year': record.get('fiscal_year', ''),
                            'project_area': record.get('project_area', ''),
                            'product_type': record.get('product_type', ''),
                            'firm_name': record.get('legal_name', ''),
                            'city': record.get('city', ''),
                            'state': record.get('state', ''),
                            'country': record.get('country', ''),
                            'additional_details': record.get('additional_details', ''),
                            'fmd_145_date': record.get('fmd_145_date', ''),
                            'source': 'FDA Inspection Database'
                        }
                        inspections.append(inspection)
                    break  # Found records for this FEI
            
            # Sort by inspection date (most recent first)
            inspections.sort(key=lambda x: x.get('inspection_date', ''), reverse=True)
            return inspections
            
        except Exception as e:
            return []

    def get_drug_inspections(self, fei_number: str) -> List[Dict]:
        """Get drug facility inspections from FDA API"""
        try:
            url = f"{self.base_openfda_url}/drug/enforcement.json"
            
            search_queries = [
                f'fei_number:"{fei_number}"',
                f'firm_fei_number:"{fei_number}"',
                f'registration_number:"{fei_number}"'
            ]
            
            inspections = []
            for query in search_queries:
                try:
                    params = {
                        'search': query,
                        'limit': 100
                    }
                    response = self.session.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('results'):
                            for result in data['results']:
                                inspection = self.parse_enforcement_record(result, 'drug')
                                if inspection:
                                    inspections.append(inspection)
                                    
                except Exception as e:
                    continue
                    
            return inspections
            
        except Exception as e:
            return []

    def get_device_inspections(self, fei_number: str) -> List[Dict]:
        """Get device facility inspections from FDA API"""
        try:
            url = f"{self.base_openfda_url}/device/enforcement.json"
            
            search_queries = [
                f'fei_number:"{fei_number}"',
                f'firm_fei_number:"{fei_number}"'
            ]
            
            inspections = []
            for query in search_queries:
                try:
                    params = {
                        'search': query,
                        'limit': 100
                    }
                    response = self.session.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('results'):
                            for result in data['results']:
                                inspection = self.parse_enforcement_record(result, 'device')
                                if inspection:
                                    inspections.append(inspection)
                                    
                except Exception as e:
                    continue
                    
            return inspections
            
        except Exception as e:
            return []

    def get_food_inspections(self, fei_number: str) -> List[Dict]:
        """Get food facility inspections"""
        try:
            url = f"{self.base_openfda_url}/food/enforcement.json"
            
            params = {
                'search': f'fei_number:"{fei_number}"',
                'limit': 50
            }
            response = self.session.get(url, params=params, timeout=10)
            
            inspections = []
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    for result in data['results']:
                        inspection = self.parse_enforcement_record(result, 'food')
                        if inspection:
                            inspections.append(inspection)
                            
            return inspections
            
        except Exception as e:
            return []

    def get_warning_letters(self, fei_number: str) -> List[Dict]:
        """Get warning letters for facility"""
        try:
            url = f"{self.base_openfda_url}/other/enforcement.json"
            
            params = {
                'search': f'fei_number:"{fei_number}"',
                'limit': 50
            }
            response = self.session.get(url, params=params, timeout=10)
            
            warnings = []
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    for result in data['results']:
                        warning = self.parse_warning_letter(result)
                        if warning:
                            warnings.append(warning)
                            
            return warnings
            
        except Exception as e:
            return []

    def parse_enforcement_record(self, record: Dict, record_type: str) -> Optional[Dict]:
        """Parse FDA enforcement record"""
        try:
            inspection = {
                'inspection_type': record_type.upper(),
                'inspection_date': record.get('report_date', record.get('recall_initiation_date', '')),
                'classification': record.get('classification', 'Unknown'),
                'status': record.get('status', 'Unknown'),
                'reason_for_recall': record.get('reason_for_recall', ''),
                'product_description': record.get('product_description', ''),
                'firm_name': record.get('recalling_firm', ''),
                'city': record.get('city', ''),
                'state': record.get('state', ''),
                'country': record.get('country', ''),
                'voluntary_mandated': record.get('voluntary_mandated', ''),
                'initial_firm_notification': record.get('initial_firm_notification', ''),
                'event_id': record.get('event_id', ''),
                'more_code_info': record.get('more_code_info', ''),
                'recall_number': record.get('recall_number', ''),
                'source': 'FDA Enforcement API'
            }
            
            return inspection
            
        except Exception as e:
            return None

    def parse_warning_letter(self, record: Dict) -> Optional[Dict]:
        """Parse warning letter"""
        try:
            return {
                'inspection_type': 'WARNING_LETTER',
                'inspection_date': record.get('report_date', ''),
                'classification': 'Warning Letter Issued',
                'status': 'Warning Letter',
                'reason_for_recall': record.get('reason_for_recall', ''),
                'product_description': record.get('product_description', ''),
                'firm_name': record.get('recalling_firm', ''),
                'city': record.get('city', ''),
                'state': record.get('state', ''),
                'country': record.get('country', ''),
                'source': 'FDA Warning Letters'
            }
        except Exception as e:
            return None

    def deduplicate_inspections(self, inspections: List[Dict]) -> List[Dict]:
        """Remove duplicate inspection records"""
        seen = set()
        unique_inspections = []
        
        for inspection in inspections:
            key = (
                inspection.get('inspection_date', ''),
                inspection.get('inspection_type', ''),
                inspection.get('firm_name', ''),
                inspection.get('recall_number', '')
            )
            
            if key not in seen:
                seen.add(key)
                unique_inspections.append(inspection)
        
        return unique_inspections

    def format_inspection_details(self, inspections: List[Dict]) -> List[Dict]:
        """Format inspection details for display"""
        formatted = []
        
        for inspection in inspections:
            formatted_inspection = {
                'date': inspection.get('inspection_date', 'Date not specified'),
                'type': inspection.get('inspection_type', 'Type not specified'),
                'classification': inspection.get('classification', 'Classification not specified'),
                'action_description': inspection.get('reason_for_recall', 'Description not available'),
                'product_affected': inspection.get('product_description', 'Product not specified'),
                'recall_number': inspection.get('recall_number', ''),
                'voluntary_or_mandated': inspection.get('voluntary_mandated', 'Not specified'),
                'firm_notification_method': inspection.get('initial_firm_notification', 'Not specified')
            }
            formatted.append(formatted_inspection)
        
        return formatted

def generate_individual_google_maps_link(row) -> str:
    """Generate Google Maps link for a single establishment location"""
    # Skip if no valid address information
    if (row['match_type'] == 'LABELER' and 
        ('Address not available' in str(row['address_line_1']) or 
         not row['address_line_1'] or 
         row['address_line_1'] == 'Unknown')):
        return None
        
    # Build address for this establishment
    address_parts = []
    if row['establishment_name'] and row['establishment_name'] != 'Unknown':
        address_parts.append(row['establishment_name'])
    if row['address_line_1'] and row['address_line_1'] != 'Unknown':
        address_parts.append(row['address_line_1'])
    if row['city'] and row['city'] != 'Unknown':
        address_parts.append(row['city'])
    if row['state'] and row['state'] != 'Unknown':
        address_parts.append(row['state'])
    if row['postal_code']:
        address_parts.append(row['postal_code'])
    if row['country'] and row['country'] != 'Unknown':
        address_parts.append(row['country'])
    
    if not address_parts:
        return None
    
    # Create Google Maps search URL for this specific location
    full_address = ', '.join(address_parts)
    encoded_address = full_address.replace(' ', '+').replace(',', '%2C').replace('&', '%26')
    return f"https://www.google.com/maps/search/{encoded_address}"

def generate_full_address(row) -> str:
    """Generate full address string for an establishment"""
    address_parts = []
    
    if row['establishment_name'] and row['establishment_name'] != 'Unknown':
        address_parts.append(row['establishment_name'])
    if row['address_line_1'] and row['address_line_1'] != 'Unknown':
        address_parts.append(row['address_line_1'])
    if row['city'] and row['city'] != 'Unknown':
        address_parts.append(row['city'])
    if row['state'] and row['state'] != 'Unknown':
        address_parts.append(row['state'])
    if row['postal_code']:
        address_parts.append(row['postal_code'])
    if row['country'] and row['country'] != 'Unknown':
        address_parts.append(row['country'])
    
    return ', '.join(address_parts) if address_parts else 'Address not available'

def create_simple_world_map(results_df):
    """Simple choropleth map showing facilities by country"""
    if len(results_df) == 0:
        return None
    
    country_counts = results_df['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'facility_count']
    
    # Complete country to ISO mapping with all variations including (ISO) format
    country_iso = {
        # North America
        'USA': 'USA', 'United States': 'USA', 'United States of America': 'USA',
        'USA (USA)': 'USA', 'United States (USA)': 'USA',
        'Canada': 'CAN', 'Canada (CAN)': 'CAN',
        'Mexico': 'MEX', 'Mexico (MEX)': 'MEX',
        'Guatemala': 'GTM', 'Guatemala (GTM)': 'GTM',
        'Belize': 'BLZ', 'Belize (BLZ)': 'BLZ',
        'El Salvador': 'SLV', 'El Salvador (SLV)': 'SLV',
        'Honduras': 'HND', 'Honduras (HND)': 'HND',
        'Nicaragua': 'NIC', 'Nicaragua (NIC)': 'NIC',
        'Costa Rica': 'CRI', 'Costa Rica (CRI)': 'CRI',
        'Panama': 'PAN', 'Panama (PAN)': 'PAN',
        
        # Europe
        'Germany': 'DEU', 'Germany (DEU)': 'DEU',
        'France': 'FRA', 'France (FRA)': 'FRA',
        'Italy': 'ITA', 'Italy (ITA)': 'ITA',
        'Spain': 'ESP', 'Spain (ESP)': 'ESP',
        'United Kingdom': 'GBR', 'UK': 'GBR', 'Britain': 'GBR', 'Great Britain': 'GBR',
        'United Kingdom (GBR)': 'GBR', 'UK (GBR)': 'GBR', 'Britain (GBR)': 'GBR',
        'England': 'GBR', 'Scotland': 'GBR', 'Wales': 'GBR', 'Northern Ireland': 'GBR',
        'England (GBR)': 'GBR', 'Scotland (GBR)': 'GBR', 'Wales (GBR)': 'GBR',
        'Netherlands': 'NLD', 'Holland': 'NLD', 'Netherlands (NLD)': 'NLD',
        'Belgium': 'BEL', 'Belgium (BEL)': 'BEL',
        'Switzerland': 'CHE', 'Switzerland (CHE)': 'CHE',
        'Austria': 'AUT', 'Austria (AUT)': 'AUT',
        'Sweden': 'SWE', 'Sweden (SWE)': 'SWE',
        'Norway': 'NOR', 'Norway (NOR)': 'NOR',
        'Denmark': 'DNK', 'Denmark (DNK)': 'DNK',
        'Finland': 'FIN', 'Finland (FIN)': 'FIN',
        'Iceland': 'ISL', 'Iceland (ISL)': 'ISL',
        'Ireland': 'IRL', 'Ireland (IRL)': 'IRL',
        'Portugal': 'PRT', 'Portugal (PRT)': 'PRT',
        'Greece': 'GRC', 'Greece (GRC)': 'GRC',
        'Poland': 'POL', 'Poland (POL)': 'POL',
        'Czech Republic': 'CZE', 'Czechia': 'CZE', 'Czech Republic (CZE)': 'CZE',
        'Slovakia': 'SVK', 'Slovakia (SVK)': 'SVK',
        'Hungary': 'HUN', 'Hungary (HUN)': 'HUN',
        'Romania': 'ROU', 'Romania (ROU)': 'ROU',
        'Bulgaria': 'BGR', 'Bulgaria (BGR)': 'BGR',
        'Croatia': 'HRV', 'Croatia (HRV)': 'HRV',
        'Slovenia': 'SVN', 'Slovenia (SVN)': 'SVN',
        'Bosnia and Herzegovina': 'BIH', 'Bosnia and Herzegovina (BIH)': 'BIH',
        'Serbia': 'SRB', 'Serbia (SRB)': 'SRB',
        'Montenegro': 'MNE', 'Montenegro (MNE)': 'MNE',
        'North Macedonia': 'MKD', 'Macedonia': 'MKD', 'North Macedonia (MKD)': 'MKD',
        'Albania': 'ALB', 'Albania (ALB)': 'ALB',
        'Lithuania': 'LTU', 'Lithuania (LTU)': 'LTU',
        'Latvia': 'LVA', 'Latvia (LVA)': 'LVA',
        'Estonia': 'EST', 'Estonia (EST)': 'EST',
        'Belarus': 'BLR', 'Belarus (BLR)': 'BLR',
        'Ukraine': 'UKR', 'Ukraine (UKR)': 'UKR',
        'Moldova': 'MDA', 'Moldova (MDA)': 'MDA',
        'Russia': 'RUS', 'Russian Federation': 'RUS', 'Russia (RUS)': 'RUS',
        'Turkey': 'TUR', 'Turkey (TUR)': 'TUR',
        'Cyprus': 'CYP', 'Cyprus (CYP)': 'CYP',
        'Malta': 'MLT', 'Malta (MLT)': 'MLT',
        'Luxembourg': 'LUX', 'Luxembourg (LUX)': 'LUX',
        'Monaco': 'MCO', 'Monaco (MCO)': 'MCO',
        'Liechtenstein': 'LIE', 'Liechtenstein (LIE)': 'LIE',
        'San Marino': 'SMR', 'San Marino (SMR)': 'SMR',
        'Vatican City': 'VAT', 'Vatican City (VAT)': 'VAT',
        'Andorra': 'AND', 'Andorra (AND)': 'AND',
        
        # Asia
        'China': 'CHN', 'People\'s Republic of China': 'CHN', 'China (CHN)': 'CHN',
        'India': 'IND', 'India (IND)': 'IND',
        'Japan': 'JPN', 'Japan (JPN)': 'JPN',
        'South Korea': 'KOR', 'Korea': 'KOR', 'Republic of Korea': 'KOR',
        'South Korea (KOR)': 'KOR', 'Korea (KOR)': 'KOR',
        'North Korea': 'PRK', 'Democratic People\'s Republic of Korea': 'PRK',
        'North Korea (PRK)': 'PRK',
        'Indonesia': 'IDN', 'Indonesia (IDN)': 'IDN',
        'Thailand': 'THA', 'Thailand (THA)': 'THA',
        'Vietnam': 'VNM', 'Vietnam (VNM)': 'VNM',
        'Philippines': 'PHL', 'Philippines (PHL)': 'PHL',
        'Malaysia': 'MYS', 'Malaysia (MYS)': 'MYS',
        'Singapore': 'SGP', 'Singapore (SGP)': 'SGP',
        'Myanmar': 'MMR', 'Burma': 'MMR', 'Myanmar (MMR)': 'MMR',
        'Cambodia': 'KHM', 'Cambodia (KHM)': 'KHM',
        'Laos': 'LAO', 'Laos (LAO)': 'LAO',
        'Brunei': 'BRN', 'Brunei (BRN)': 'BRN',
        'Taiwan': 'TWN', 'Republic of China': 'TWN', 'Taiwan (TWN)': 'TWN',
        'Hong Kong': 'HKG', 'Hong Kong (HKG)': 'HKG',
        'Macau': 'MAC', 'Macau (MAC)': 'MAC',
        'Mongolia': 'MNG', 'Mongolia (MNG)': 'MNG',
        'Kazakhstan': 'KAZ', 'Kazakhstan (KAZ)': 'KAZ',
        'Uzbekistan': 'UZB', 'Uzbekistan (UZB)': 'UZB',
        'Turkmenistan': 'TKM', 'Turkmenistan (TKM)': 'TKM',
        'Kyrgyzstan': 'KGZ', 'Kyrgyzstan (KGZ)': 'KGZ',
        'Tajikistan': 'TJK', 'Tajikistan (TJK)': 'TJK',
        'Afghanistan': 'AFG', 'Afghanistan (AFG)': 'AFG',
        'Pakistan': 'PAK', 'Pakistan (PAK)': 'PAK',
        'Bangladesh': 'BGD', 'Bangladesh (BGD)': 'BGD',
        'Sri Lanka': 'LKA', 'Sri Lanka (LKA)': 'LKA',
        'Nepal': 'NPL', 'Nepal (NPL)': 'NPL',
        'Bhutan': 'BTN', 'Bhutan (BTN)': 'BTN',
        'Maldives': 'MDV', 'Maldives (MDV)': 'MDV',
        'Iran': 'IRN', 'Islamic Republic of Iran': 'IRN', 'Iran (IRN)': 'IRN',
        'Iraq': 'IRQ', 'Iraq (IRQ)': 'IRQ',
        'Syria': 'SYR', 'Syria (SYR)': 'SYR',
        'Lebanon': 'LBN', 'Lebanon (LBN)': 'LBN',
        'Jordan': 'JOR', 'Jordan (JOR)': 'JOR',
        'Israel': 'ISR', 'Israel (ISR)': 'ISR',
        'Palestine': 'PSE', 'Palestine (PSE)': 'PSE',
        'Saudi Arabia': 'SAU', 'Saudi Arabia (SAU)': 'SAU',
        'Yemen': 'YEM', 'Yemen (YEM)': 'YEM',
        'Oman': 'OMN', 'Oman (OMN)': 'OMN',
        'United Arab Emirates': 'ARE', 'UAE': 'ARE', 'United Arab Emirates (ARE)': 'ARE',
        'Qatar': 'QAT', 'Qatar (QAT)': 'QAT',
        'Bahrain': 'BHR', 'Bahrain (BHR)': 'BHR',
        'Kuwait': 'KWT', 'Kuwait (KWT)': 'KWT',
        'Georgia': 'GEO', 'Georgia (GEO)': 'GEO',
        'Armenia': 'ARM', 'Armenia (ARM)': 'ARM',
        'Azerbaijan': 'AZE', 'Azerbaijan (AZE)': 'AZE',
        
        # Africa
        'South Africa': 'ZAF', 'South Africa (ZAF)': 'ZAF',
        'Nigeria': 'NGA', 'Nigeria (NGA)': 'NGA',
        'Egypt': 'EGY', 'Egypt (EGY)': 'EGY',
        'Kenya': 'KEN', 'Kenya (KEN)': 'KEN',
        'Ethiopia': 'ETH', 'Ethiopia (ETH)': 'ETH',
        'Ghana': 'GHA', 'Ghana (GHA)': 'GHA',
        'Morocco': 'MAR', 'Morocco (MAR)': 'MAR',
        'Algeria': 'DZA', 'Algeria (DZA)': 'DZA',
        'Tunisia': 'TUN', 'Tunisia (TUN)': 'TUN',
        'Libya': 'LBY', 'Libya (LBY)': 'LBY',
        'Sudan': 'SDN', 'Sudan (SDN)': 'SDN',
        'South Sudan': 'SSD', 'South Sudan (SSD)': 'SSD',
        'Chad': 'TCD', 'Chad (TCD)': 'TCD',
        'Niger': 'NER', 'Niger (NER)': 'NER',
        'Mali': 'MLI', 'Mali (MLI)': 'MLI',
        'Burkina Faso': 'BFA', 'Burkina Faso (BFA)': 'BFA',
        'Senegal': 'SEN', 'Senegal (SEN)': 'SEN',
        'Guinea': 'GIN', 'Guinea (GIN)': 'GIN',
        'Sierra Leone': 'SLE', 'Sierra Leone (SLE)': 'SLE',
        'Liberia': 'LBR', 'Liberia (LBR)': 'LBR',
        'Ivory Coast': 'CIV', 'Côte d\'Ivoire': 'CIV', 'Ivory Coast (CIV)': 'CIV',
        'Togo': 'TGO', 'Togo (TGO)': 'TGO',
        'Benin': 'BEN', 'Benin (BEN)': 'BEN',
        'Cameroon': 'CMR', 'Cameroon (CMR)': 'CMR',
        'Central African Republic': 'CAF', 'Central African Republic (CAF)': 'CAF',
        'Democratic Republic of the Congo': 'COD', 'Congo (DRC)': 'COD',
        'Democratic Republic of the Congo (COD)': 'COD',
        'Republic of the Congo': 'COG', 'Congo': 'COG', 'Republic of the Congo (COG)': 'COG',
        'Gabon': 'GAB', 'Gabon (GAB)': 'GAB',
        'Equatorial Guinea': 'GNQ', 'Equatorial Guinea (GNQ)': 'GNQ',
        'São Tomé and Príncipe': 'STP', 'São Tomé and Príncipe (STP)': 'STP',
        'Angola': 'AGO', 'Angola (AGO)': 'AGO',
        'Zambia': 'ZMB', 'Zambia (ZMB)': 'ZMB',
        'Zimbabwe': 'ZWE', 'Zimbabwe (ZWE)': 'ZWE',
        'Botswana': 'BWA', 'Botswana (BWA)': 'BWA',
        'Namibia': 'NAM', 'Namibia (NAM)': 'NAM',
        'Lesotho': 'LSO', 'Lesotho (LSO)': 'LSO',
        'Eswatini': 'SWZ', 'Swaziland': 'SWZ', 'Eswatini (SWZ)': 'SWZ',
        'Mozambique': 'MOZ', 'Mozambique (MOZ)': 'MOZ',
        'Malawi': 'MWI', 'Malawi (MWI)': 'MWI',
        'Tanzania': 'TZA', 'Tanzania (TZA)': 'TZA',
        'Uganda': 'UGA', 'Uganda (UGA)': 'UGA',
        'Rwanda': 'RWA', 'Rwanda (RWA)': 'RWA',
        'Burundi': 'BDI', 'Burundi (BDI)': 'BDI',
        'Djibouti': 'DJI', 'Djibouti (DJI)': 'DJI',
        'Eritrea': 'ERI', 'Eritrea (ERI)': 'ERI',
        'Somalia': 'SOM', 'Somalia (SOM)': 'SOM',
        'Madagascar': 'MDG', 'Madagascar (MDG)': 'MDG',
        'Mauritius': 'MUS', 'Mauritius (MUS)': 'MUS',
        'Seychelles': 'SYC', 'Seychelles (SYC)': 'SYC',
        'Comoros': 'COM', 'Comoros (COM)': 'COM',
        'Cape Verde': 'CPV', 'Cape Verde (CPV)': 'CPV',
        'Gambia': 'GMB', 'Gambia (GMB)': 'GMB',
        'Guinea-Bissau': 'GNB', 'Guinea-Bissau (GNB)': 'GNB',
        
        # South America
        'Brazil': 'BRA', 'Brazil (BRA)': 'BRA',
        'Argentina': 'ARG', 'Argentina (ARG)': 'ARG',
        'Chile': 'CHL', 'Chile (CHL)': 'CHL',
        'Peru': 'PER', 'Peru (PER)': 'PER',
        'Colombia': 'COL', 'Colombia (COL)': 'COL',
        'Venezuela': 'VEN', 'Venezuela (VEN)': 'VEN',
        'Ecuador': 'ECU', 'Ecuador (ECU)': 'ECU',
        'Bolivia': 'BOL', 'Bolivia (BOL)': 'BOL',
        'Paraguay': 'PRY', 'Paraguay (PRY)': 'PRY',
        'Uruguay': 'URY', 'Uruguay (URY)': 'URY',
        'Guyana': 'GUY', 'Guyana (GUY)': 'GUY',
        'Suriname': 'SUR', 'Suriname (SUR)': 'SUR',
        'French Guiana': 'GUF', 'French Guiana (GUF)': 'GUF',
        
        # Oceania
        'Australia': 'AUS', 'Australia (AUS)': 'AUS',
        'New Zealand': 'NZL', 'New Zealand (NZL)': 'NZL',
        'Papua New Guinea': 'PNG', 'Papua New Guinea (PNG)': 'PNG',
        'Fiji': 'FJI', 'Fiji (FJI)': 'FJI',
        'Solomon Islands': 'SLB', 'Solomon Islands (SLB)': 'SLB',
        'Vanuatu': 'VUT', 'Vanuatu (VUT)': 'VUT',
        'Samoa': 'WSM', 'Samoa (WSM)': 'WSM',
        'Tonga': 'TON', 'Tonga (TON)': 'TON',
        'Kiribati': 'KIR', 'Kiribati (KIR)': 'KIR',
        'Tuvalu': 'TUV', 'Tuvalu (TUV)': 'TUV',
        'Nauru': 'NRU', 'Nauru (NRU)': 'NRU',
        'Palau': 'PLW', 'Palau (PLW)': 'PLW',
        'Marshall Islands': 'MHL', 'Marshall Islands (MHL)': 'MHL',
        'Micronesia': 'FSM', 'Micronesia (FSM)': 'FSM',
        
        # Caribbean
        'Cuba': 'CUB', 'Cuba (CUB)': 'CUB',
        'Jamaica': 'JAM', 'Jamaica (JAM)': 'JAM',
        'Haiti': 'HTI', 'Haiti (HTI)': 'HTI',
        'Dominican Republic': 'DOM', 'Dominican Republic (DOM)': 'DOM',
        'Puerto Rico': 'PRI', 'Puerto Rico (PRI)': 'PRI',
        'Trinidad and Tobago': 'TTO', 'Trinidad and Tobago (TTO)': 'TTO',
        'Barbados': 'BRB', 'Barbados (BRB)': 'BRB',
        'Bahamas': 'BHS', 'Bahamas (BHS)': 'BHS',
        'Saint Lucia': 'LCA', 'Saint Lucia (LCA)': 'LCA',
        'Grenada': 'GRD', 'Grenada (GRD)': 'GRD',
        'Saint Vincent and the Grenadines': 'VCT', 'Saint Vincent and the Grenadines (VCT)': 'VCT',
        'Antigua and Barbuda': 'ATG', 'Antigua and Barbuda (ATG)': 'ATG',
        'Dominica': 'DMA', 'Dominica (DMA)': 'DMA',
        'Saint Kitts and Nevis': 'KNA', 'Saint Kitts and Nevis (KNA)': 'KNA'
    }
    
    country_counts['iso'] = country_counts['country'].map(country_iso)
    
    # Filter out unmapped countries for the map
    mapped_countries = country_counts[country_counts['iso'].notna()]
    
    # Return None if no countries can be mapped
    if len(mapped_countries) == 0:
        return None
    
    # Get the range of facility counts for better color scaling
    min_facilities = mapped_countries['facility_count'].min()
    max_facilities = mapped_countries['facility_count'].max()
    
    # Create custom color scale - darker blue to very dark blue (more visible)
    fig = go.Figure(data=go.Choropleth(
        locations=mapped_countries['iso'],
        z=mapped_countries['facility_count'],
        text=mapped_countries['country'],
        colorscale=[
            [0.0, '#64B5F6'],  # Darker light blue for 1 facility (much more visible)
            [0.2, '#42A5F5'],  # Medium blue
            [0.4, '#2196F3'],  # Blue
            [0.6, '#1976D2'],  # Dark blue
            [0.8, '#0D47A1'],  # Very dark blue
            [1.0, '#0A1929']   # Black/blue for highest count
        ],
        colorbar=dict(
            title="Facilities",
            tickmode='linear',
            tick0=min_facilities,
            dtick=1,  # Show only whole numbers
            tickformat='d'  # Integer format
        ),
        hovertemplate='<b>%{text}</b><br>Facilities: %{z}<extra></extra>',
        zmin=min_facilities,
        zmax=max_facilities
    ))
    
    fig.update_layout(
        title='',  # Remove title
        geo=dict(showframe=False, showcoastlines=True),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)  # Remove margins around the map
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Medication Manufacturing Location Lookup", 
        page_icon="💊",
        layout="wide"
    )
    
    st.title("💊 Medication Manufacturing Location Lookup")
    st.markdown("### Find where medications are manufactured")
    st.markdown("Enter a National Drug Code (NDC) number to see if it has manufacturing establishments, locations, and operations in public FDA data.")
    
    # SIMPLIFIED LOADING - No complex progress bars needed!
    if 'mapper' not in st.session_state:
        # Simple spinner for fast loading (2-5 seconds)
        with st.spinner("🚀 Loading databases..."):
            st.session_state.mapper = NDCToLocationMapper()
        
        # Simple error handling
        if not st.session_state.mapper.database_loaded:
            st.error("❌ Could not load database files. Please ensure the optimized database file is available.")
            st.info("💡 Make sure 'fda_databases_optimized.pkl' is in your app directory.")
            st.stop()
        else:
            # Brief success message that auto-disappears
            success_placeholder = st.empty()
            success_placeholder.success("✅ Database loaded successfully!")
            time.sleep(1)
            success_placeholder.empty()

    # Input section with Enter key functionality - no horizontal line
    # Use form to enable Enter key submission
    with st.form("ndc_search_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            ndc_input = st.text_input(
                "Enter NDC Number:", 
                placeholder="Example: 0000-0000-00",
                help="National Drug Code format: 12345-678-90 or 1234567890"
            )
        with col2:
            st.write("")  # Spacing
            search_btn = st.form_submit_button("🔍 Search", type="primary")
    
    # Try A Random NDC button
    if st.button("Try An Example NDC", key="random_ndc_btn"):
        import random
        random_ndcs = [
            "50242-060-01",
            "0003-0893-91", 
            "0169-4132-11",
            "61755-005-54",
            "0955-3900-01",
            "0069-0043-02",
            "50242-077-01",
            "65628-206-05",
            "57894-030-01"
        ]
        ndc_input = random.choice(random_ndcs)
        search_btn = True
    
    # Information link - removed "About FDA drug databases"
    st.markdown("📖 [How to find a medication's National Drug Code](https://dailymed.nlm.nih.gov/dailymed/help.cfm)")
    
    # Search functionality
    if search_btn and ndc_input:
        st.write("🔍 **Debug Output:**")
        
        with st.spinner(f"Looking up manufacturing locations for {ndc_input}..."):
            try:
                # ADDED: First, let's see what the SPL contains
                product_info = st.session_state.mapper.get_ndc_info_comprehensive(ndc_input)
                if product_info and product_info.spl_id:
                    st.write(f"**SPL ID:** {product_info.spl_id}")
                    
                    # ADDED: Show all establishments found in SPL
                    matches = st.session_state.mapper.find_fei_duns_matches_in_spl(product_info.spl_id)
                results_df = st.session_state.mapper.process_single_ndc(ndc_input)
                
                if len(results_df) > 0:
                    first_row = results_df.iloc[0]
                    
                    if first_row['search_method'] == 'no_establishments_found':
                        # FIXED: Show proper message for no manufacturing establishments
                        st.warning(f"⚠️ No manufacturing establishments were identified in the structured product label")
                        
                        # Product, Labeler, and NDC with same text size
                        st.subheader(f"**Product:** {first_row['product_name']}")
                        st.subheader(f"**Labeler:** {first_row['labeler_name']}")
                        st.subheader(f"**National Drug Code:** {first_row['ndc']}")
                        
                        if first_row['spl_id']:
                            spl_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={first_row['spl_id']}"
                            st.markdown(f"📄 **Structured Product Label:** [View on DailyMed]({spl_url})")
                        
                        st.info("💡 This product may not have detailed establishment information in its official documentation (~70% of products don't include manufacturing details).")
                    
                    else:
                        # Full results with establishments
                        st.success(f"✅ Found {len(results_df)} manufacturing establishment{'s' if len(results_df) != 1 else ''}")
                        
                        # Product, Labeler, and NDC with same text size
                        st.subheader(f"**Product:** {first_row['product_name']}")
                        st.subheader(f"**Labeler:** {first_row['labeler_name']}")
                        st.subheader(f"**National Drug Code:** {first_row['ndc']}")
                        
                        if first_row['spl_id']:
                            spl_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={first_row['spl_id']}"
                            st.markdown(f"📄 **Structured Product Label:** [View on DailyMed]({spl_url})")
                        
                        # Manufacturing establishments header with count first and countries
                        country_counts = results_df['country'].value_counts()
                        country_summary = ", ".join([f"{country}: {count}" for country, count in country_counts.items()])
                        st.subheader(f"🏭 {len(results_df)} Manufacturing Establishment{'s' if len(results_df) != 1 else ''} in Public Data - {country_summary}")

                        # Add map right after the header
                        map_fig = create_simple_world_map(results_df)
                        if map_fig:
                            st.plotly_chart(map_fig, use_container_width=True)
                        
                        # Manufacturing establishments - header without address
                        for idx, row in results_df.iterrows():
                            # Use just "Establishment X" in header, removing any address
                            with st.expander(f"Establishment {idx + 1}", expanded=True):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Show establishment name in content, not header
                                    if row['fei_number']:
                                        st.write(f"**🔢 FDA Establishment Identifier:** {row['fei_number']}")
                                    if row['duns_number']:
                                        st.write(f"**🔢 Business Identifier:** {row['duns_number']}")
                                    if row['firm_name'] and row['firm_name'] != 'Unknown':
                                        st.write(f"**🏢 Company Name:** {row['firm_name']}")
                                
                                with col2:
                                    if row['country'] and row['country'] != 'Unknown':
                                        st.write(f"**🌍 Country:** {row['country']}")
                                    if row['spl_operations'] and row['spl_operations'] != 'None found for this National Drug Code':
                                        st.write(f"**⚙️ Manufacturing Operations:** {row['spl_operations']}")

                                # Show inspection information if available
                                if row['fei_number']:
                                    inspections = st.session_state.mapper.get_facility_inspections(row['fei_number'])
                                    if inspections:
                                        inspection_summary = st.session_state.mapper.get_inspection_summary(inspections)
                                        st.write(f"**🔍 Latest Inspection:** {inspection_summary['status']}")
                                
                                # Full address in address section
                                full_address = generate_full_address(row)
                                if full_address != 'Address not available':
                                    st.write(f"**📍 Address:** {full_address}")
                                    
                                    maps_link = generate_individual_google_maps_link(row)
                                    if maps_link:
                                        st.markdown(f"🗺️ [View on Google Maps]({maps_link})")
                                else:
                                    st.write("**📍 Address:** Address not available")
                        
                        # CSV Download option (no header, just button)
                        # Prepare clean CSV data
                        csv_data = results_df.copy()
                        csv_data['full_address'] = csv_data.apply(generate_full_address, axis=1)
                        
                        # Select relevant columns for CSV
                        csv_columns = ['ndc', 'product_name', 'labeler_name', 'establishment_name', 
                                     'firm_name', 'full_address', 'country', 'spl_operations']
                        if any(results_df['fei_number'].notna()):
                            csv_columns.append('fei_number')
                        if any(results_df['duns_number'].notna()):
                            csv_columns.append('duns_number')
                        
                        csv_export = csv_data[csv_columns].to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv_export,
                            file_name=f"ndc_{ndc_input.replace('-', '')}_establishments.csv",
                            mime="text/csv"
                        )
                        
                else:
                    st.error(f"❌ No results found for: {ndc_input}")
                    st.info("💡 This National Drug Code may not exist in the FDA database. Please check the format and try again.")
                    
            except Exception as e:
                st.error(f"❌ Error processing request: {str(e)}")
                with st.expander("Technical Details"):
                    st.exception(e)
    
    # Sidebar info
    st.sidebar.title("About This Tool")
    st.sidebar.markdown("""
    🔍 **Look up your medication** in public FDA databases  
    📄 **Analyze official documents** for manufacturing info  
    🏭 **Find manufacturing facilities** worldwide  
    🌍 **Show locations** on maps  
    
    **Coverage:**
    Approximately 30% of medications have some manufacturing establishment info in public FDA data.
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚠️ Important Disclaimer:**")
    st.sidebar.markdown("""
    This tool is provided for **informational and educational purposes only**. 
    
    - Information may not be complete or current
    - Not intended for medical decision-making
    - Does not replace consultation with healthcare providers
    - Manufacturing locations may change over time
    - Always consult your pharmacist or doctor for medication questions
    
    Data sources: FDA databases, official product labeling, and establishment registrations.
    """)

    if 'mapper' in st.session_state and st.session_state.mapper.database_loaded:
        st.sidebar.markdown("---")
                # Add database date if available
        if st.session_state.mapper.database_date:
            st.sidebar.markdown(f"**Database Date:** {st.session_state.mapper.database_date}")       
        st.sidebar.markdown("**Database Status:**")
        st.sidebar.success("✅ Loaded and Ready")

if __name__ == "__main__":
    main()

