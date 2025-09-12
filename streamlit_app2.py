import streamlit as st
import pandas as pd
import os
import requests
import json
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import re
import warnings
from datetime import datetime, timedelta

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
        self.database_loaded = False
        self.database_date = None  # Track when database was created
        
        # Auto-load database
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
                # More flexible DUNS column matching
                elif ('duns' in col_lower and 'number' in col_lower) or col_lower == 'dunsnumber':
                    duns_col = col_original
                # More flexible ADDRESS column matching
                elif 'address' in col_lower:
                    address_col = col_original
                # More flexible FIRM_NAME column matching
                elif ('firm' in col_lower and 'name' in col_lower) or col_lower == 'firmname':
                    firm_name_col = col_original

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

                    # Get BOTH FEI and DUNS from the same row
                    fei_number = None
                    duns_number = None
                    
                    if fei_col and not pd.isna(row[fei_col]):
                        fei_raw = str(row[fei_col]).strip()
                        if fei_raw != 'nan' and fei_raw != '' and fei_raw != '0000000' and fei_raw != '0000000000':
                            fei_clean = re.sub(r'[^\d]', '', fei_raw)
                            if len(fei_clean) >= 7:
                                fei_number = fei_raw

                    if duns_col and not pd.isna(row[duns_col]):
                        duns_raw = str(row[duns_col]).strip()
                        if duns_raw != 'nan' and duns_raw != '':
                            duns_clean = re.sub(r'[^\d]', '', duns_raw)
                            if len(duns_clean) >= 8:
                                duns_number = duns_raw

                    # Only process if we have at least one identifier
                    if not fei_number and not duns_number:
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
                        'original_fei': fei_number,      # Store both identifiers
                        'original_duns': duns_number     # Store both identifiers
                    }

                    # Store under FEI variants if FEI exists
                    if fei_number:
                        fei_variants = self._generate_all_id_variants(fei_number)
                        for key in fei_variants:
                            if key:
                                self.fei_database[key] = establishment_data.copy()
                        fei_count += 1

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
        """Load inspection outcomes database from spreadsheet"""
        try:
            # Read the inspection data file
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
            return date_only
        
        sorted_inspections = sorted(inspections, 
                                  key=lambda x: parse_date(x.get('inspection_date', '')), 
                                  reverse=True)
        most_recent = sorted_inspections[0]
        
        # Get the most recent date and outcome (remove timestamp)
        most_recent_date = parse_date(most_recent.get('inspection_date', 'Date unknown'))
        most_recent_classification = most_recent.get('classification', 'Outcome unknown')
        
        # Simplify classification names
        classification_map = {
            'No Action Indicated (NAI)': 'NAI',
            'Voluntary Action Indicated (VAI)': 'VAI', 
            'Official Action Indicated (OAI)': 'OAI',
            'No Action Indicated': 'NAI',
            'Voluntary Action Indicated': 'VAI',
            'Official Action Indicated': 'OAI'
        }
        
        simplified_classification = classification_map.get(most_recent_classification, most_recent_classification)
        
        # Find all OAI inspection dates
        oai_dates = []
        for inspection in sorted_inspections:
            classification = inspection.get('classification', '')
            if 'Official Action Indicated' in classification or 'OAI' in classification:
                oai_date = parse_date(inspection.get('inspection_date', ''))
                if oai_date and oai_date not in oai_dates:
                    oai_dates.append(oai_date)
        
        # Build status string
        if most_recent_date != 'Date unknown':
            status = f"{simplified_classification} {most_recent_date}"
        else:
            status = simplified_classification
        
        # Add OAI dates if any exist and they're different from most recent
        if oai_dates and (not oai_dates or oai_dates[0] != most_recent_date or simplified_classification != 'OAI'):
            if len(oai_dates) == 1:
                status += f" | OAI: {oai_dates[0]}"
            else:
                status += f" | OAI dates: {', '.join(oai_dates)}"
        
        return {
            'total_records': len(inspections),
            'most_recent_date': most_recent_date,
            'most_recent_outcome': simplified_classification,
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

    def get_ndc_info_comprehensive(self, ndc: str) -> Optional[ProductInfo]:
        """Get NDC info from DailyMed"""
        try:
            search_url = f"{self.dailymed_base_url}/services/v2/spls.json"
            params = {'ndc': ndc, 'page_size': 1}
            response = self.session.get(search_url, params=params)

            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    spl_data = data['data'][0]
                    product_name = spl_data.get('title', 'Unknown')
                    labeler_name = spl_data.get('labeler', 'Unknown')
                    
                    return ProductInfo(
                        ndc=ndc,
                        product_name=product_name,
                        labeler_name=labeler_name,
                        spl_id=spl_data.get('setid')
                    )
        except Exception as e:
            pass

        return None

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
            # Try EXPANDED formats for DUNS lookup
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
        """Find FEI and DUNS numbers in SPL that match the spreadsheet database"""
        matches = []
        
        try:
            spl_url = f"{self.dailymed_base_url}/services/v2/spls/{spl_id}.xml"
            response = self.session.get(spl_url)

            if response.status_code != 200:
                return matches

            content = response.text
            id_pattern = r'<id\s+([^>]*extension="(\d{7,15})"[^>]*)'
            id_matches = re.findall(id_pattern, content, re.IGNORECASE)
            
            for full_match, extension in id_matches:
                clean_extension = re.sub(r'[^\d]', '', extension)
                
                fei_match_found = False
                fei_variants = self._generate_all_id_variants(extension)
                
                for fei_key in fei_variants:
                    if fei_key in self.fei_database:
                        establishment_name = self.fei_database[fei_key].get('establishment_name', 'Unknown')
                        
                        match = FEIMatch(
                            fei_number=clean_extension,
                            xml_location="SPL Document",
                            match_type='FEI_NUMBER',
                            establishment_name=establishment_name
                        )
                        matches.append(match)
                        fei_match_found = True
                        break
                
                if not fei_match_found:
                    duns_variants = self._generate_all_id_variants(extension)
                    
                    for duns_key in duns_variants:
                        if duns_key in self.duns_database:
                            establishment_name = self.duns_database[duns_key].get('establishment_name', 'Unknown')
                            
                            match = FEIMatch(
                                fei_number=clean_extension,
                                xml_location="SPL Document",
                                match_type='DUNS_NUMBER',
                                establishment_name=establishment_name
                            )
                            matches.append(match)
                            break
                            
        except Exception as e:
            pass
            
        return matches

    def extract_establishments_with_fei(self, spl_id: str, target_ndc: str) -> Tuple[List[str], List[str], List[Dict]]:
        """Extract operations, quotes, and detailed establishment info with FEI/DUNS numbers for specific NDC"""
        try:
            spl_url = f"{self.dailymed_base_url}/services/v2/spls/{spl_id}.xml"
            response = self.session.get(spl_url)

            if response.status_code != 200:
                return [], [], []

            content = response.text
            establishments_info = []
            processed_numbers = set()

            matches = self.find_fei_duns_matches_in_spl(spl_id)
            establishment_sections = re.findall(r'<assignedEntity[^>]*>.*?</assignedEntity>', content, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                if match.fei_number in processed_numbers:
                    continue
                
                processed_numbers.add(match.fei_number)
                
                if match.match_type == 'FEI_NUMBER':
                    establishment_info = self.lookup_fei_establishment(match.fei_number)
                else:
                    establishment_info = self.lookup_duns_establishment(match.fei_number)
                
                if establishment_info:
                    establishment_operations = ['Manufacture']  # Default operation
                    establishment_quotes = ['Found establishment in SPL']
                    
                    establishment_info['xml_location'] = match.xml_location
                    establishment_info['match_type'] = match.match_type
                    establishment_info['xml_context'] = ''
                    
                    establishment_info['operations'] = establishment_operations
                    establishment_info['quotes'] = establishment_quotes
                    
                    establishments_info.append(establishment_info)

            return [], [], establishments_info

        except Exception as e:
            return [], [], []

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
                
                establishments = establishments_info
        
        return establishments[:10]

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

    def get_facility_inspections(self, fei_number: str) -> List[Dict]:
        """Get inspection history - prioritize local database, fallback to API"""
        inspections = []
        
        try:
            # First try local inspection database
            local_inspections = self.get_facility_inspections_from_database(fei_number)
            if local_inspections:
                inspections.extend(local_inspections)
            
            return inspections
            
        except Exception as e:
            logger.error(f"Error getting inspections for FEI {fei_number}: {str(e)}")
            return []

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
                            'inspection_date': record.get('inspection_end_date', ''),
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

def main():
    st.set_page_config(
        page_title="Medication Manufacturing Location Lookup", 
        page_icon="💊",
        layout="wide"
    )
    
    st.title("💊 Medication Manufacturing Location Lookup")
    st.markdown("### Find where your medications are manufactured")
    st.markdown("Enter a National Drug Code (NDC) number to see if it has manufacturing establishments, locations, and operations in public FDA data.")
    
    # Auto-load database and show status (simplified)
    if 'mapper' not in st.session_state:
        st.session_state.mapper = NDCToLocationMapper()
            
    if not st.session_state.mapper.database_loaded:
        st.error("❌ Could not load establishment database")
        st.stop()
    
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
    st.markdown("📖 [How to find your medication's National Drug Code](https://dailymed.nlm.nih.gov/dailymed/help.cfm)")
    
    # Search functionality
    if search_btn and ndc_input:
        with st.spinner(f"Looking up manufacturing locations for {ndc_input}..."):
            try:
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
                        
                        st.info("💡 This product may not have detailed establishment information in its official documentation (~70% of products don't include manufacturing details), or the establishments may not be in our database.")
                    
                    else:
                        # Full results with establishments
                        st.success(f"✅ Found {len(results_df)} manufacturing establishments")
                        
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
                        st.subheader(f"🏭 {len(results_df)} Manufacturing Establishments - {country_summary}")
                        
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
                                        st.write(f"**🔍 Inspection:** {inspection_summary['status']}")
                                
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
    
    **What you can discover:**
    - ✅ Where your medication is made
    - ✅ What company manufactures it  
    - ✅ Manufacturing operations performed
    - ✅ Global supply chain information
    - ✅ Interactive maps of facilities
    
    **Coverage:**
    Approximately 30% of medications have some manufacturing establishment info in public FDA data.
    """)
    
    if 'mapper' in st.session_state and st.session_state.mapper.database_loaded:
        st.sidebar.markdown("---")
        st.sidebar.metric("FDA Database Entries", f"{len(st.session_state.mapper.fei_database):,}")
        st.sidebar.metric("Business Database Entries", f"{len(st.session_state.mapper.duns_database):,}")
        if hasattr(st.session_state.mapper, 'inspection_database'):
            st.sidebar.metric("Facilities with Inspections", f"{len(st.session_state.mapper.inspection_database):,}")
        
        # Add database date if available
        if st.session_state.mapper.database_date:
            st.sidebar.markdown(f"**Database Date:** {st.session_state.mapper.database_date}")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Database Status:**")
        st.sidebar.success("✅ Loaded and Ready")
    
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

if __name__ == "__main__":
    main()
