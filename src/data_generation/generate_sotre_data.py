import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import string

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Store codes from the invoice dataset
store_codes = ['SH001', 'SH002', 'SH003', 'SH004', 'SH005', 'SH006', 'SH007', 'SH008', 'SH009', 'SH010', 
               'SH011', 'SH012', 'SH013', 'SH014', 'SH015']

# Indian cities and states for shoe stores
indian_cities_states = [
    ('Mumbai', 'Maharashtra', '400001'),
    ('Delhi', 'Delhi', '110001'),
    ('Bangalore', 'Karnataka', '560001'),
    ('Hyderabad', 'Telangana', '500001'),
    ('Chennai', 'Tamil Nadu', '600001'),
    ('Kolkata', 'West Bengal', '700001'),
    ('Pune', 'Maharashtra', '411001'),
    ('Ahmedabad', 'Gujarat', '380001'),
    ('Jaipur', 'Rajasthan', '302001'),
    ('Lucknow', 'Uttar Pradesh', '226001'),
    ('Kanpur', 'Uttar Pradesh', '208001'),
    ('Nagpur', 'Maharashtra', '440001'),
    ('Indore', 'Madhya Pradesh', '452001'),
    ('Bhopal', 'Madhya Pradesh', '462001'),
    ('Visakhapatnam', 'Andhra Pradesh', '530001')
]

# Store zones and regions
zones = ['North', 'South', 'East', 'West', 'Central']
regions = ['NCR', 'Maharashtra', 'South India', 'West India', 'East India', 'Central India', 'North India']

# Business names for shoe stores
business_names = [
    'Step Forward Shoes', 'Footwear Palace', 'Shoe Junction', 'Comfort Footwear', 'Style Steps',
    'Premium Footwear', 'Shoe Station', 'Footwear Express', 'Shoe World', 'Comfort Zone Shoes',
    'Elite Footwear', 'Shoe Gallery', 'Fashion Footwear', 'Shoe Paradise', 'Trendy Steps'
]

# Indian names for managers and employees
indian_names = [
    'Rajesh Kumar', 'Priya Sharma', 'Amit Singh', 'Sunita Patel', 'Vikash Gupta',
    'Neha Agarwal', 'Rohit Jain', 'Kavita Mehta', 'Suresh Reddy', 'Pooja Verma',
    'Manoj Tiwari', 'Ritu Bansal', 'Deepak Yadav', 'Anjali Khanna', 'Sanjay Mishra',
    'Rekha Saxena', 'Ashish Pandey', 'Shweta Srivastava', 'Rajeev Chandra', 'Meera Joshi',
    'Prakash Nair', 'Lakshmi Iyer', 'Kiran Desai', 'Ramesh Choudhary', 'Geeta Malhotra'
]

def generate_phone_number():
    """Generate Indian phone number"""
    return f"+91-{random.randint(70000, 99999)}{random.randint(10000, 99999)}"

def generate_landline():
    """Generate Indian landline number"""
    area_codes = ['011', '022', '044', '080', '040', '033', '020', '079', '0141', '0522']
    area_code = random.choice(area_codes)
    return f"{area_code}-{random.randint(20000000, 99999999)}"

def generate_gstin():
    """Generate GSTIN number format"""
    state_codes = ['27', '07', '29', '36', '33', '19', '21', '24', '08', '09']
    state_code = random.choice(state_codes)
    pan_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    entity_code = random.choice(['1', '2', '4'])
    check_digit = random.choice(string.ascii_uppercase + string.digits)
    return f"{state_code}{pan_part}{entity_code}Z{check_digit}"

def generate_email(name, domain_type='personal'):
    """Generate email address"""
    name_parts = name.lower().split()
    if len(name_parts) >= 2:
        email_name = f"{name_parts[0]}.{name_parts[1]}"
    else:
        email_name = name_parts[0]
    
    if domain_type == 'business':
        domains = ['company.com', 'footwear.co.in', 'shoes.in', 'retail.com']
    else:
        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
    
    domain = random.choice(domains)
    return f"{email_name}@{domain}"

def generate_employee_code():
    """Generate employee code"""
    return f"EMP{random.randint(1000, 9999)}"

def generate_sap_store_code(store_code):
    """Generate SAP store code"""
    return f"SAP{store_code}"

def generate_store_area():
    """Generate store carpet area in sq ft"""
    return random.randint(800, 3000)

def generate_store_header_data():
    """Generate store header data for all stores"""
    
    store_data = []
    
    for i, store_code in enumerate(store_codes):
        # Basic store information
        store_id = i + 1
        city, state, base_postal = random.choice(indian_cities_states)
        postal_code = str(int(base_postal) + random.randint(0, 999)).zfill(6)
        
        # Business details
        business_name = random.choice(business_names)
        address = f"{random.randint(1, 999)}, {random.choice(['MG Road', 'Commercial Street', 'Mall Road', 'Main Street', 'Market Street', 'Gandhi Road', 'Nehru Street'])}"
        
        # Location and operations
        zone = random.choice(zones)
        region = random.choice(regions)
        carpet_area = generate_store_area()
        area_grading = random.choice(['A+', 'A', 'B+', 'B', 'C+'])
        
        # Personnel
        rsm_name = random.choice(indian_names)
        asm_name = random.choice([name for name in indian_names if name != rsm_name])
        store_manager = random.choice([name for name in indian_names if name not in [rsm_name, asm_name]])
        merchandiser = random.choice(indian_names)
        merchandiser1 = random.choice([name for name in indian_names if name != merchandiser])
        
        # Technical details
        server_id = f"SRV{random.randint(100, 999)}"
        gstin = generate_gstin()
        
        # Store record
        store_record = {
            'ID': store_id,
            'Code': store_code,
            'BName': business_name,
            'Address': address,
            'City': city,
            'State': state,
            'PostalCode': postal_code,
            'Phone': generate_phone_number(),
            'sapStoreCode': generate_sap_store_code(store_code),
            'sapStoreTypeCode': random.choice(['RET001', 'RET002', 'RET003', 'RET004']),
            'storeZone': zone,
            'storeLocation': f"{city} - {random.choice(['Central', 'East', 'West', 'North', 'South'])}",
            'storeCarpetArea': carpet_area,
            'areaGrading': area_grading,
            'rsmName': rsm_name,
            'rsmEmail': generate_email(rsm_name, 'business'),
            'rsmEmployeeCode': generate_employee_code(),
            'asmName': asm_name,
            'asmEmail': generate_email(asm_name, 'business'),
            'asmEmployeeCode': generate_employee_code(),
            'serverID': server_id,
            'GSTIN': gstin,
            'businessPhone': generate_phone_number(),
            'landline': generate_landline(),
            'removedDetail': random.choice(['Active', 'Active', 'Active', 'Active', 'Temporarily Closed']) if random.random() > 0.9 else 'Active',
            'merchandiser': merchandiser,
            'merchandiser1': merchandiser1,
            'customRegion': f"REG{random.randint(10, 99)}",
            'StoreCode': store_code,
            'storeManager': store_manager,
            'lic': f"LIC{random.randint(100000, 999999)}",
            'emailID': generate_email(store_manager, 'business')
        }
        
        store_data.append(store_record)
    
    return store_data

# Generate the store header dataset
print("Generating Store Header data for 15 shoe stores...")
store_header_data = generate_store_header_data()

# Create DataFrame
df_stores = pd.DataFrame(store_header_data)

# Display summary
print(f"\nStore Header Summary:")
print(f"Total Stores: {len(df_stores)}")
print(f"Store Codes: {', '.join(df_stores['Code'].tolist())}")

print(f"\nCity Distribution:")
print(df_stores['City'].value_counts())

print(f"\nZone Distribution:")
print(df_stores['storeZone'].value_counts())

print(f"\nStore Area Statistics:")
print(f"Average Carpet Area: {df_stores['storeCarpetArea'].mean():.0f} sq ft")
print(f"Min Area: {df_stores['storeCarpetArea'].min()} sq ft")
print(f"Max Area: {df_stores['storeCarpetArea'].max()} sq ft")

print(f"\nArea Grading Distribution:")
print(df_stores['areaGrading'].value_counts())

print(f"\nSample Store Data:")
print(df_stores[['Code', 'BName', 'City', 'State', 'storeZone', 'rsmName', 'storeManager']].head().to_string(index=False))

# Save to CSV
output_file = "store_data.csv"
df_stores.to_csv(output_file, index=False)
print(f"\nStore Header dataset saved to: {output_file}")

# Show detailed view of first 2 stores
print(f"\nDetailed view of first 2 stores:")
for i in range(2):
    print(f"\n=== Store {i+1}: {df_stores.iloc[i]['Code']} ===")
    store_info = df_stores.iloc[i]
    print(f"Business Name: {store_info['BName']}")
    print(f"Address: {store_info['Address']}, {store_info['City']}, {store_info['State']} - {store_info['PostalCode']}")
    print(f"Phone: {store_info['Phone']}")
    print(f"Zone: {store_info['storeZone']}")
    print(f"Carpet Area: {store_info['storeCarpetArea']} sq ft")
    print(f"Area Grading: {store_info['areaGrading']}")
    print(f"RSM: {store_info['rsmName']} ({store_info['rsmEmail']})")
    print(f"ASM: {store_info['asmName']} ({store_info['asmEmail']})")
    print(f"Store Manager: {store_info['storeManager']} ({store_info['emailID']})")
    print(f"GSTIN: {store_info['GSTIN']}")
    print(f"SAP Store Code: {store_info['sapStoreCode']}")

# Verify store codes match with invoice dataset
print(f"\n=== Verification ===")
invoice_store_codes = ['SH001', 'SH002', 'SH003', 'SH004', 'SH005', 'SH006', 'SH007', 'SH008', 'SH009', 'SH010', 
                      'SH011', 'SH012', 'SH013', 'SH014', 'SH015']
header_store_codes = df_stores['Code'].tolist()

print(f"Invoice dataset store codes: {len(invoice_store_codes)} stores")
print(f"Header dataset store codes: {len(header_store_codes)} stores")
print(f"Codes match: {set(invoice_store_codes) == set(header_store_codes)}")

if set(invoice_store_codes) == set(header_store_codes):
    print("✅ All store codes match perfectly between invoice and header datasets!")
else:
    print("❌ Store codes mismatch detected!")
    print(f"Missing in header: {set(invoice_store_codes) - set(header_store_codes)}")
    print(f"Extra in header: {set(header_store_codes) - set(invoice_store_codes)}")